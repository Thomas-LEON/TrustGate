"""
TrustGate — API Route Definitions.

Endpoints:
- POST /v1/evaluate: Explainable Risk Scoring with Rubric & Grounding + Gateway Fallback & Cache.
- POST /v1/chat/completions: OpenAI-compatible universal LLM Gateway endpoint.
- GET  /v1/gateway/stats: Live quantitative FinOps & SRE telemetry.
- GET  /health: Comprehensive subsystem health status.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Header, Request, Response

from schemas.request import EvaluateRequest
from schemas.response import (
    EvaluateResponse,
    Decision,
    ConfidenceResult,
    ConfidenceLevel,
    CriteriaBreakdown,
    GroundingResult,
    GroundingMethod,
    AuditMetadata,
)
from schemas.chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatChoice,
    UsageInfo,
)
from core.rubric import (
    get_rubric,
    build_scoring_prompt,
    compute_final_score,
    parse_llm_criteria,
)
from core.grounding import verify_all_citations
from core.gateway import LLMGateway
from core.cache import semantic_cache
from core.limiter import rate_limiter
from core.observability import observability, TransactionMetric
from core.config import settings

router = APIRouter()
gateway = LLMGateway()


async def enforce_rate_limit(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
    estimated_tokens: int = 200,
) -> tuple[str, str]:
    """Extract client key and enforce token bucket rate limit."""
    # Resolve client key
    client_key = "anonymous-client"
    if x_api_key:
        client_key = x_api_key
    elif authorization and authorization.startswith("Bearer "):
        client_key = authorization.replace("Bearer ", "").strip()

    limit_res = await rate_limiter.check_and_consume(client_key, estimated_tokens=estimated_tokens)

    if not limit_res.allowed:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded (Token Bucket)",
                "tier": limit_res.tier,
                "retry_after_seconds": limit_res.retry_after_seconds,
                "limit_rpm": limit_res.limit_rpm,
                "limit_tpm": limit_res.limit_tpm,
            },
            headers={"Retry-After": str(int(limit_res.retry_after_seconds) + 1)},
        )

    return client_key, limit_res.tier


@router.post(
    "/v1/evaluate",
    response_model=EvaluateResponse,
    summary="Evaluate context with auditable LLM scoring & Gateway resilience",
    description=(
        "Sends context through the Semantic Cache, passes through the Resilient LLM Gateway, "
        "enforces a structured evaluation rubric, verifies citations, and records FinOps metrics."
    ),
)
async def evaluate(
    request: EvaluateRequest,
    response: Response,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> EvaluateResponse:
    """Core audited evaluation endpoint with cache & multi-provider gateway."""
    t0 = time.perf_counter()

    # 1. Rate Limiting Check
    client_key, client_tier = await enforce_rate_limit(
        x_api_key=x_api_key,
        authorization=authorization,
        estimated_tokens=len(request.context) // 4 + 300,
    )

    task_type = request.task_type.value
    rubric = get_rubric(task_type)
    prompt = build_scoring_prompt(request.context, task_type)

    # 2. Semantic Cache Check
    cache_match = semantic_cache.get(prompt, threshold=settings.cache_similarity_threshold)
    if cache_match:
        entry, match_score, match_type = cache_match
        latency_ms = int((time.perf_counter() - t0) * 1000)

        # Parse cached JSON content
        try:
            cached_data = json.loads(entry.response_content)
            cached_response = EvaluateResponse(**cached_data)

            # Update latency & mark cached
            cached_response.audit_metadata.latency_ms = latency_ms
            cached_response.audit_metadata.cached = True
            cached_response.audit_metadata.timestamp = datetime.now(timezone.utc).isoformat()

            # Record telemetry
            observability.record_transaction(
                TransactionMetric(
                    endpoint="/v1/evaluate",
                    client_tier=client_tier,
                    client_id=client_key,
                    cached=True,
                    cache_type=match_type,
                    model_used=entry.model_origin,
                    logical_tier="CACHE",
                    fallback_occurred=False,
                    total_tokens=entry.total_tokens,
                    cost_usd=0.0,
                    cost_saved_usd=entry.estimated_cost_usd,
                    latency_ms=latency_ms,
                    status_code=200,
                )
            )

            # Set response headers
            response.headers["X-Cache"] = f"HIT-{match_type} (score={match_score})"
            return cached_response
        except Exception:
            # If cached parsing fails, proceed to live execution
            pass

    # 3. Cache Miss: Execute via Resilient LLM Gateway
    response.headers["X-Cache"] = "MISS"
    try:
        gw_resp = await gateway.generate(
            prompt=prompt,
            temperature=0.3,
        )
    except Exception as exc:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        observability.record_transaction(
            TransactionMetric(
                endpoint="/v1/evaluate",
                client_tier=client_tier,
                client_id=client_key,
                status_code=502,
                error=str(exc)[:200],
                latency_ms=latency_ms,
            )
        )
        raise HTTPException(status_code=502, detail=f"LLM Gateway failure: {exc}")

    # 4. Parse LLM criteria
    llm_criteria = parse_llm_criteria(gw_resp.content)
    if not llm_criteria:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to parse LLM criteria JSON. Raw response: {gw_resp.content[:500]}",
        )

    # 5. Grounding verification
    llm_criteria = verify_all_citations(
        llm_criteria,
        request.context,
        threshold=request.config.grounding_threshold,
    )

    # 6. Deterministic score recalculation
    computed_score, severity_label = compute_final_score(llm_criteria, rubric)

    # 7. Build rubric breakdown
    hallucination_flags = sum(
        1 for c in llm_criteria if not c.get("grounding", {}).get("verified", False)
    )

    from rapidfuzz import fuzz

    breakdown = []
    for c in llm_criteria:
        c_name = c.get("criteria", "Unknown")
        c_norm = c_name.strip().lower()

        # 1. Exact match
        rubric_entry = next((r for r in rubric if r["criteria"].strip().lower() == c_norm), None)

        # 2. Substring match
        if not rubric_entry:
            rubric_entry = next(
                (r for r in rubric if r["criteria"].strip().lower() in c_norm or c_norm in r["criteria"].strip().lower()),
                None,
            )

        # 3. Fuzzy match
        if not rubric_entry:
            best_r = max(rubric, key=lambda r: fuzz.token_set_ratio(c_norm, r["criteria"].lower()), default=None)
            if best_r and fuzz.token_set_ratio(c_norm, best_r["criteria"].lower()) >= 50:
                rubric_entry = best_r

        weight = rubric_entry["weight"] if rubric_entry else 0.0
        max_pts = rubric_entry["max_points"] if rubric_entry else max(1, int(c.get("score", 10)))
        max_pts = max(1, max_pts)  # Guarantee max_points >= 1 for Pydantic validation

        grounding_data = c.get("grounding", {"verified": False, "match_ratio": 0.0, "method": "NOT_FOUND"})

        breakdown.append(
            CriteriaBreakdown(
                criteria=c_name,
                score=min(max(int(c.get("score", 0)), 0), max_pts),
                max_points=max_pts,
                weight=weight,
                justification=c.get("justification", ""),
                source_quote=c.get("source_quote", ""),
                grounding=GroundingResult(
                    verified=grounding_data.get("verified", False),
                    match_ratio=grounding_data.get("match_ratio", 0.0),
                    method=GroundingMethod(grounding_data.get("method", "NOT_FOUND")),
                ),
            )
        )

    latency_ms = int((time.perf_counter() - t0) * 1000)

    final_response = EvaluateResponse(
        decision=Decision(
            computed_score=computed_score,
            scale=[0, 100],
            severity_label=severity_label,
            confidence=ConfidenceResult(
                level=ConfidenceLevel.NOT_EVALUATED,
                variance=None,
                runs_evaluated=1,
            ),
        ),
        rubric_breakdown=breakdown,
        audit_metadata=AuditMetadata(
            model=gw_resp.model_used,
            logical_tier=gw_resp.logical_tier,
            fallback_occurred=gw_resp.fallback_occurred,
            cached=False,
            hallucination_flags=hallucination_flags,
            unverified_citations=hallucination_flags,
            latency_ms=latency_ms,
            total_tokens=gw_resp.total_tokens,
            cost_usd=gw_resp.cost_usd,
            timestamp=datetime.now(timezone.utc).isoformat(),
        ),
    )

    # 8. Store in Semantic Cache
    semantic_cache.set(
        prompt=prompt,
        response_content=final_response.model_dump_json(),
        model=gw_resp.model_used,
        prompt_tokens=gw_resp.prompt_tokens,
        completion_tokens=gw_resp.completion_tokens,
        total_tokens=gw_resp.total_tokens,
        cost_usd=gw_resp.cost_usd,
    )

    # 9. Record Telemetry
    observability.record_transaction(
        TransactionMetric(
            endpoint="/v1/evaluate",
            client_tier=client_tier,
            client_id=client_key,
            cached=False,
            model_used=gw_resp.model_used,
            logical_tier=gw_resp.logical_tier,
            fallback_occurred=gw_resp.fallback_occurred,
            prompt_tokens=gw_resp.prompt_tokens,
            completion_tokens=gw_resp.completion_tokens,
            total_tokens=gw_resp.total_tokens,
            cost_usd=gw_resp.cost_usd,
            latency_ms=latency_ms,
            status_code=200,
            unverified_citations=hallucination_flags,
            hallucination_detected=(hallucination_flags > 0),
        )
    )

    return final_response


@router.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    summary="Universal OpenAI-compatible chat completion",
    description="Drop-in replacement for OpenAI API with tiered fallback, rate limiting, and caching.",
)
async def chat_completions(
    request: ChatCompletionRequest,
    response: Response,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> ChatCompletionResponse:
    """Universal OpenAI-compatible endpoint."""
    if request.stream:
        raise HTTPException(
            status_code=501,
            detail="Streaming responses are not supported in this version.",
        )

    t0 = time.perf_counter()

    # Rate limiting
    client_key, client_tier = await enforce_rate_limit(
        x_api_key=x_api_key,
        authorization=authorization,
        estimated_tokens=sum(len(m.content) // 4 for m in request.messages) + 200,
    )

    # Extract last user message for semantic cache check
    last_user_msg = next((m.content for m in reversed(request.messages) if m.role == "user"), "")

    if last_user_msg:
        cache_match = semantic_cache.get(last_user_msg)
        if cache_match:
            entry, match_score, match_type = cache_match
            latency_ms = int((time.perf_counter() - t0) * 1000)

            response.headers["X-Cache"] = f"HIT-{match_type} (score={match_score})"
            observability.record_transaction(
                TransactionMetric(
                    endpoint="/v1/chat/completions",
                    client_tier=client_tier,
                    client_id=client_key,
                    cached=True,
                    cache_type=match_type,
                    model_used=entry.model_origin,
                    logical_tier="CACHE",
                    total_tokens=entry.total_tokens,
                    cost_usd=0.0,
                    cost_saved_usd=entry.estimated_cost_usd,
                    latency_ms=latency_ms,
                    status_code=200,
                )
            )

            return ChatCompletionResponse(
                model=entry.model_origin,
                choices=[ChatChoice(message=ChatMessage(role="assistant", content=entry.response_content))],
                usage=UsageInfo(
                    prompt_tokens=entry.prompt_tokens,
                    completion_tokens=entry.completion_tokens,
                    total_tokens=entry.total_tokens,
                ),
                gateway_metadata={
                    "cached": True,
                    "match_type": match_type,
                    "similarity_score": match_score,
                    "cost_saved_usd": entry.estimated_cost_usd,
                    "latency_ms": latency_ms,
                },
            )

    # Cache miss -> Live call through gateway
    response.headers["X-Cache"] = "MISS"
    messages_payload = [{"role": m.role, "content": m.content} for m in request.messages]

    try:
        gw_resp = await gateway.chat(
            messages=messages_payload,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            preferred_model=request.model,
        )
    except Exception as exc:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        observability.record_transaction(
            TransactionMetric(
                endpoint="/v1/chat/completions",
                client_tier=client_tier,
                client_id=client_key,
                status_code=502,
                error=str(exc)[:200],
                latency_ms=latency_ms,
            )
        )
        raise HTTPException(status_code=502, detail=f"Gateway execution failure: {exc}")

    latency_ms = int((time.perf_counter() - t0) * 1000)

    # Store in semantic cache
    if last_user_msg:
        semantic_cache.set(
            prompt=last_user_msg,
            response_content=gw_resp.content,
            model=gw_resp.model_used,
            prompt_tokens=gw_resp.prompt_tokens,
            completion_tokens=gw_resp.completion_tokens,
            total_tokens=gw_resp.total_tokens,
            cost_usd=gw_resp.cost_usd,
        )

    # Record Telemetry
    observability.record_transaction(
        TransactionMetric(
            endpoint="/v1/chat/completions",
            client_tier=client_tier,
            client_id=client_key,
            cached=False,
            model_used=gw_resp.model_used,
            logical_tier=gw_resp.logical_tier,
            fallback_occurred=gw_resp.fallback_occurred,
            prompt_tokens=gw_resp.prompt_tokens,
            completion_tokens=gw_resp.completion_tokens,
            total_tokens=gw_resp.total_tokens,
            cost_usd=gw_resp.cost_usd,
            latency_ms=latency_ms,
            status_code=200,
        )
    )

    return ChatCompletionResponse(
        model=gw_resp.model_used,
        choices=[ChatChoice(message=ChatMessage(role="assistant", content=gw_resp.content))],
        usage=UsageInfo(
            prompt_tokens=gw_resp.prompt_tokens,
            completion_tokens=gw_resp.completion_tokens,
            total_tokens=gw_resp.total_tokens,
        ),
        gateway_metadata={
            "cached": False,
            "logical_tier": gw_resp.logical_tier,
            "fallback_occurred": gw_resp.fallback_occurred,
            "latency_ms": latency_ms,
            "cost_usd": gw_resp.cost_usd,
        },
    )


@router.get("/v1/gateway/stats", summary="Get live FinOps and SRE Gateway statistics")
async def get_gateway_stats():
    """Returns real-time aggregated metrics (P50/P95 latency, tokens, costs, cache hit rate)."""
    return {
        "gateway_status": "ONLINE",
        "cache_stats": semantic_cache.stats(),
        "finops_telemetry": observability.get_aggregated_stats(),
        "configured_tiers": {
            name: {"rpm": tier.rpm, "tpm": tier.tpm}
            for name, tier in settings.tiers.items()
        },
    }


@router.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "TrustGate Enterprise LLM Gateway",
        "version": "1.0.0",
        "primary_model": settings.model_primary,
        "secondary_model": settings.model_secondary,
        "tertiary_model": settings.model_tertiary,
        "cache_entries": len(semantic_cache._store),
    }
