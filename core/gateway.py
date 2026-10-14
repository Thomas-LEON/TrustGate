"""
TrustGate — Resilient Multi-Provider LLM Gateway.

Features:
- Asynchronous execution (non-blocking for FastAPI event loop).
- Dynamic routing with automatic fallback across logical tiers (PRIMARY -> SECONDARY -> TERTIARY).
- Transient error retry with exponential backoff and jitter.
- Detailed execution telemetry (latency, token usage, cost in USD, fallback audit trail).
"""
from __future__ import annotations

import asyncio
import os
import random
import time
from typing import Any, Optional
from pydantic import BaseModel, Field

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from core.config import settings
import litellm

# Suppress verbose litellm logs
litellm.suppress_debug_info = True


class AttemptLog(BaseModel):
    """Log of a single model invocation attempt."""
    model: str
    tier: str
    status: str  # "SUCCESS", "FAILED"
    error: Optional[str] = None
    latency_ms: int = 0


class GatewayResponse(BaseModel):
    """Audited execution response from the LLM Gateway."""
    content: str
    model_used: str
    logical_tier: str
    fallback_occurred: bool = False
    attempts: list[AttemptLog] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    cached: bool = False


class LLMGateway:
    """Enterprise LLM Gateway with tiered fallback and resilience."""

    def __init__(self, config=None):
        self.config = config or settings
        # Ordered logical tiers
        self.tiers = [
            ("PRIMARY", self.config.model_primary),
            ("SECONDARY", self.config.model_secondary),
            ("TERTIARY", self.config.model_tertiary),
        ]

    async def _call_model_with_retry(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        max_retries: int = 2,
    ) -> tuple[Any, int]:
        """
        Execute call with exponential backoff & jitter for transient errors.
        """
        last_error = None
        t0 = time.perf_counter()

        for attempt in range(max_retries + 1):
            try:
                call_kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                }
                if max_tokens is not None:
                    call_kwargs["max_tokens"] = max_tokens

                # Call LiteLLM asynchronously
                response = await litellm.acompletion(**call_kwargs)
                elapsed_ms = int((time.perf_counter() - t0) * 1000)
                return response, elapsed_ms
            except Exception as exc:
                last_error = exc
                err_str = str(exc).lower()

                # Check if error is transient (429 rate limit, 500/503 server error, timeout)
                is_transient = any(
                    code in err_str for code in ["429", "rate limit", "500", "503", "timeout", "overloaded"]
                )

                if is_transient and attempt < max_retries:
                    # Exponential backoff: 0.5s, 1.0s + jitter
                    sleep_time = (0.5 * (2 ** attempt)) + random.uniform(0.05, 0.25)
                    await asyncio.sleep(sleep_time)
                    continue
                else:
                    break

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        raise RuntimeError(f"Model {model} failed: {last_error}") from last_error

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        preferred_model: Optional[str] = None,
    ) -> GatewayResponse:
        """
        Chat completion through the gateway with automatic cascade fallback.
        """
        overall_t0 = time.perf_counter()
        attempts: list[AttemptLog] = []

        # If a preferred model is specified, try it first
        candidate_tiers = list(self.tiers)
        if preferred_model:
            candidate_tiers.insert(0, ("CUSTOM_OVERRIDE", preferred_model))

        last_exception = None

        for tier_name, model_id in candidate_tiers:
            if not model_id:
                continue

            try:
                raw_response, call_latency_ms = await self._call_model_with_retry(
                    model=model_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                # Extract response text
                content = raw_response.choices[0].message.content or ""

                # Extract tokens & calculate cost
                usage = getattr(raw_response, "usage", None)
                prompt_tokens = usage.prompt_tokens if usage else 0
                completion_tokens = usage.completion_tokens if usage else 0
                total_tokens = usage.total_tokens if usage else (prompt_tokens + completion_tokens)

                # Cost estimation via LiteLLM
                cost_usd = 0.0
                try:
                    cost_usd = float(litellm.completion_cost(raw_response))
                except Exception:
                    # Fallback cost calculation: ~$0.15 / 1M prompt, $0.60 / 1M completion
                    cost_usd = round(
                        (prompt_tokens * 0.00000015) + (completion_tokens * 0.00000060), 6
                    )

                overall_latency = int((time.perf_counter() - overall_t0) * 1000)
                attempts.append(
                    AttemptLog(
                        model=model_id,
                        tier=tier_name,
                        status="SUCCESS",
                        latency_ms=call_latency_ms,
                    )
                )

                return GatewayResponse(
                    content=content,
                    model_used=model_id,
                    logical_tier=tier_name,
                    fallback_occurred=(tier_name != candidate_tiers[0][0]),
                    attempts=attempts,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cost_usd=cost_usd,
                    latency_ms=overall_latency,
                    cached=False,
                )

            except Exception as exc:
                last_exception = exc
                attempts.append(
                    AttemptLog(
                        model=model_id,
                        tier=tier_name,
                        status="FAILED",
                        error=str(exc)[:200],
                        latency_ms=0,
                    )
                )
                continue

        # All tiers exhausted
        overall_latency = int((time.perf_counter() - overall_t0) * 1000)
        raise RuntimeError(
            f"All gateway models exhausted. Attempts: {[a.model_dump() for a in attempts]}. Last error: {last_exception}"
        )

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
    ) -> GatewayResponse:
        """Convenience method for prompt-based completions."""
        messages = [{"role": "user", "content": prompt}]
        return await self.chat(messages, temperature=temperature, max_tokens=max_tokens)
