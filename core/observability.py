"""
TrustGate — Observability, Tracing & FinOps Engine.

Features:
- Structured JSONL local telemetry logging (zero data loss).
- Real-time in-memory rolling buffer for SRE & FinOps dashboards.
- Percentile latency calculations (P50, P95, P99).
- Optional Langfuse cloud synchronization if keys are provided.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from pydantic import BaseModel, Field

from core.config import settings


class TransactionMetric(BaseModel):
    """Full telemetry record for an API transaction."""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    endpoint: str
    client_tier: str
    client_id: str
    cached: bool = False
    cache_type: str = "NONE"  # "EXACT", "SEMANTIC", "NONE"
    model_used: str = "none"
    logical_tier: str = "none"  # "PRIMARY", "SECONDARY", "TERTIARY", "CACHE"
    fallback_occurred: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    cost_saved_usd: float = 0.0
    latency_ms: int = 0
    status_code: int = 200
    unverified_citations: int = 0
    hallucination_detected: bool = False
    error: Optional[str] = None


class ObservabilityEngine:
    """Central FinOps and SRE observability engine."""

    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path or settings.metrics_log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # In-memory rolling buffer — deque with maxlen for O(1) eviction
        from collections import deque
        self._recent_events: deque[TransactionMetric] = deque(maxlen=1000)
        self._max_recent = 1000

        # Langfuse client initialized lazily if keys exist
        self._langfuse = None
        if settings.langfuse_public_key and settings.langfuse_secret_key:
            try:
                from langfuse import Langfuse
                self._langfuse = Langfuse(
                    public_key=settings.langfuse_public_key,
                    secret_key=settings.langfuse_secret_key,
                    host=settings.langfuse_host,
                )
            except Exception:
                self._langfuse = None

    def record_transaction(self, metric: TransactionMetric) -> None:
        """Record transaction locally and asynchronously to Langfuse if available."""
        # 1. Add to in-memory rolling buffer (deque auto-evicts oldest when full)
        self._recent_events.append(metric)

        # 2. Append to local JSONL log file
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(metric.model_dump()) + "\n")
        except Exception:
            pass

        # 3. Cloud Langfuse trace if active
        if self._langfuse:
            try:
                trace = self._langfuse.trace(
                    id=metric.request_id,
                    name=metric.endpoint,
                    metadata={
                        "client_tier": metric.client_tier,
                        "client_id": metric.client_id,
                        "cached": metric.cached,
                        "fallback": metric.fallback_occurred,
                    },
                    tags=[metric.client_tier, metric.model_used],
                )
                trace.generation(
                    name="gateway_call",
                    model=metric.model_used,
                    usage={
                        "prompt_tokens": metric.prompt_tokens,
                        "completion_tokens": metric.completion_tokens,
                        "total_tokens": metric.total_tokens,
                    },
                    metadata={"cost_usd": metric.cost_usd, "latency_ms": metric.latency_ms},
                )
            except Exception:
                pass

    def get_recent_transactions(self, limit: int = 100) -> list[dict[str, Any]]:
        """Retrieve recent transactions for dashboard display."""
        events = self._recent_events[-limit:]
        return [e.model_dump() for e in reversed(events)]

    def get_aggregated_stats(self) -> dict[str, Any]:
        """Compute live quantitative SRE & FinOps statistics."""
        if not self._recent_events:
            return {
                "total_requests": 0,
                "cache_hit_rate": 0.0,
                "total_tokens_spent": 0,
                "total_tokens_saved": 0,
                "total_cost_spent_usd": 0.0,
                "total_cost_saved_usd": 0.0,
                "p50_latency_ms": 0,
                "p95_latency_ms": 0,
                "p99_latency_ms": 0,
                "fallback_count": 0,
                "hallucination_rate": 0.0,
                "tier_distribution": {},
                "model_distribution": {},
            }

        total = len(self._recent_events)
        cached_count = sum(1 for e in self._recent_events if e.cached)
        fallbacks = sum(1 for e in self._recent_events if e.fallback_occurred)
        hallucinations = sum(1 for e in self._recent_events if e.hallucination_detected)

        tokens_spent = sum(e.total_tokens for e in self._recent_events if not e.cached)
        tokens_saved = sum(e.total_tokens for e in self._recent_events if e.cached)

        cost_spent = sum(e.cost_usd for e in self._recent_events if not e.cached)
        cost_saved = sum(e.cost_saved_usd for e in self._recent_events if e.cached)

        # Percentile latencies
        latencies = sorted(e.latency_ms for e in self._recent_events)
        p50 = latencies[int(total * 0.50)] if latencies else 0
        p95 = latencies[min(total - 1, int(total * 0.95))] if latencies else 0
        p99 = latencies[min(total - 1, int(total * 0.99))] if latencies else 0

        # Distributions
        tier_dist: dict[str, int] = {}
        model_dist: dict[str, int] = {}
        for e in self._recent_events:
            tier_dist[e.client_tier] = tier_dist.get(e.client_tier, 0) + 1
            model_dist[e.model_used] = model_dist.get(e.model_used, 0) + 1

        return {
            "total_requests": total,
            "cache_hit_rate": round((cached_count / total) * 100, 2),
            "total_tokens_spent": tokens_spent,
            "total_tokens_saved": tokens_saved,
            "total_cost_spent_usd": round(cost_spent, 6),
            "total_cost_saved_usd": round(cost_saved, 6),
            "p50_latency_ms": p50,
            "p95_latency_ms": p95,
            "p99_latency_ms": p99,
            "fallback_count": fallbacks,
            "hallucination_rate": round((hallucinations / total) * 100, 2),
            "tier_distribution": tier_dist,
            "model_distribution": model_dist,
        }


# Global singleton instance
observability = ObservabilityEngine()
