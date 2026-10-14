"""
TrustGate — Asynchronous Multi-Tenant Token Bucket Rate Limiter.

Features:
- Dual-bucket algorithm per client:
  1. Request Bucket (RPM - Requests Per Minute).
  2. Token Bucket (TPM - LLM Tokens Per Minute).
- Generic Priority Tiers (CRITICAL, STANDARD, BATCH).
- Burst capacity handling.
- RFC-compliant RateLimit telemetry & Retry-After calculations.
"""
from __future__ import annotations

import time
import asyncio
from typing import Optional
from pydantic import BaseModel, Field

from core.config import settings, TierConfig


class BucketState(BaseModel):
    """Current state of a Token Bucket."""
    capacity: float
    current_tokens: float
    refill_rate_per_sec: float
    last_refill_timestamp: float


class RateLimitResult(BaseModel):
    """Result of rate limit check."""
    allowed: bool
    tier: str
    remaining_requests: int
    remaining_tokens: int
    retry_after_seconds: float = 0.0
    limit_rpm: int
    limit_tpm: int


class TokenBucketRateLimiter:
    """Asynchronous multi-tenant Token Bucket Rate Limiter."""

    def __init__(self, config=None):
        self.config = config or settings
        self._lock = asyncio.Lock()

        # State storage: client_id -> {"requests": BucketState, "tokens": BucketState}
        self._clients: dict[str, dict[str, BucketState]] = {}

    def _resolve_tier(self, client_id: str) -> TierConfig:
        """Resolve client identifier or API key to a configured TierConfig."""
        # 1. Direct mapping
        tier_name = self.config.key_tier_mapping.get(client_id)
        if not tier_name:
            # Check if client_id contains tier keyword
            cid_lower = client_id.lower()
            if "critical" in cid_lower or "p0" in cid_lower or "ctfr" in cid_lower:
                tier_name = "CRITICAL"
            elif "batch" in cid_lower or "p2" in cid_lower or "front" in cid_lower:
                tier_name = "BATCH"
            else:
                tier_name = "STANDARD"

        return self.config.tiers.get(tier_name, self.config.tiers["STANDARD"])

    def _get_or_create_buckets(self, client_id: str, tier: TierConfig) -> dict[str, BucketState]:
        """Initialize buckets for a client if not already present."""
        now = time.time()
        if client_id not in self._clients:
            # Burst capacity calculation
            burst_req_cap = float(tier.rpm * tier.burst_multiplier)
            burst_tok_cap = float(tier.tpm * tier.burst_multiplier)

            # Refill rates (per second)
            refill_req = tier.rpm / 60.0
            refill_tok = tier.tpm / 60.0

            self._clients[client_id] = {
                "requests": BucketState(
                    capacity=burst_req_cap,
                    current_tokens=burst_req_cap,
                    refill_rate_per_sec=refill_req,
                    last_refill_timestamp=now,
                ),
                "tokens": BucketState(
                    capacity=burst_tok_cap,
                    current_tokens=burst_tok_cap,
                    refill_rate_per_sec=refill_tok,
                    last_refill_timestamp=now,
                ),
            }

        return self._clients[client_id]

    def _refill(self, bucket: BucketState, now: float) -> None:
        """Add tokens accrued since last refill up to capacity."""
        elapsed = now - bucket.last_refill_timestamp
        if elapsed > 0:
            accrued = elapsed * bucket.refill_rate_per_sec
            bucket.current_tokens = min(bucket.capacity, bucket.current_tokens + accrued)
            bucket.last_refill_timestamp = now

    async def check_and_consume(
        self,
        client_id: str,
        estimated_tokens: int = 100,
    ) -> RateLimitResult:
        """
        Check rate limits and consume 1 request token and estimated LLM tokens.
        """
        async with self._lock:
            now = time.time()
            tier = self._resolve_tier(client_id)
            buckets = self._get_or_create_buckets(client_id, tier)

            req_bucket = buckets["requests"]
            tok_bucket = buckets["tokens"]

            # Refill both buckets
            self._refill(req_bucket, now)
            self._refill(tok_bucket, now)

            # Check if sufficient tokens
            req_cost = 1.0
            tok_cost = float(max(1, estimated_tokens))

            has_req = req_bucket.current_tokens >= req_cost
            has_tok = tok_bucket.current_tokens >= tok_cost

            if has_req and has_tok:
                # Consume tokens
                req_bucket.current_tokens -= req_cost
                tok_bucket.current_tokens -= tok_cost

                return RateLimitResult(
                    allowed=True,
                    tier=tier.name,
                    remaining_requests=int(req_bucket.current_tokens),
                    remaining_tokens=int(tok_bucket.current_tokens),
                    retry_after_seconds=0.0,
                    limit_rpm=tier.rpm,
                    limit_tpm=tier.tpm,
                )
            else:
                # Calculate time required to refill needed tokens
                deficit_req = max(0.0, req_cost - req_bucket.current_tokens)
                deficit_tok = max(0.0, tok_cost - tok_bucket.current_tokens)

                time_req = deficit_req / req_bucket.refill_rate_per_sec if deficit_req > 0 else 0.0
                time_tok = deficit_tok / tok_bucket.refill_rate_per_sec if deficit_tok > 0 else 0.0

                retry_after = round(max(time_req, time_tok), 2)

                return RateLimitResult(
                    allowed=False,
                    tier=tier.name,
                    remaining_requests=int(req_bucket.current_tokens),
                    remaining_tokens=int(tok_bucket.current_tokens),
                    retry_after_seconds=max(0.1, retry_after),
                    limit_rpm=tier.rpm,
                    limit_tpm=tier.tpm,
                )

    def reset(self, client_id: Optional[str] = None) -> None:
        """Reset limits for a client or all clients."""
        if client_id:
            self._clients.pop(client_id, None)
        else:
            self._clients.clear()


# Global singleton instance
rate_limiter = TokenBucketRateLimiter()
