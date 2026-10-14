"""
Unit tests for TrustGate Token Bucket Rate Limiter.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import pytest
from core.limiter import TokenBucketRateLimiter
from core.config import settings, TierConfig


@pytest.mark.asyncio
async def test_token_bucket_normal_and_burst():
    # Setup custom limiter with very tight limits for testing
    class TestSettings:
        key_tier_mapping = {"key-fast": "CRITICAL", "key-slow": "BATCH"}
        tiers = {
            "CRITICAL": TierConfig(name="CRITICAL", rpm=60, tpm=1000, burst_multiplier=1.0),
            "BATCH": TierConfig(name="BATCH", rpm=2, tpm=100, burst_multiplier=1.0),
            "STANDARD": TierConfig(name="STANDARD", rpm=10, tpm=500, burst_multiplier=1.0),
        }

    limiter = TokenBucketRateLimiter(config=TestSettings())

    # 1. Fast client can make immediate request
    res1 = await limiter.check_and_consume("key-fast", estimated_tokens=100)
    assert res1.allowed is True
    assert res1.tier == "CRITICAL"

    # 2. Slow client has capacity = 2 requests
    s1 = await limiter.check_and_consume("key-slow", estimated_tokens=10)
    assert s1.allowed is True
    s2 = await limiter.check_and_consume("key-slow", estimated_tokens=10)
    assert s2.allowed is True

    # 3. Third request should be REJECTED (capacity = 2)
    s3 = await limiter.check_and_consume("key-slow", estimated_tokens=10)
    assert s3.allowed is False
    assert s3.retry_after_seconds > 0
    print(f"\n[SUCCESS] Rate Limiter correctly rejected 3rd request! Retry-After: {s3.retry_after_seconds}s")


if __name__ == "__main__":
    asyncio.run(test_token_bucket_normal_and_burst())
