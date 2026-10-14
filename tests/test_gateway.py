"""
Unit tests for TrustGate LLM Gateway and fallback cascade.
"""
import os
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import pytest
from core.gateway import LLMGateway, GatewayResponse
from core.config import settings


@pytest.mark.asyncio
async def test_gateway_initialization():
    gateway = LLMGateway()
    assert len(gateway.tiers) == 3
    assert gateway.tiers[0][0] == "PRIMARY"
    assert gateway.tiers[1][0] == "SECONDARY"
    assert gateway.tiers[2][0] == "TERTIARY"


@pytest.mark.asyncio
async def test_gateway_fallback_simulation():
    """Verify that if PRIMARY model fails all retries, gateway automatically falls back to SECONDARY."""
    gateway = LLMGateway()

    import litellm

    class MockChoice:
        class MockMessage:
            content = '{"score": 90, "justification": "Fallback succeeded"}'
        message = MockMessage()

    class MockUsage:
        prompt_tokens = 50
        completion_tokens = 25
        total_tokens = 75

    class MockLiteLLMResponse:
        choices = [MockChoice()]
        usage = MockUsage()

    async def mock_acompletion(model, messages, temperature=0.3):
        # Primary model persistently fails
        if model == settings.model_primary:
            raise Exception("500 Internal Server Error: Primary model down")
        # Secondary succeeds
        return MockLiteLLMResponse()

    litellm.acompletion = mock_acompletion

    resp = await gateway.generate("Test prompt")
    assert isinstance(resp, GatewayResponse)
    assert resp.fallback_occurred is True
    assert resp.logical_tier == "SECONDARY"
    assert resp.model_used == settings.model_secondary
    assert len(resp.attempts) == 2
    assert resp.attempts[0].status == "FAILED"
    assert resp.attempts[1].status == "SUCCESS"
    print("\n[SUCCESS] Fallback test PASSED: successfully fell back from PRIMARY to SECONDARY!")


if __name__ == "__main__":
    asyncio.run(test_gateway_initialization())
    asyncio.run(test_gateway_fallback_simulation())
