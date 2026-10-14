"""
End-to-End API integration tests for TrustGate.
"""
import os
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient
from main import app
from core.cache import semantic_cache
from core.limiter import rate_limiter

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "1.0.0"
    assert "primary_model" in data


def test_gateway_stats_endpoint():
    response = client.get("/v1/gateway/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["gateway_status"] == "ONLINE"
    assert "cache_stats" in data
    assert "finops_telemetry" in data


def test_chat_completions_mock_and_cache(monkeypatch):
    """Test /v1/chat/completions flow: 1st call executes & caches, 2nd call hits cache with 0 cost."""
    semantic_cache.clear()
    rate_limiter.reset()

    class MockGatewayResp:
        content = "Hello! TrustGate gateway is functioning properly."
        model_used = "gemini/gemini-3.8-flash"
        logical_tier = "PRIMARY"
        fallback_occurred = False
        prompt_tokens = 15
        completion_tokens = 12
        total_tokens = 27
        cost_usd = 0.000015

    async def mock_chat(*args, **kwargs):
        return MockGatewayResp()

    from api import routes
    monkeypatch.setattr(routes.gateway, "chat", mock_chat)

    payload = {
        "messages": [{"role": "user", "content": "What is the status of the gateway?"}],
        "temperature": 0.2,
    }

    # 1. First call -> Cache MISS
    res1 = client.post("/v1/chat/completions", json=payload, headers={"X-API-Key": "key-ctfr"})
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["choices"][0]["message"]["content"] == MockGatewayResp.content
    assert res1.headers.get("X-Cache") == "MISS"

    # 2. Second identical call -> Cache HIT
    res2 = client.post("/v1/chat/completions", json=payload, headers={"X-API-Key": "key-ctfr"})
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["choices"][0]["message"]["content"] == MockGatewayResp.content
    assert "HIT" in res2.headers.get("X-Cache", "")
    assert data2["gateway_metadata"]["cached"] is True
    print("\n[SUCCESS] Chat completions cache HIT confirmed!")


def test_rate_limiting_enforcement(monkeypatch):
    """Verify that exceeding rate limits returns HTTP 429."""
    rate_limiter.reset()

    class MockGatewayResp:
        content = "pong"
        model_used = "gemini/gemini-3.8-flash"
        logical_tier = "PRIMARY"
        fallback_occurred = False
        prompt_tokens = 5
        completion_tokens = 5
        total_tokens = 10
        cost_usd = 0.000001

    async def mock_chat(*args, **kwargs):
        return MockGatewayResp()

    from api import routes
    monkeypatch.setattr(routes.gateway, "chat", mock_chat)

    # Send multiple requests with small batch key until 429
    limited = False
    for _ in range(40):
        r = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "ping"}]},
            headers={"X-API-Key": "key-front"},
        )
        if r.status_code == 429:
            limited = True
            assert "Retry-After" in r.headers
            break

    assert limited, "Expected client to be rate limited"
    print("\n[SUCCESS] Token Bucket rate limiting 429 enforcement confirmed!")


def test_stream_not_implemented():
    """Verify that stream=True returns HTTP 501."""
    res = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
        headers={"X-API-Key": "key-ctfr"},
    )
    assert res.status_code == 501
    assert "Streaming responses are not supported" in res.json().get("detail", "")


def test_evaluate_endpoint_mock(monkeypatch):
    """Test full /v1/evaluate pipeline with mocked LLM criteria generation."""
    rate_limiter.reset()

    class MockEvaluateGwResp:
        content = """[
            {
                "criteria": "Technical Impact",
                "score": 35,
                "justification": "Remote code execution allows full takeover.",
                "source_quote": "allows remote code execution without authentication"
            },
            {
                "criteria": "Exploit Availability",
                "score": 25,
                "justification": "Exploitation scripts exist.",
                "source_quote": "Exploitation scripts are actively circulating"
            },
            {
                "criteria": "Exposure & Scope",
                "score": 10,
                "justification": "Isolated VLAN limits blast radius.",
                "source_quote": "isolated VLAN with no direct internet access"
            }
        ]"""
        model_used = "gemini/gemini-3.8-flash"
        logical_tier = "PRIMARY"
        fallback_occurred = False
        prompt_tokens = 150
        completion_tokens = 100
        total_tokens = 250
        cost_usd = 0.0001

    async def mock_generate(*args, **kwargs):
        return MockEvaluateGwResp()

    from api import routes
    monkeypatch.setattr(routes.gateway, "generate", mock_generate)

    payload = {
        "context": "A critical vulnerability CVE-2026-1104 in OpenSSL allows remote code execution without authentication. Exploitation scripts are actively circulating on public repositories. The impacted service runs inside an isolated VLAN with no direct internet access.",
        "task_type": "THREAT_SEVERITY",
        "config": {"sampling_runs": 1, "grounding_threshold": 0.85},
    }

    res = client.post("/v1/evaluate", json=payload, headers={"X-API-Key": "key-ctfr"})
    assert res.status_code == 200
    data = res.json()
    assert "decision" in data
    assert "computed_score" in data["decision"]
    assert len(data["rubric_breakdown"]) == 3
    assert data["audit_metadata"]["logical_tier"] == "PRIMARY"
    assert data["audit_metadata"]["model"] == "gemini/gemini-3.8-flash"


if __name__ == "__main__":
    test_health_endpoint()
    test_gateway_stats_endpoint()
    test_stream_not_implemented()
    print("\nAll standalone API tests passed.")
