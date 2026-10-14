"""
Unit tests for TrustGate In-Memory Semantic Cache.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.cache import SemanticCache


def test_semantic_cache_exact_and_similarity():
    cache = SemanticCache(similarity_threshold=0.85)

    prompt_orig = "A critical vulnerability CVE-2026-1104 in OpenSSL allows remote code execution without authentication."
    mock_resp = '{"decision": "HIGH", "score": 85}'

    # 1. Store response
    cache.set(
        prompt=prompt_orig,
        response_content=mock_resp,
        model="gemini/gemini-3.6-flash",
        prompt_tokens=100,
        completion_tokens=50,
        cost_usd=0.00045,
    )

    assert len(cache._store) == 1

    # 2. Exact match lookup
    match = cache.get(prompt_orig)
    assert match is not None
    entry, score, match_type = match
    assert match_type == "EXACT"
    assert score == 1.0
    assert entry.response_content == mock_resp

    # 3. Semantic similarity lookup (rephrased query with same tokens)
    rephrased = "Critical vulnerability CVE-2026-1104 in OpenSSL allows remote code execution without authentication"
    sem_match = cache.get(rephrased)
    assert sem_match is not None
    s_entry, s_score, s_type = sem_match
    assert s_score >= 0.85
    assert s_entry.response_content == mock_resp

    # 4. Completely unrelated query (should MISS)
    unrelated = "Quarterly financial results for retail banking department showed 12 percent growth."
    assert cache.get(unrelated) is None

    # 5. Check stats
    stats = cache.stats()
    assert stats["total_queries"] == 3
    assert stats["total_hits"] == 2
    assert stats["total_tokens_saved"] == 300  # 150 * 2 hits
    assert stats["total_cost_saved_usd"] > 0
    print("\n[SUCCESS] Semantic cache tests PASSED! Hit rate:", stats["hit_rate_pct"], "%")


if __name__ == "__main__":
    test_semantic_cache_exact_and_similarity()
