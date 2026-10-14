"""Smoke and unit tests for TrustGate core engines."""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.rubric import build_scoring_prompt, compute_final_score, get_rubric
from core.grounding import verify_all_citations


def test_rubric_generation():
    """Verify rubric retrieval and prompt generation."""
    rubric = get_rubric("THREAT_SEVERITY")
    assert len(rubric) == 3
    names = [c["criteria"] for c in rubric]
    assert "Technical Impact" in names
    assert "Exploit Availability" in names
    assert "Exposure & Scope" in names

    prompt = build_scoring_prompt("Test vulnerability in OpenSSL.", "THREAT_SEVERITY")
    assert len(prompt) > 100
    assert "OpenSSL" in prompt


def test_grounding_verification():
    """Verify exact and fuzzy citation matching."""
    context = "A critical vulnerability CVE-2026-1104 in OpenSSL allows remote code execution without authentication."
    fake_criteria = [
        {
            "criteria": "Technical Impact",
            "score": 35,
            "justification": "RCE",
            "source_quote": "remote code execution without authentication",
        },
        {
            "criteria": "Exploit Availability",
            "score": 10,
            "justification": "unknown",
            "source_quote": "this is a hallucinated quote that does not exist anywhere",
        },
    ]

    verified = verify_all_citations(fake_criteria, context)
    assert len(verified) == 2
    assert verified[0]["grounding"]["verified"] is True
    assert verified[1]["grounding"]["verified"] is False


def test_deterministic_score_computation():
    """Verify weighted score calculation."""
    rubric = get_rubric("THREAT_SEVERITY")
    verified = [
        {
            "criteria": "Technical Impact",
            "score": 40,
            "max_points": 40,
            "grounding": {"verified": True},
        },
        {
            "criteria": "Exploit Availability",
            "score": 30,
            "max_points": 30,
            "grounding": {"verified": True},
        },
        {
            "criteria": "Exposure & Scope",
            "score": 30,
            "max_points": 30,
            "grounding": {"verified": True},
        },
    ]
    score, label = compute_final_score(verified, rubric)
    assert score == 100
    assert label == "CRITICAL"
