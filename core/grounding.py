"""
TrustGate — Grounding & Citation Verifier.

Verifies that every source_quote returned by the LLM
actually exists in the original input context.
"""
from __future__ import annotations

from rapidfuzz import fuzz


def verify_citation(
    quote: str,
    context: str,
    threshold: float = 0.85,
) -> dict:
    """
    Verify a single citation against the source context.

    Returns:
        {"verified": bool, "match_ratio": float, "method": str}
    """
    # Handle explicit "no evidence" marker
    if not quote or quote.strip().upper() == "NO_EVIDENCE":
        return {"verified": False, "match_ratio": 0.0, "method": "NOT_FOUND"}

    quote_clean = quote.strip()
    context_clean = context.strip()

    # 1. Exact substring match
    if quote_clean in context_clean:
        return {"verified": True, "match_ratio": 1.0, "method": "EXACT"}

    # Case-insensitive exact match
    if quote_clean.lower() in context_clean.lower():
        return {"verified": True, "match_ratio": 1.0, "method": "EXACT"}

    # 2. Fuzzy match — use Token Set Ratio for robustness
    #    (handles word reordering, extra punctuation, truncation)
    ratio = fuzz.token_set_ratio(quote_clean, context_clean) / 100.0

    if ratio >= threshold:
        return {"verified": True, "match_ratio": round(ratio, 3), "method": "FUZZY"}

    # 3. Sliding window fuzzy match
    #    For long contexts, compare the quote against overlapping windows
    #    of similar length in the context.
    quote_len = len(quote_clean)
    if quote_len > 10 and len(context_clean) > quote_len:
        best_ratio = 0.0
        step = max(1, quote_len // 4)

        for i in range(0, len(context_clean) - quote_len + 1, step):
            window = context_clean[i : i + quote_len + 20]  # slight oversize for tolerance
            window_ratio = fuzz.ratio(quote_clean.lower(), window.lower()) / 100.0
            best_ratio = max(best_ratio, window_ratio)

            # Early exit if we find a strong match
            if best_ratio >= threshold:
                return {
                    "verified": True,
                    "match_ratio": round(best_ratio, 3),
                    "method": "FUZZY",
                }

        if best_ratio >= threshold:
            return {"verified": True, "match_ratio": round(best_ratio, 3), "method": "FUZZY"}

    # 4. No match found
    return {"verified": False, "match_ratio": round(ratio, 3), "method": "NOT_FOUND"}


def verify_all_citations(
    criteria_list: list[dict],
    context: str,
    threshold: float = 0.85,
) -> list[dict]:
    """
    Verify all citations in a list of rubric criteria.

    Args:
        criteria_list: LLM output — list of dicts with "source_quote" keys.
        context: The original input context.
        threshold: Minimum fuzzy ratio to consider verified.

    Returns:
        Same list, with a "grounding" dict added to each criterion.
    """
    for criterion in criteria_list:
        quote = criterion.get("source_quote", "")
        grounding = verify_citation(quote, context, threshold)
        criterion["grounding"] = grounding

    return criteria_list
