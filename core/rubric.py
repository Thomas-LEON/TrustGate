"""
TrustGate — Rubric Engine.

The LLM never computes the final score. It evaluates atomic criteria.
The global score is recalculated deterministically server-side.
"""
from __future__ import annotations

import json
import re
from typing import Any

# ---------------------------------------------------------------------------
# Rubric Templates
# ---------------------------------------------------------------------------
# Each task type has a rubric: a list of criteria with max_points and weight.
# The LLM must evaluate each criterion independently.

RUBRIC_TEMPLATES: dict[str, list[dict[str, Any]]] = {
    "THREAT_SEVERITY": [
        {
            "criteria": "Technical Impact",
            "max_points": 40,
            "weight": 0.40,
            "description": "How severe is the technical impact? Consider: RCE, privilege escalation, data exfiltration, DoS. Score 0 (informational) to 40 (critical RCE with full system compromise).",
        },
        {
            "criteria": "Exploit Availability",
            "max_points": 30,
            "weight": 0.30,
            "description": "Is there a known exploit, PoC, or active exploitation? Score 0 (theoretical) to 30 (weaponized exploit actively used in the wild).",
        },
        {
            "criteria": "Exposure & Scope",
            "max_points": 30,
            "weight": 0.30,
            "description": "How exposed is the affected asset? Consider: internet-facing, network segmentation, number of affected systems. Score 0 (fully isolated, single system) to 30 (internet-facing, thousands of instances).",
        },
    ],
    "CREDIT_RISK": [
        {
            "criteria": "Financial Health",
            "max_points": 35,
            "weight": 0.35,
            "description": "Assess the entity's financial stability: revenue trends, debt ratios, liquidity. Score 0 (extremely healthy) to 35 (imminent default risk).",
        },
        {
            "criteria": "Market & Sector Risk",
            "max_points": 30,
            "weight": 0.30,
            "description": "Evaluate systemic and sector-specific risks: regulatory changes, market volatility, competitive pressures. Score 0 (stable blue-chip sector) to 30 (highly volatile/distressed sector).",
        },
        {
            "criteria": "Counterparty Exposure",
            "max_points": 35,
            "weight": 0.35,
            "description": "Assess concentration risk, contagion potential, and counterparty dependencies. Score 0 (diversified, no single dependency) to 35 (concentrated, systemic counterparty).",
        },
    ],
    "COMPLIANCE_CHECK": [
        {
            "criteria": "Regulatory Alignment",
            "max_points": 40,
            "weight": 0.40,
            "description": "Does the subject meet applicable regulatory requirements? Score 0 (fully compliant) to 40 (severe non-compliance with enforcement risk).",
        },
        {
            "criteria": "Control Effectiveness",
            "max_points": 30,
            "weight": 0.30,
            "description": "Are internal controls in place and functioning? Score 0 (robust controls, regularly tested) to 30 (no controls or controls bypassed).",
        },
        {
            "criteria": "Remediation Urgency",
            "max_points": 30,
            "weight": 0.30,
            "description": "How urgent is remediation? Score 0 (advisory, no deadline) to 30 (immediate action required, active regulatory scrutiny).",
        },
    ],
}

# Default rubric for CUSTOM task_type
DEFAULT_RUBRIC = [
    {
        "criteria": "Severity",
        "max_points": 40,
        "weight": 0.40,
        "description": "Overall severity of the situation described. Score 0 (negligible) to 40 (critical).",
    },
    {
        "criteria": "Likelihood",
        "max_points": 30,
        "weight": 0.30,
        "description": "Probability that the described risk materializes. Score 0 (theoretical) to 30 (near-certain).",
    },
    {
        "criteria": "Business Impact",
        "max_points": 30,
        "weight": 0.30,
        "description": "Financial, reputational, or operational impact on the organization. Score 0 (negligible) to 30 (existential threat).",
    },
]


def get_rubric(task_type: str) -> list[dict[str, Any]]:
    """Return the rubric template for a given task type."""
    return RUBRIC_TEMPLATES.get(task_type, DEFAULT_RUBRIC)


# ---------------------------------------------------------------------------
# Prompt Compiler
# ---------------------------------------------------------------------------

def build_scoring_prompt(context: str, task_type: str) -> str:
    """
    Build the structured prompt that forces the LLM to evaluate
    each rubric criterion independently with source citations.
    """
    rubric = get_rubric(task_type)

    criteria_block = ""
    for i, c in enumerate(rubric, 1):
        criteria_block += (
            f"\n### Criterion {i}: {c['criteria']}\n"
            f"- Max score: {c['max_points']}\n"
            f"- Evaluation guide: {c['description']}\n"
        )

    prompt = f"""You are an expert risk analyst performing a structured evaluation.

TASK TYPE: {task_type}

CONTEXT TO EVALUATE:
---
{context}
---

INSTRUCTIONS:
You MUST evaluate the context against each criterion below independently.
For EACH criterion, you MUST provide:
1. A "score" (integer, 0 to the max score specified).
2. A "justification" (1-2 sentences explaining your reasoning).
3. A "source_quote" (a VERBATIM excerpt from the context above that supports your score. Copy it EXACTLY as it appears in the context. Do NOT paraphrase or invent text.)

CRITICAL RULES:
- Do NOT compute a total score. Only score each criterion individually.
- The source_quote MUST be a direct, word-for-word substring from the CONTEXT above.
- If no supporting evidence exists in the context for a criterion, set score to 0 and source_quote to "NO_EVIDENCE".

EVALUATION CRITERIA:
{criteria_block}

Respond with ONLY a valid JSON array. Each element must have exactly these keys:
"criteria", "score", "justification", "source_quote"

Example format:
[
  {{"criteria": "...", "score": 25, "justification": "...", "source_quote": "..."}},
  ...
]
"""
    return prompt


# ---------------------------------------------------------------------------
# Deterministic Score Calculator
# ---------------------------------------------------------------------------

def compute_final_score(
    llm_criteria: list[dict],
    rubric: list[dict],
) -> tuple[float, str]:
    """
    Recalculate the global score deterministically from sub-scores.
    
    Returns:
        (score_0_to_100, severity_label)
    """
    total = 0.0
    max_total = 0.0

    from rapidfuzz import fuzz

    for r in rubric:
        r_name = r["criteria"].strip().lower()

        # 1. Exact match
        match = next(
            (c for c in llm_criteria if c.get("criteria", "").strip().lower() == r_name),
            None,
        )

        # 2. Substring match
        if not match:
            match = next(
                (c for c in llm_criteria if r_name in c.get("criteria", "").strip().lower() or c.get("criteria", "").strip().lower() in r_name),
                None,
            )

        # 3. Fuzzy match
        if not match:
            best_c = max(llm_criteria, key=lambda c: fuzz.token_set_ratio(c.get("criteria", "").lower(), r_name), default=None)
            if best_c and fuzz.token_set_ratio(best_c.get("criteria", "").lower(), r_name) >= 50:
                match = best_c

        if match:
            # Clamp score to valid range
            raw_score = min(max(int(match.get("score", 0)), 0), r["max_points"])
            total += raw_score
        max_total += r["max_points"]

    if max_total == 0:
        return 0.0, "UNKNOWN"

    normalized = (total / max_total) * 100.0
    normalized = round(normalized, 1)

    # Severity classification
    if normalized >= 80:
        label = "CRITICAL"
    elif normalized >= 60:
        label = "HIGH"
    elif normalized >= 40:
        label = "MEDIUM"
    elif normalized >= 20:
        label = "LOW"
    else:
        label = "INFORMATIONAL"

    return normalized, label


# ---------------------------------------------------------------------------
# LLM Response Parser
# ---------------------------------------------------------------------------

def parse_llm_criteria(raw_text: str) -> list[dict] | None:
    """
    Extract and parse the JSON array from the LLM response.
    Handles markdown wrappers and conversational filler.
    """
    # Try to find JSON array in markdown code block
    match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw_text, re.DOTALL | re.IGNORECASE)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find bare JSON array
    match = re.search(r"\[.*\]", raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None
