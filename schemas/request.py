"""
TrustGate — Pydantic v2 Request Schemas.
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class TaskType(str, Enum):
    """Supported scoring task types. Each maps to a predefined rubric."""
    THREAT_SEVERITY = "THREAT_SEVERITY"
    CREDIT_RISK = "CREDIT_RISK"
    COMPLIANCE_CHECK = "COMPLIANCE_CHECK"
    CUSTOM = "CUSTOM"


class EvaluateConfig(BaseModel):
    """Optional configuration overrides for an evaluation request."""
    sampling_runs: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Number of parallel LLM passes for variance estimation. 1 = no variance check.",
    )
    grounding_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Minimum fuzzy match ratio to consider a citation verified.",
    )
    model: Optional[str] = Field(
        default=None,
        description="Override the default LLM model (e.g., 'models/gemini-2.5-flash').",
    )


class EvaluateRequest(BaseModel):
    """Input payload for POST /v1/evaluate."""
    context: str = Field(
        ...,
        min_length=20,
        description="The raw text to evaluate (threat report, incident description, etc.).",
    )
    task_type: TaskType = Field(
        default=TaskType.THREAT_SEVERITY,
        description="The type of scoring task. Determines which rubric is applied.",
    )
    config: EvaluateConfig = Field(
        default_factory=EvaluateConfig,
        description="Optional evaluation configuration.",
    )

    model_config = {"json_schema_extra": {
        "examples": [{
            "context": "A critical vulnerability CVE-2026-1104 in OpenSSL allows remote code execution without authentication. Exploitation scripts are actively circulating on public repositories. The impacted service runs inside an isolated VLAN with no direct internet access.",
            "task_type": "THREAT_SEVERITY",
            "config": {"sampling_runs": 3, "grounding_threshold": 0.85}
        }]
    }}
