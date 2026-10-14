"""
TrustGate — Pydantic v2 Response Schemas.
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    UNSTABLE = "UNSTABLE"
    NOT_EVALUATED = "NOT_EVALUATED"


class GroundingMethod(str, Enum):
    EXACT = "EXACT"
    FUZZY = "FUZZY"
    NOT_FOUND = "NOT_FOUND"


class GroundingResult(BaseModel):
    """Result of citation verification against source context."""
    verified: bool
    match_ratio: float = Field(ge=0.0, le=1.0)
    method: GroundingMethod


class CriteriaBreakdown(BaseModel):
    """Decomposed evaluation of a single rubric criterion."""
    criteria: str
    score: int = Field(default=0, ge=0)
    max_points: int = Field(default=1, ge=0)
    weight: float = Field(default=0.0, ge=0.0, le=1.0, description="Normalized weight of this criterion.")
    justification: str = ""
    source_quote: str = ""
    grounding: GroundingResult


class ConfidenceResult(BaseModel):
    """Statistical confidence derived from multi-sampling."""
    level: ConfidenceLevel
    variance: Optional[float] = Field(default=None, description="Standard deviation across runs.")
    runs_evaluated: int = Field(ge=1)


class Decision(BaseModel):
    """The final audited decision."""
    computed_score: float = Field(ge=0, le=100)
    scale: list[int] = Field(default=[0, 100])
    severity_label: str
    confidence: ConfidenceResult


class AuditMetadata(BaseModel):
    """Metadata for audit trail."""
    model: str
    logical_tier: Optional[str] = "PRIMARY"
    fallback_occurred: bool = False
    cached: bool = False
    hallucination_flags: int = Field(ge=0, description="Number of criteria with unverified citations.")
    unverified_citations: int = Field(ge=0)
    latency_ms: int = Field(ge=0)
    total_tokens: int = 0
    cost_usd: float = 0.0
    timestamp: str


class EvaluateResponse(BaseModel):
    """Full audited response from POST /v1/evaluate."""
    decision: Decision
    rubric_breakdown: list[CriteriaBreakdown]
    audit_metadata: AuditMetadata
