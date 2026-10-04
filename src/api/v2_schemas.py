from pydantic import BaseModel, Field
from src.v2.evidence.quality import EvidenceQuality
from src.v2.models.schemas import (
    Adjudication,
    Claim,
    Disagreement,
    Evidence,
    EvaluatorResult,
    FinalVerdict,
    ReferenceAnswer,
    ReferenceValidation,
)


class EvaluationRequest(BaseModel):
    """Request to evaluate a candidate response using autonomous V2."""

    question: str = Field(min_length=1)
    response: str = Field(min_length=1)


class EvaluationResponse(BaseModel):
    """Complete autonomous V2 evaluation response."""

    final_verdict: FinalVerdict

    evidence: list[Evidence] = Field(default_factory=list)
    
    evidence_quality: list[EvidenceQuality] = Field(default_factory=list)
    
    reference_answer: ReferenceAnswer | None = None

    reference_validation: ReferenceValidation | None = None

    claims: list[Claim] = Field(default_factory=list)

    evaluations: list[EvaluatorResult] = Field(default_factory=list)

    disagreement: Disagreement | None = None

    adjudication: Adjudication | None = None

    memory_hit: bool = False
    memory_similarity: float | None = Field(default=None, ge=0.0, le=1.0)