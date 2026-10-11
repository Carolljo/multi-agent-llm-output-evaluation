from pydantic import BaseModel, Field

from src.v2.models.schemas import (
    Adjudication,
    Disagreement,
    EvaluatorResult,
    FinalVerdict,
    ReferenceAnswer,
    ReferenceValidation,
    Evidence,
    Claim,
)


class EvaluationRequest(BaseModel):
    """Request to evaluate an answer or generate one autonomously."""

    question: str = Field(min_length=1)
    response: str | None = Field(default=None, min_length=1)


class EvaluationResponse(BaseModel):
    """Complete autonomous V2 evaluation response."""

    final_verdict: FinalVerdict
    evidence: list[Evidence] = Field(default_factory=list)
    reference_answer: ReferenceAnswer | None = None
    reference_validation: ReferenceValidation | None = None
    claims: list[Claim] = Field(default_factory=list)
    evaluations: list[EvaluatorResult] = Field(default_factory=list)
    disagreement: Disagreement | None = None
    adjudication: Adjudication | None = None
