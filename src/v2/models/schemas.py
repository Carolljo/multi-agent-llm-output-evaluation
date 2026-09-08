from typing import Any, Literal

from pydantic import BaseModel, Field


class EvaluationInput(BaseModel):
    """Input provided by the user to start a V2 evaluation."""

    question: str
    response: str


class Evidence(BaseModel):
    """A retrieved evidence chunk used to ground evaluation."""

    evidence_id: str
    document_id: str
    chunk_id: str
    content: str

    source: str | None = None
    relevance_score: float | None = Field(default=None, ge=0.0, le=1.0)

    metadata: dict[str, Any] = Field(default_factory=dict)


class Claim(BaseModel):
    """A factual or logical claim extracted from an answer."""

    claim_id: str
    text: str

    evidence_refs: list[str] = Field(default_factory=list)

    importance: float | None = Field(default=None, ge=0.0, le=1.0)


class ReferenceAnswer(BaseModel):
    """Reference answer generated from retrieved evidence."""

    answer: str
    claims: list[Claim] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)

    confidence: float = Field(ge=0.0, le=1.0)


class ReferenceValidation(BaseModel):
    """Independent validation of the generated reference answer."""

    status: Literal["valid", "invalid", "needs_review"]

    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)

    supported_claims: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)

    reasoning: str


class EvaluatorResult(BaseModel):
    """Structured result produced by one V2 evaluator."""

    evaluator: Literal[
        "accuracy",
        "completeness",
        "logical_consistency",
        "evidence_grounding",
    ]

    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)

    claims_evaluated: int = Field(ge=0)

    errors: list[str] = Field(default_factory=list)
    reasoning: str

    evidence_refs: list[str] = Field(default_factory=list)


class Disagreement(BaseModel):
    """Represents disagreement between independent evaluators."""

    detected: bool

    evaluators: list[str] = Field(default_factory=list)

    score_range: float | None = Field(default=None, ge=0.0, le=1.0)

    reason: str
    requires_adjudication: bool


class Adjudication(BaseModel):
    """Final reasoning used to resolve evaluator disagreement."""

    verdict: str

    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)

    reasoning: str

    evidence_refs: list[str] = Field(default_factory=list)


class FinalVerdict(BaseModel):
    """Final confidence-scored V2 evaluation verdict."""

    verdict: str

    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)

    evaluator_results: list[EvaluatorResult] = Field(default_factory=list)

    disagreement: Disagreement | None = None
    adjudication: Adjudication | None = None

    key_errors: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)