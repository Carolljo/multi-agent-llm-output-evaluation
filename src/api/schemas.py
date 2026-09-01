from pydantic import BaseModel, Field


class EvaluationRequest(BaseModel):
    question: str = Field(min_length=1)
    response: str = Field(min_length=1)
    reference_answer: str = Field(min_length=1)


class EvaluationItem(BaseModel):
    criterion: str
    score: int = Field(ge=1, le=10)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    issues: list[str]


class DisagreementResponse(BaseModel):
    has_disagreement: bool
    severity: str
    score_spread: int = Field(ge=0)
    reasons: list[str]


class AdjudicationResponse(BaseModel):
    final_verdict: str
    final_score: int = Field(ge=1, le=10)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    issues: list[str]


class FinalResultResponse(BaseModel):
    verdict: str
    score: float = Field(ge=1.0, le=10.0)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    issues: list[str]


class EvaluationResponse(BaseModel):
    evaluations: list[EvaluationItem]
    disagreement: DisagreementResponse
    adjudication: AdjudicationResponse | None
    final_result: FinalResultResponse