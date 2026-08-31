from pydantic import BaseModel, Field


class FinalEvaluationResult(BaseModel):
    verdict: str
    score: float = Field(ge=1.0, le=10.0)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    issues: list[str]