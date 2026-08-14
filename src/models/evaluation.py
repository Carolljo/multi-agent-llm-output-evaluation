from pydantic import BaseModel, Field


class EvaluationResult(BaseModel):
    criterion: str
    score: int = Field(ge=1, le=10)
    reasoning: str
    issues: list[str]
    confidence: float = Field(ge=0.0, le=1.0)