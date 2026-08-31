from pydantic import BaseModel, Field

from src.models.disagreement import DisagreementResult
from src.models.evaluation import EvaluationResult


class EvaluationSummary(BaseModel):
    evaluations: list[EvaluationResult]

    overall_score: float = Field(
        ge=1.0,
        le=10.0
    )

    overall_confidence: float = Field(
        ge=0.0,
        le=1.0
    )

    summary: str

    key_issues: list[str]

    needs_adjudication: bool

    disagreement: DisagreementResult