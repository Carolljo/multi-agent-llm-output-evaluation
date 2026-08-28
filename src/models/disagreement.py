from pydantic import BaseModel, Field


class DisagreementResult(BaseModel):
    """Represents disagreement detected between evaluator results."""

    has_disagreement: bool
    severity: str
    score_spread: int = Field(ge=0)
    reasons: list[str]