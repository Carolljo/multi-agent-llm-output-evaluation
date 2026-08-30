from typing import Literal

from pydantic import BaseModel, Field


class AdjudicationResult(BaseModel):
    final_verdict: Literal[
        "Correct",
        "Mostly Correct",
        "Partially Correct",
        "Incorrect",
    ]

    final_score: int = Field(ge=1, le=10)

    confidence: float = Field(ge=0.0, le=1.0)

    reasoning: str

    issues: list[str]