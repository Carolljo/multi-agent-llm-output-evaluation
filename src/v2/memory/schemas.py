from typing import Literal

from pydantic import BaseModel, Field

from src.v2.models.schemas import Claim, Evidence


class VerifiedAnswer(BaseModel):
    """A reference answer that has independently passed validation."""

    memory_id: str
    question: str
    answer: str
    claims: list[Claim] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    source_documents: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)

    validation_status: Literal["valid"] = "valid"
    validation_score: float = Field(ge=0.0, le=1.0)
    validation_confidence: float = Field(ge=0.0, le=1.0)

    created_at: str
    updated_at: str
    version: int = Field(default=1, ge=1)


class AnswerMemoryMatch(BaseModel):
    """A verified answer retrieved from answer memory."""

    memory: VerifiedAnswer
    similarity: float = Field(ge=0.0, le=1.0)
