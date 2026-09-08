from typing import Optional, TypedDict

from src.v2.models.schemas import (
    Adjudication,
    Claim,
    Disagreement,
    EvaluationInput,
    EvaluatorResult,
    Evidence,
    FinalVerdict,
    ReferenceAnswer,
    ReferenceValidation,
)


class EvaluationState(TypedDict):
    """Shared state passed through the V2 evaluation graph."""

    # Initial input
    evaluation_input: EvaluationInput

    # RAG layer
    evidence: list[Evidence]

    # Autonomous reference pipeline
    reference_answer: Optional[ReferenceAnswer]
    reference_validation: Optional[ReferenceValidation]

    # Claim-level representation
    claims: list[Claim]

    # Parallel evaluators
    accuracy: Optional[EvaluatorResult]
    completeness: Optional[EvaluatorResult]
    logic: Optional[EvaluatorResult]
    grounding: Optional[EvaluatorResult]

    # Collected evaluator outputs
    evaluations: list[EvaluatorResult]

    # Arbitration
    disagreement: Optional[Disagreement]
    adjudication: Optional[Adjudication]

    # Final output
    final_verdict: Optional[FinalVerdict]