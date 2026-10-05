from typing import Optional, TypedDict

from src.v2.aggregation.aggregator import AggregatedEvaluation
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
from src.v2.memory.decision import MemoryDecision
from src.v2.memory.schemas import AnswerMemoryMatch
from src.v2.evidence.quality import EvidenceQuality


class EvaluationState(TypedDict):
    """Shared state passed through the V2 evaluation graph."""

    # Initial input
    evaluation_input: EvaluationInput

    # Verified answer memory
    memory_matches: list[AnswerMemoryMatch]
    memory_hit: bool
    memory_decision: Optional[MemoryDecision]
    memory_decision_reason: Optional[str]
    memory_similarity: Optional[float]
    memory_validation_score: Optional[float]
    memory_validation_confidence: Optional[float]
    memory_decision_score: Optional[float]
    memory_answer: Optional[str]
    memory_evidence: list[Evidence]

    # RAG layer
    evidence: list[Evidence]
    
    # Evidence quality
    evidence_quality: list[EvidenceQuality]

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

    # Deterministic aggregation
    aggregated: Optional[AggregatedEvaluation]

    # Arbitration
    disagreement: Optional[Disagreement]
    adjudication: Optional[Adjudication]

    # Final output
    final_verdict: Optional[FinalVerdict]