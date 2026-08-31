from typing import Optional, TypedDict

from src.models.adjudication import AdjudicationResult
from src.models.disagreement import DisagreementResult
from src.models.evaluation import EvaluationResult
from src.models.final_result import FinalEvaluationResult
from src.models.inputs import EvaluationInput
from src.models.summary import EvaluationSummary


class EvaluationState(TypedDict):
    evaluation_input: EvaluationInput

    accuracy: Optional[EvaluationResult]
    logic: Optional[EvaluationResult]
    completeness: Optional[EvaluationResult]

    evaluations: list[EvaluationResult]

    evaluation_summary: Optional[EvaluationSummary]
    disagreement: Optional[DisagreementResult]
    adjudication: Optional[AdjudicationResult]
    final_result: Optional[FinalEvaluationResult]