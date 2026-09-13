import os

from fastapi import FastAPI, HTTPException

from src.api.schemas import EvaluationRequest, EvaluationResponse
from src.v2.agents.accuracy import AccuracyEvaluator
from src.v2.agents.completeness import CompletenessEvaluator
from src.v2.agents.grounding import EvidenceGroundingEvaluator
from src.v2.agents.logic import LogicalConsistencyEvaluator
from src.v2.arbitration.adjudicator import Adjudicator
from src.v2.claims.extractor import ClaimExtractor
from src.v2.graph.workflow import build_autonomous_workflow
from src.v2.llm.client import LLMClient
from src.v2.models.schemas import EvaluationInput
from src.v2.rag.retriever import ChromaRetriever
from src.v2.reference.generator import ReferenceAnswerGenerator
from src.v2.reference.validator import ReferenceAnswerValidator


app = FastAPI(
    title="Multi-Agent LLM Output Evaluation & Arbitration System",
    version="2.0.0",
    description=(
        "Autonomous evidence-grounded LLM response evaluation "
        "and arbitration API."
    ),
)


def build_workflow():
    """Construct the autonomous V2 evaluation workflow."""

    model = os.getenv("OLLAMA_MODEL", "qwen3:1.7b")
    ollama_host = os.getenv(
        "OLLAMA_HOST",
        "http://localhost:11434",
    )

    llm_client = LLMClient(
        model=model,
        host=ollama_host,
    )

    retriever = ChromaRetriever()

    reference_generator = ReferenceAnswerGenerator(
        llm_client
    )

    reference_validator = ReferenceAnswerValidator(
        llm_client
    )

    claim_extractor = ClaimExtractor(
        llm_client
    )

    accuracy_evaluator = AccuracyEvaluator(
        llm_client
    )

    completeness_evaluator = CompletenessEvaluator(
        llm_client
    )

    logic_evaluator = LogicalConsistencyEvaluator(
        llm_client
    )

    grounding_evaluator = EvidenceGroundingEvaluator(
        llm_client
    )

    adjudicator = Adjudicator(
        llm_client
    )

    return build_autonomous_workflow(
        retriever=retriever,
        reference_generator=reference_generator,
        reference_validator=reference_validator,
        claim_extractor=claim_extractor,
        accuracy_evaluator=accuracy_evaluator,
        completeness_evaluator=completeness_evaluator,
        logic_evaluator=logic_evaluator,
        grounding_evaluator=grounding_evaluator,
        adjudicator=adjudicator,
    )


workflow = build_workflow()


@app.get("/health")
def health_check():
    """Return API health status."""

    return {
        "status": "healthy",
        "version": "2.0.0",
    }


@app.post(
    "/evaluate",
    response_model=EvaluationResponse,
)
def evaluate(request: EvaluationRequest):
    """Evaluate a candidate response using autonomous V2."""

    try:
        result = workflow.invoke(
            {
                "evaluation_input": EvaluationInput(
                    question=request.question,
                    response=request.response,
                ),
                "evidence": [],
                "reference_answer": None,
                "reference_validation": None,
                "claims": [],
                "accuracy": None,
                "completeness": None,
                "logic": None,
                "grounding": None,
                "evaluations": [],
                "aggregated": None,
                "disagreement": None,
                "adjudication": None,
                "final_verdict": None,
            }
        )

        return EvaluationResponse(
            final_verdict=result["final_verdict"],
            evidence=result["evidence"],
            reference_answer=result["reference_answer"],
            reference_validation=result["reference_validation"],
            claims=result["claims"],
            evaluations=result["evaluations"],
            disagreement=result["disagreement"],
            adjudication=result["adjudication"],
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Evaluation failed.",
        ) from exc