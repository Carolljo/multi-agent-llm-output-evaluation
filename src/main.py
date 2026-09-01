import os

from fastapi import FastAPI, HTTPException
from src.api.schemas import (
    AdjudicationResponse,
    DisagreementResponse,
    EvaluationItem,
    EvaluationRequest,
    EvaluationResponse,
    FinalResultResponse,
)
from src.graph.graph import create_evaluation_graph
from src.models.inputs import EvaluationInput


app = FastAPI(
    title="Multi-Agent LLM Output Evaluation API",
    description=(
        "API for evaluating LLM outputs using multiple independent "
        "evaluators and adjudication."
    ),
    version="1.0.0",
)


MODEL_NAME = os.getenv("LLM_MODEL", "qwen3:1.7b")

evaluation_graph = create_evaluation_graph(model=MODEL_NAME)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/evaluate", response_model=EvaluationResponse)
def evaluate(request: EvaluationRequest):
    evaluation_input = EvaluationInput(
        question=request.question,
        response=request.response,
        reference_answer=request.reference_answer,
    )

    try:
        result = evaluation_graph.invoke(
            {
                "evaluation_input": evaluation_input,
                "accuracy": None,
                "logic": None,
                "completeness": None,
                "evaluations": [],
                "evaluation_summary": None,
                "disagreement": None,
                "adjudication": None,
                "final_result": None,
            }
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Evaluation pipeline failed.",
        ) from exc

    final_result = result["final_result"]

    evaluations = [
        EvaluationItem(
            criterion=evaluation.criterion,
            score=evaluation.score,
            confidence=evaluation.confidence,
            reasoning=evaluation.reasoning,
            issues=evaluation.issues,
        )
        for evaluation in result["evaluations"]
    ]

    disagreement = result["disagreement"]

    disagreement_response = DisagreementResponse(
        has_disagreement=disagreement.has_disagreement,
        severity=disagreement.severity,
        score_spread=disagreement.score_spread,
        reasons=disagreement.reasons,
    )

    adjudication = result["adjudication"]

    adjudication_response = None

    if adjudication is not None:
        adjudication_response = AdjudicationResponse(
            final_verdict=adjudication.final_verdict,
            final_score=adjudication.final_score,
            confidence=adjudication.confidence,
            reasoning=adjudication.reasoning,
            issues=adjudication.issues,
        )

    return EvaluationResponse(
        evaluations=evaluations,
        disagreement=disagreement_response,
        adjudication=adjudication_response,
        final_result=FinalResultResponse(
            verdict=final_result.verdict,
            score=final_result.score,
            confidence=final_result.confidence,
            reasoning=final_result.reasoning,
            issues=final_result.issues,
        ),
    )