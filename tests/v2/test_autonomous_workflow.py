import pytest

from src.v2.graph.workflow import build_autonomous_workflow
from src.v2.models.schemas import (
    Adjudication,
    Claim,
    Disagreement,
    EvaluationInput,
    Evidence,
    EvaluatorResult,
    ReferenceAnswer,
    ReferenceValidation,
)


class FakeRetriever:
    def __init__(self):
        self.called = False

    def search(self, *, query, top_k):
        self.called = True

        assert query == "How many authentication methods are supported?"
        assert top_k == 5

        return [
            Evidence(
                evidence_id="E1",
                document_id="DOC1",
                chunk_id="CH1",
                content="Three authentication methods are supported.",
                source="test-document",
            )
        ]


class FakeReferenceGenerator:
    def __init__(self):
        self.called = False

    def generate(self, *, question, evidence):
        self.called = True

        assert question == "How many authentication methods are supported?"
        assert len(evidence) == 1
        assert evidence[0].evidence_id == "E1"

        return ReferenceAnswer(
            answer="Three authentication methods are supported.",
            claims=[
                Claim(
                    claim_id="RC1",
                    text="Three authentication methods are supported.",
                    evidence_refs=["E1"],
                    importance=1.0,
                )
            ],
            evidence_refs=["E1"],
            confidence=0.95,
        )


class FakeReferenceValidator:
    def __init__(self):
        self.called = False

    def validate(self, *, question, reference_answer, evidence):
        self.called = True

        assert question == "How many authentication methods are supported?"
        assert reference_answer.answer == (
            "Three authentication methods are supported."
        )
        assert len(evidence) == 1

        return ReferenceValidation(
            status="valid",
            score=1.0,
            confidence=0.95,
            supported_claims=["RC1"],
            unsupported_claims=[],
            missing_information=[],
            reasoning="The reference answer is fully supported.",
        )


class FakeClaimExtractor:
    def __init__(self):
        self.called = False

    def extract(
        self,
        *,
        question,
        response,
        reference_answer,
        evidence,
    ):
        self.called = True

        assert question == "How many authentication methods are supported?"
        assert response == "Three authentication methods are supported."
        assert reference_answer.answer == (
            "Three authentication methods are supported."
        )
        assert len(evidence) == 1

        return [
            Claim(
                claim_id="C1",
                text="Three authentication methods are supported.",
                evidence_refs=["E1"],
                importance=1.0,
            )
        ]


class FakeEvaluator:
    def __init__(self, evaluator_name, score=0.9):
        self.evaluator_name = evaluator_name
        self.score = score
        self.called = False

    def evaluate(
        self,
        *,
        question,
        claims,
        reference_answer,
        evidence,
    ):
        self.called = True

        assert question == "How many authentication methods are supported?"
        assert len(claims) == 1
        assert reference_answer is not None
        assert len(evidence) == 1

        return EvaluatorResult(
            evaluator=self.evaluator_name,
            score=self.score,
            confidence=0.9,
            claims_evaluated=1,
            errors=[],
            reasoning="Evaluation completed.",
            evidence_refs=["E1"],
        )


class FakeAdjudicator:
    def __init__(self):
        self.called = False

    def adjudicate(
        self,
        *,
        question,
        response,
        reference_answer,
        claims,
        evidence,
        evaluations,
        disagreement,
    ):
        self.called = True

        return Adjudication(
            verdict="accept",
            confidence=0.92,
            reasoning="The response is supported.",
            evidence_refs=["E1"],
        )


def build_initial_state():
    return {
        "evaluation_input": EvaluationInput(
            question="How many authentication methods are supported?",
            response="Three authentication methods are supported.",
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


def build_dependencies():
    retriever = FakeRetriever()
    reference_generator = FakeReferenceGenerator()
    reference_validator = FakeReferenceValidator()
    claim_extractor = FakeClaimExtractor()

    accuracy = FakeEvaluator("accuracy", 0.9)
    completeness = FakeEvaluator("completeness", 0.9)
    logic = FakeEvaluator("logical_consistency", 0.9)
    grounding = FakeEvaluator("evidence_grounding", 0.9)

    adjudicator = FakeAdjudicator()

    return (
        retriever,
        reference_generator,
        reference_validator,
        claim_extractor,
        accuracy,
        completeness,
        logic,
        grounding,
        adjudicator,
    )


def test_autonomous_workflow_runs_end_to_end():
    (
        retriever,
        reference_generator,
        reference_validator,
        claim_extractor,
        accuracy,
        completeness,
        logic,
        grounding,
        adjudicator,
    ) = build_dependencies()

    workflow = build_autonomous_workflow(
        retriever=retriever,
        reference_generator=reference_generator,
        reference_validator=reference_validator,
        claim_extractor=claim_extractor,
        accuracy_evaluator=accuracy,
        completeness_evaluator=completeness,
        logic_evaluator=logic,
        grounding_evaluator=grounding,
        adjudicator=adjudicator,
    )

    result = workflow.invoke(build_initial_state())

    assert retriever.called is True
    assert reference_generator.called is True
    assert reference_validator.called is True
    assert claim_extractor.called is True

    assert accuracy.called is True
    assert completeness.called is True
    assert logic.called is True
    assert grounding.called is True

    assert result["reference_answer"] is not None
    assert result["reference_validation"].status == "valid"

    assert len(result["claims"]) == 1
    assert len(result["evaluations"]) == 4

    assert result["aggregated"].score == pytest.approx(0.9)
    assert result["disagreement"].detected is False

    assert result["adjudication"] is None
    assert result["final_verdict"].verdict == "accept"
    assert result["final_verdict"].score == pytest.approx(0.9)

    assert adjudicator.called is False


def test_autonomous_workflow_adjudicates_on_disagreement():
    (
        retriever,
        reference_generator,
        reference_validator,
        claim_extractor,
        accuracy,
        completeness,
        logic,
        grounding,
        adjudicator,
    ) = build_dependencies()

    grounding.score = 0.4

    workflow = build_autonomous_workflow(
        retriever=retriever,
        reference_generator=reference_generator,
        reference_validator=reference_validator,
        claim_extractor=claim_extractor,
        accuracy_evaluator=accuracy,
        completeness_evaluator=completeness,
        logic_evaluator=logic,
        grounding_evaluator=grounding,
        adjudicator=adjudicator,
    )

    result = workflow.invoke(build_initial_state())

    assert len(result["evaluations"]) == 4

    assert result["aggregated"].score == pytest.approx(0.775)

    assert result["disagreement"].detected is True
    assert result["disagreement"].requires_adjudication is True

    assert adjudicator.called is True

    assert result["adjudication"].verdict == "accept"
    assert result["adjudication"].confidence == pytest.approx(0.92)

    # Numerical score remains deterministic.
    assert result["final_verdict"].score == pytest.approx(0.775)

    # Qualitative verdict comes from adjudication.
    assert result["final_verdict"].verdict == "accept"


def test_autonomous_workflow_rejects_invalid_reference():
    (
        retriever,
        reference_generator,
        reference_validator,
        claim_extractor,
        accuracy,
        completeness,
        logic,
        grounding,
        adjudicator,
    ) = build_dependencies()

    def invalid_reference(*, question, reference_answer, evidence):
        return ReferenceValidation(
            status="invalid",
            score=0.2,
            confidence=0.95,
            supported_claims=[],
            unsupported_claims=["RC1"],
            missing_information=[],
            reasoning="The reference contains unsupported information.",
        )

    reference_validator.validate = invalid_reference

    workflow = build_autonomous_workflow(
        retriever=retriever,
        reference_generator=reference_generator,
        reference_validator=reference_validator,
        claim_extractor=claim_extractor,
        accuracy_evaluator=accuracy,
        completeness_evaluator=completeness,
        logic_evaluator=logic,
        grounding_evaluator=grounding,
        adjudicator=adjudicator,
    )

    with pytest.raises(
        ValueError,
        match="Generated reference answer failed validation",
    ):
        workflow.invoke(build_initial_state())

    assert claim_extractor.called is False
    assert accuracy.called is False
    assert completeness.called is False
    assert logic.called is False
    assert grounding.called is False