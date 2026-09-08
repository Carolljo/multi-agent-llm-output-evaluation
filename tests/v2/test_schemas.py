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


def test_evaluation_input():
    data = EvaluationInput(
        question="What is machine learning?",
        response="Machine learning is a method where systems learn patterns from data.",
    )

    assert data.question
    assert data.response


def test_evidence():
    evidence = Evidence(
        evidence_id="E1",
        document_id="doc_001",
        chunk_id="chunk_001",
        content="Machine learning enables systems to learn from data.",
        source="ml_guide.pdf",
        relevance_score=0.92,
    )

    assert evidence.evidence_id == "E1"
    assert evidence.relevance_score == 0.92


def test_claim():
    claim = Claim(
        claim_id="C1",
        text="Machine learning systems learn patterns from data.",
        evidence_refs=["E1"],
        importance=0.9,
    )

    assert claim.evidence_refs == ["E1"]


def test_reference_answer():
    reference = ReferenceAnswer(
        answer="Machine learning allows systems to learn patterns from data.",
        claims=[
            Claim(
                claim_id="C1",
                text="Systems learn patterns from data.",
                evidence_refs=["E1"],
            )
        ],
        evidence_refs=["E1"],
        confidence=0.91,
    )

    assert len(reference.claims) == 1
    assert reference.confidence == 0.91


def test_reference_validation():
    validation = ReferenceValidation(
        status="valid",
        score=0.95,
        confidence=0.93,
        supported_claims=["C1"],
        unsupported_claims=[],
        missing_information=[],
        reasoning="The reference claims are supported by the retrieved evidence.",
    )

    assert validation.status == "valid"
    assert validation.score == 0.95


def test_evaluator_result():
    result = EvaluatorResult(
        evaluator="accuracy",
        score=0.88,
        confidence=0.91,
        claims_evaluated=5,
        errors=["C4 contains an unsupported factual claim."],
        reasoning="Most claims are factually supported.",
        evidence_refs=["E1", "E2"],
    )

    assert result.evaluator == "accuracy"
    assert result.claims_evaluated == 5


def test_disagreement():
    disagreement = Disagreement(
        detected=True,
        evaluators=["accuracy", "evidence_grounding"],
        score_range=0.42,
        reason="The evaluators produced significantly different scores.",
        requires_adjudication=True,
    )

    assert disagreement.detected is True
    assert disagreement.requires_adjudication is True


def test_adjudication():
    adjudication = Adjudication(
        verdict="Mostly correct",
        score=0.78,
        confidence=0.86,
        reasoning="The grounding evaluator identified one unsupported claim.",
        evidence_refs=["E1", "E2"],
    )

    assert adjudication.score == 0.78


def test_final_verdict():
    result = EvaluatorResult(
        evaluator="accuracy",
        score=0.9,
        confidence=0.92,
        claims_evaluated=4,
        reasoning="The response is largely accurate.",
    )

    verdict = FinalVerdict(
        verdict="Pass",
        score=0.87,
        confidence=0.9,
        evaluator_results=[result],
        key_errors=[],
        evidence_refs=["E1"],
    )

    assert verdict.verdict == "Pass"
    assert len(verdict.evaluator_results) == 1