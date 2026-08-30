from src.models.adjudication import AdjudicationResult


def test_valid_adjudication_result():
    result = AdjudicationResult(
        final_verdict="Mostly Correct",
        final_score=8,
        confidence=0.9,
        reasoning="The response is mostly correct but contains a minor issue.",
        issues=["Minor factual imprecision."],
    )

    assert result.final_verdict == "Mostly Correct"
    assert result.final_score == 8
    assert result.confidence == 0.9
    assert len(result.issues) == 1


def test_adjudication_result_with_no_issues():
    result = AdjudicationResult(
        final_verdict="Correct",
        final_score=10,
        confidence=0.95,
        reasoning="The response is accurate, logically sound, and complete.",
        issues=[],
    )

    assert result.final_verdict == "Correct"
    assert result.final_score == 10
    assert result.issues == []