from src.models.evaluation import EvaluationResult


result = EvaluationResult(
    criterion="accuracy",
    score=9,
    
    reasoning="The answer is factually correct.",
    issues=[],
    confidence=0.95,
)

print(result)