from src.models.inputs import EvaluationInput

evaluation_input = EvaluationInput(
    question="What is the capital of France?",
    reference_answer="The capital of France is Paris.",
    response="The capital of France is Paris."
)

print(evaluation_input)