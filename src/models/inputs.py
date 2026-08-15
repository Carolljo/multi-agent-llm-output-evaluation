from pydantic import BaseModel


class EvaluationInput(BaseModel):
    question: str
    response: str
    reference_answer: str