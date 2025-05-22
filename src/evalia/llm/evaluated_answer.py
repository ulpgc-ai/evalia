from typing import List

from pydantic import BaseModel


class EvaluatedAnswer(BaseModel):
    index: int
    score: float
    comment: str

class EvaluatedAnswers(BaseModel):
    results: List[EvaluatedAnswer]