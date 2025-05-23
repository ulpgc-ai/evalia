from typing import List

from pydantic import BaseModel

class EvaluatedAnswer(BaseModel):
    index: int
    score: float

class EvaluatedJustifiedAnswer(EvaluatedAnswer):
    comment: str

class EvaluatedAnswers(BaseModel):
    results: List[EvaluatedAnswer]

class EvaluatedJustifiedAnswers(BaseModel):
    results: List[EvaluatedJustifiedAnswer]