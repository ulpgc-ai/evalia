from typing import List

from evalia.llm import LanguageModelResponse


class LanguageModelTask:
  """
    A class representing a task for a language model.
  """
  def __init__(self, id: str, responses: List[LanguageModelResponse] | None = None):
    self.id = id
    self.responses: List[LanguageModelResponse] | None = responses