from abc import ABC, abstractmethod
from typing import List

from evalia.llm import LanguageModelResponse


class LanguageModelManager(ABC):
    """Interface to interact with different language models."""

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def generate_text(self, query_id: str, query_list: List[str], system_context: str = "", temperature: float = 0.0) -> List[LanguageModelResponse]:
        pass

    @abstractmethod
    def get_llm_name(self) -> str:
        pass