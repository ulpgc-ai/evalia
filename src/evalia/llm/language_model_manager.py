from abc import ABC, abstractmethod
from typing import List

from evalia.llm import LanguageModelResponse


class LanguageModelManager(ABC):
    """Interface to interact with different language models."""

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def generate_text(self, query_id: str, initial_prompt: str, query_list: List[str], system_context: str = "", temperature: float = 0.0) -> List[LanguageModelResponse]:
        """
        Generate text using the language model.
        :param query_id: the ID of the query (normally the evaluator's ID)
        :param initial_prompt: the initial prompt to be used for the query, with initial context and the question to be evaluated
        :param query_list: the list of responses to be evaluated
        :param system_context: instructional context to be included in the message sequence
        :param temperature: temperature for the model
        :return: a list of LanguageModelResponse objects containing the model's responses and stats
        """
        pass

    @abstractmethod
    def get_llm_name(self) -> str:
        """
        Get the name of the language model.
        :return: the name of the language model as a string
        """
        pass