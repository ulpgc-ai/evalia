import os
from abc import abstractmethod
from typing import List

from openai import OpenAI

from evalia.llm import LanguageModelManager, LanguageModelResponse


class GPTManager(LanguageModelManager):
    """
    Interfaz para interactuar con diferentes modelos de lenguaje de OpenAI.
    Esta clase hereda de LanguageModelManager y proporciona una interfaz para interactuar
    con modelos de lenguaje específicos de OpenAI.
    """

    def __init__(self, model: str):
        super().__init__(model)
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    @abstractmethod
    def generate_text(self, query_id: str, query_list: List[str], system_context: str = "", temperature: float = 0.0) -> List[LanguageModelResponse]:
        pass

    def get_llm_name(self) -> str:
        return "GPT"