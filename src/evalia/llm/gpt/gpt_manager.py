import json
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
    def generate_text(self, query_id: str, initial_prompt: str, query_list: List[str], system_context: str = "", temperature: float = 0.0) -> List[LanguageModelResponse]:
        pass

    def get_llm_name(self) -> str:
        return "GPT"

    def convert_to_gpt_messages(self, query: str, initial_prompt: str, system_context: str) -> List[dict]:
        """
        Builds a message list formatted for the OpenAI Chat API.
        :param query: The user query or batch of responses to be evaluated.
        :param system_context: Instructional context to be included in the message sequence.
        :return: A list of messages structured in the format expected by OpenAI.
        """
        return [
            {
                "role": "system",
                "content": json.dumps(system_context)
            },
            {
                "role": "user",
                "content": json.dumps(initial_prompt)
            },
            {
                "role": "assistant",
                "content": "Sí, he entendido las instrucciones. Pásame las respuestas para evaluar."
            },
            {
                "role": "user",
                "content": json.dumps(query)
            }
        ]