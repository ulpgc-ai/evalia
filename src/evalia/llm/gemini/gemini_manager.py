import os
from abc import abstractmethod
from typing import List
from google import genai
from evalia.llm import LanguageModelManager, LanguageModelResponse, LanguageModelTask


class GeminiManager(LanguageModelManager):

    def __init__(self, model):
        super().__init__(model=model)
        self.client = self.initialize_client()

    @staticmethod
    def initialize_client():
        return genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

    @abstractmethod
    def start_task(self, query_id: str, initial_prompt: str, query_list: List[str], system_context: str = "",
                   temperature: float = 0.0) -> LanguageModelTask:
        pass

    @abstractmethod
    def get_response(self, task: LanguageModelTask, timeout: int, retry: int) -> List[LanguageModelResponse]:
        pass

    def get_llm_name(self) -> str:
        return "Gemini"

    def __getstate__(self):
        state = self.__dict__.copy()
        state["client"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.client = self.initialize_client()
