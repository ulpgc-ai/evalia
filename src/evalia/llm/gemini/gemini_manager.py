import os
from abc import abstractmethod
from typing import List
from google import genai
from evalia.llm import LanguageModelManager, LanguageModelResponse, LanguageModelTask


class GeminiManager(LanguageModelManager):

    def __init__(self, model, temperature: float = 0.0, use_vertex: bool = False):
        super().__init__(model=model, temperature=temperature)
        self.use_vertex = use_vertex
        self.client = self.initialize_client(use_vertex)

    @staticmethod
    def initialize_client(use_vertex: bool = False):
        if use_vertex:
            return genai.Client(vertexai=use_vertex)
        else:
            return genai.Client(api_key=os.getenv('GOOGLE_API_KEY'), vertexai=use_vertex)

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
        self.client = self.initialize_client(self.use_vertex)
