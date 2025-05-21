from abc import abstractmethod
from typing import List, Literal

import anthropic
from anthropic.types import MessageParam

from evalia.llm import LanguageModelManager, LanguageModelTask, LanguageModelResponse


class ClaudeManager(LanguageModelManager):

    def __init__(self, model: str, temperature: float = 0.0):
        super().__init__(model, temperature)
        self.client = self.initialize_client()

    @staticmethod
    def initialize_client():
        return anthropic.Anthropic()

    @staticmethod
    def create_claude_input_message(role: Literal['user', 'assistant'], prompt: str) -> MessageParam:
        return {
            "role": role,
            "content": prompt
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["client"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.client = self.initialize_client()

    @abstractmethod
    def start_task(self, query_id: str, initial_prompt: str, query_list: List[str], system_context: str = "",
                   temperature: float = 0.0) -> LanguageModelTask:
        pass

    @abstractmethod
    def get_response(self, task: LanguageModelTask, timeout: int, retry: int) -> List[LanguageModelResponse]:
        pass

    def get_llm_name(self) -> str:
        return "Claude"