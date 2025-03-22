from abc import ABC, abstractmethod

from src.evalia.model_manager import ModelResponse


class ModelManager(ABC):
    """Interface to interact with different language models."""

    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key

    def load_model(self, model: str):
        self.model = model

    @abstractmethod
    def generate_text(self, prompt: str, system_context: str = "", temperature: str = "") -> ModelResponse:
        pass

    @abstractmethod
    def get_llm_name(self) -> str:
        pass
