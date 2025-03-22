""" 
Versión mock de GPTManager.
Procesa los mensajes sin dialogar realmente con GPT.
"""

from src.evalia.logs import get_logger
from src.evalia.model_manager import ModelManager, ModelResponse

logger = get_logger(__name__)

class GPTMockManager(ModelManager):
    """Versión mock de GPTManager. Devuelve respuestas ficticias."""

    def __init__(self):
        super().__init__(model="mock", api_key="mock-key")

    def generate_text(self, prompt: str, system_context: str = "", temperature: str = "") -> ModelResponse:
        return ModelResponse(response="Mock response")
    
    def get_llm_name(self) -> str:
        return "Mock GPT"
