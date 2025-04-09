""" 
Versión mock de GPTManager.
Procesa los mensajes sin dialogar realmente con GPT.
"""

from evalia.logs import get_logger
from evalia.llm import LanguageModelManager, LanguageModelResponse

logger = get_logger(__name__)

class GPTMockManager(LanguageModelManager):
    """Versión mock de GPTManager. Devuelve respuestas ficticias."""

    def __init__(self):
        super().__init__(model="mock", api_key="mock-key")

    def generate_text(self, prompt: str, system_context: str = "", temperature: str = "") -> LanguageModelResponse:
        return LanguageModelResponse(response="Mock response")
    
    def get_llm_name(self) -> str:
        return "Mock GPT"
