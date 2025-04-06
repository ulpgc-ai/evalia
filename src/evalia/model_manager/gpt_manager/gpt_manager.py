from abc import ABC

from src.evalia.model_manager import ModelManager


class GPTManager(ABC, ModelManager):
    """
    Interfaz para interactuar con diferentes modelos de lenguaje de OpenAI.
    Esta clase hereda de ModelManager y proporciona una interfaz para interactuar
    con modelos de lenguaje específicos de OpenAI.
    """
