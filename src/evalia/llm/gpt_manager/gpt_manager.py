from abc import ABC

from evalia.llm import LanguageModelManager


class GPTManager(ABC, LanguageModelManager):
    """
    Interfaz para interactuar con diferentes modelos de lenguaje de OpenAI.
    Esta clase hereda de ModelManager y proporciona una interfaz para interactuar
    con modelos de lenguaje específicos de OpenAI.
    """
