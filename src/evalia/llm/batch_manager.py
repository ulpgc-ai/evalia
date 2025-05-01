from abc import ABC, abstractmethod
from typing import List

from evalia.llm import LanguageModelTask, LanguageModelResponse


class BatchManager(ABC):

    @abstractmethod
    def cancel_task(self, task: LanguageModelTask):
        """
        Cancels a batch task.
        :param task: LanguageModelTask object to cancel
        """
        pass

    @abstractmethod
    def save_task(self, task: LanguageModelTask, filename: str):
        """
        Saves a batch task to a file.
        :param task: LanguageModelTask object to save
        :param filename: Path to the file where the task will be saved
        """
        pass

    @abstractmethod
    def load_task(self, filename: str) -> LanguageModelTask:
        """
        Loads a batch task from a file.
        :param filename: Path to the file containing the task
        :return: LanguageModelTask object loaded from the file
        """
        pass