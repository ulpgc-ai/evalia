from abc import ABC, abstractmethod
from typing import List

from evalia.llm import LanguageModelTask, LanguageModelResponse


class BatchManager(ABC):
    @abstractmethod
    def start_task(self, query_id: str, query_list: List[str], temperature: float) -> LanguageModelTask:
        """
        Starts a batch task.
        :param query_id: the ID of the query
        :param query_list: list of queries to process
        :param temperature: temperature for the model
        :return: LanguageModelTask object representing the batch task
        """
        pass

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

    @abstractmethod
    def get_response(self, task: LanguageModelTask, timeout: int, retry: int) -> List[LanguageModelResponse]:
        """
        Retrieves the response for a batch task.
        :param task: LanguageModelTask object
        :param timeout: maximum wait time (seconds). If zero, waits indefinitely.
        :param retry: wait time between attempts (seconds). If zero, it only tries once.
        :return: List of LanguageModelResponse objects
        """
        pass