from abc import ABC, abstractmethod
from typing import List


class SmartManager(ABC):
    @abstractmethod
    def count_tokens(self, messages: List[str]) -> int:
        """
        Calculate the total number of tokens used by a list of messages.
        :param messages: List of messages to count tokens for.
        :return: Total number of tokens.
        """
        pass