from typing import List

from evalia.llm import LanguageModelResponse, LanguageModelTask
from evalia.llm.gpt import GPTManager
from evalia.logs import get_logger

logger = get_logger(__name__)


class GPTMockManager(GPTManager):
    """
    Mock version of GPTManager. Processes messages without actually calling the OpenAI API.
    """

    def __init__(self):
        super().__init__(model="mock")

    def start_task(self, query_id: str, initial_prompt: str, query_list: List[str], system_context: str = "") -> LanguageModelTask:
        responses = [
            LanguageModelResponse(
                response=f"Mock response for query: {query}",
                elapsed_time=1,
                input_tokens=len(query.split()),
                output_tokens=len(f"Mock response for query: {query}".split()),
            )
            for query in query_list
        ]
        self.task = LanguageModelTask(id=query_id, responses=responses)
        return self.task

    def get_response(self, task: LanguageModelTask, timeout: int, retry: int) -> List[LanguageModelResponse]:
        return task.responses
