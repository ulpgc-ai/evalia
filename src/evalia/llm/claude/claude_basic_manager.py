import json
import time
from typing import List

import yaml

from evalia.llm import LanguageModelTask, LanguageModelResponse
from evalia.llm.claude.claude_manager import ClaudeManager


class ClaudeBasicManager(ClaudeManager):
    def start_task(self, query_id: str, initial_prompt: str, query_list: List[str], system_context: str = "") -> LanguageModelTask:
        if self.task is not None:
            return self.task
        responses: List[LanguageModelResponse] = []
        for query in query_list:
            start_time = time.time()
            response = self.client.messages.create(
                model=self.model,
                messages=self.convert_to_claude_messages(query, initial_prompt),
                system=system_context,
                max_tokens=4096,
                temperature=self.temperature
            )
            elapsed_time = time.time() - start_time
            responses.append(LanguageModelResponse(
                response=ClaudeManager.parse_claude_response(response),
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                elapsed_time=elapsed_time
            ))

        self.task = LanguageModelTask(id=query_id, responses=responses)
        self.responses = responses
        return self.task

    def get_response(self, task: LanguageModelTask, timeout: int, retry: int) -> List[LanguageModelResponse]:
        if self.responses is not None:
            return self.responses
        return task.responses