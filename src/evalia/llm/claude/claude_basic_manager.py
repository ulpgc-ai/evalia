import time
from typing import List

from evalia.llm import LanguageModelTask, LanguageModelResponse
from evalia.llm.claude.claude_manager import ClaudeManager


class ClaudeBasicManager(ClaudeManager):
    def start_task(self, query_id: str, initial_prompt: str, query_list: List[str], system_context: str = "",
                   temperature: float = 0.0) -> LanguageModelTask:
        if self.task is not None:
            return self.task
        responses: List[LanguageModelResponse] = []
        for query in query_list:
            start_time = time.time()
            response = self.client.messages.create(
                model=self.model,
                messages=self.convert_to_claude_messages(query, initial_prompt, system_context),
                system=system_context,
                max_tokens=1000,
                temperature=0
            )
            elapsed_time = time.time() - start_time
            responses.append(LanguageModelResponse(
                response=self.parse_claude_response(response.content[0].text),
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