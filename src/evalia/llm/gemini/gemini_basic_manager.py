import time
from typing import List

from google.genai import types
from google.genai.types import GenerateContentResponse

from evalia.llm import LanguageModelTask, LanguageModelResponse
from evalia.llm.gemini import GeminiManager


class GeminiBasicManager(GeminiManager):
    def start_task(self, query_id: str, initial_prompt: str, query_list: List[str], system_context: str = "") -> LanguageModelTask:
        if self.task:
            return self.task
        responses = []
        for prompt in query_list:
            start_time = time.time()
            gemini_response: GenerateContentResponse = self.client.models.generate_content(
                model=self.model,
                contents=initial_prompt + prompt,
                config=types.GenerateContentConfig(system_instruction=system_context, temperature=self.temperature)
            )
            end_time = time.time()
            responses.append(
                LanguageModelResponse(
                    response=gemini_response.text,
                    elapsed_time=round((end_time - start_time), 3),
                    input_tokens=round(gemini_response.usage_metadata.prompt_token_count, 3),
                    output_tokens=round(gemini_response.usage_metadata.candidates_token_count, 3)
                )
            )
        self.task = LanguageModelTask(id=query_id, responses=responses)
        self.responses = responses
        return self.task

    def get_response(self, task: LanguageModelTask, timeout: int, retry: int) -> List[LanguageModelResponse]:
        if self.responses is not None:
            return self.responses
        return task.responses