import os
from typing import List

from google import genai
from google.genai import types

from evalia.llm import LanguageModelManager, LanguageModelResponse


class GeminiManager(LanguageModelManager):

    def __init__(self, model):
        super().__init__(model=model)
        self.client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

    def generate_text(self, query_id: str, initial_prompt: str, query_list: List[str], system_context: str = "", temperature: float = 0.0) -> List[LanguageModelResponse]:
        responses = []
        for prompt in query_list:
            gemini_response = self.client.models.generate_content(
                model=self.model,
                contents=initial_prompt + prompt,
                config=types.GenerateContentConfig(system_instruction=system_context)
            )
            responses.append(
                LanguageModelResponse(
                    response=gemini_response.text,
                    elapsed_time=0,
                    input_tokens=len(prompt.split()),
                    output_tokens=len(gemini_response.text.split())
                )
            )
        return responses

    def get_llm_name(self) -> str:
        return "Gemini"