import os
import pickle
from typing import List

from google import genai
from google.genai import types

from evalia.llm import LanguageModelManager, LanguageModelResponse, LanguageModelTask


class GeminiManager(LanguageModelManager):

    def __init__(self, model):
        super().__init__(model=model)
        self.client = self.initialize_client()

    @staticmethod
    def initialize_client():
        return genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

    def start_task(self, query_id: str, initial_prompt: str, query_list: List[str], system_context: str = "",
                   temperature: float = 0.0) -> LanguageModelTask:
        if self.task:
            return self.task
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
        self.task = LanguageModelTask(id=query_id, responses=responses)
        self.responses = responses
        return self.task

    def get_response(self, task: LanguageModelTask, timeout: int, retry: int) -> List[LanguageModelResponse]:
        if self.responses is not None:
            return self.responses
        return task.responses

    def get_llm_name(self) -> str:
        return "Gemini"

    def __getstate__(self):
        state = self.__dict__.copy()
        state["client"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.client = self.initialize_client()

def test_serialize():
    smart_manager = GeminiManager(model="gemini-2.0-flash")
    with open('data.pkl', 'wb') as file:
        pickle.dump(smart_manager, file)

def test_deserialize():
    with open('data.pkl', 'rb') as file:
        smart_manager = pickle.load(file)
    assert isinstance(smart_manager, GeminiManager)
    assert smart_manager.model == "gemini-2.0-flash"
    assert smart_manager.client is not None
