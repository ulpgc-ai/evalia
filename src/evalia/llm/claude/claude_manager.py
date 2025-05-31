import json
from abc import abstractmethod
from typing import List, Literal, Type

import anthropic
import yaml
from anthropic.types import MessageParam, Message
from pydantic import BaseModel

from evalia.llm import LanguageModelManager, LanguageModelTask, LanguageModelResponse
from evalia.llm.evaluated_answer import EvaluatedJustifiedAnswers


class ClaudeManager(LanguageModelManager):

    def __init__(self, model: str, temperature: float = 0.0, structured_output_class: Type[BaseModel] = EvaluatedJustifiedAnswers):
        super().__init__(model, temperature, structured_output_class)
        self.client = self.initialize_client()

    @staticmethod
    def initialize_client():
        return anthropic.Anthropic()

    @staticmethod
    def create_claude_input_message(role: Literal['user', 'assistant'], prompt: str) -> MessageParam:
        return {
            "role": role,
            "content": prompt
        }

    def convert_to_claude_messages(self, query: str, initial_prompt: str) -> List[dict]:
        """
        Builds a message list formatted for the Anthropic Chat API.
        :param query: The user query or batch of responses to be evaluated.
        :param initial_prompt: The initial prompt to be used for the query, with initial context and the question to be evaluated.
        :return: A list of messages structured in the format expected by Anthropìc.
        """
        return [
            ClaudeManager.create_claude_input_message("user", initial_prompt),
            self.generate_yaml_structure_prompt(),
            ClaudeManager.create_claude_input_message("user", query),
            ClaudeManager.create_claude_input_message("assistant", "results:"),
        ]

    def generate_yaml_structure_prompt(self) -> MessageParam:
        """
        Generate a string representing the YAML format expected, based on the structured_output_class.
        """
        justified_answers = self.structured_output_class is EvaluatedJustifiedAnswers
        comment_additional_field = 'and "comment" (required string), this is a justification of the score' if justified_answers else ''
        prompt = f"""
            Output in YAML format with keys: 
            "results" (list of dicts with "index" (0, 1, 2, ..., this is the index of the answer in the list), 
            "score" (float) {comment_additional_field}. Use block scalars (`|`) to format multi-line strings properly. Do not use inline strings for comments.
            Here are the answers (JUST EVALUATE THEM):             
        """
        return self.create_claude_input_message("user", prompt)

    @staticmethod
    def parse_claude_response(response: Message) -> str:
        return json.dumps(yaml.safe_load("results: " + response.content[0].text))

    def __getstate__(self):
        state = self.__dict__.copy()
        state["client"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.client = self.initialize_client()

    @abstractmethod
    def start_task(self, query_id: str, initial_prompt: str, query_list: List[str], system_context: str = "") -> LanguageModelTask:
        pass

    @abstractmethod
    def get_response(self, task: LanguageModelTask, timeout: int, retry: int) -> List[LanguageModelResponse]:
        pass

    def get_llm_name(self) -> str:
        return "Claude"