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
        """
        Initializes the Anthropic client for interacting with Claude's API.

        :return: An instance of the Anthropic client.
        """
        return anthropic.Anthropic()

    @staticmethod
    def create_claude_input_message(role: Literal['user', 'assistant'], prompt: str) -> MessageParam:
        """
        Creates a message in the format expected by the Anthropic Chat API.

        :param role: The role of the message sender, either 'user' or 'assistant'.
        :param prompt: The content of the message to be sent.
        :return: A dictionary representing the message in the format expected by the Anthropic API.
        """
        return {
            "role": role,
            "content": prompt
        }

    def convert_to_claude_messages(self, query: str, initial_prompt: str) -> List[dict]:
        """
        Builds a message list formatted for the Anthropic Chat API.

        :param query: The user query or batch of responses to be evaluated.
        :param initial_prompt: The initial prompt to be used for the query, with initial context and the question to be evaluated.
        :return: A list of messages structured in the format expected by Anthropic.
        """
        return [
            ClaudeManager.create_claude_input_message("user", initial_prompt),
            self.generate_yaml_structure_prompt(),
            ClaudeManager.create_claude_input_message("user", query),
            ClaudeManager.create_claude_input_message("assistant", "results:"),
        ]

    def generate_yaml_structure_prompt(self) -> MessageParam:
        """
        Generates a string representing the YAML format expected, based on the structured_output_class.

        :return: A MessageParam object containing the YAML structure prompt.
        """
        justified_answers = self.structured_output_class is EvaluatedJustifiedAnswers
        comment_additional_field = ', and "comment" (required string), this is a justification of the score. . Use block scalars (`|`) to format multi-line strings properly. Do not use inline strings for comments.' if justified_answers else '.'
        prompt = f"""
            Output in YAML format with keys: 
            "results" (list of dicts with "index" (0, 1, 2, ..., this is the index of the answer in the list), 
            "score" (float){comment_additional_field}
            Don't send additional messages, just the YAML output.             
        """
        return self.create_claude_input_message("user", prompt)

    @staticmethod
    def parse_claude_response(response: Message) -> str:
        """
        Parses the YAML response from Claude's API and converts it to a JSON string.

        :param response: The response from Claude's API, expected to have a 'text' field containing YAML content.
        :return: A JSON string representation of the parsed YAML content.
        """
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