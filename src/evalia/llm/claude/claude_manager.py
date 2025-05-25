import json
from abc import abstractmethod
from typing import List, Literal, Type

import anthropic
from anthropic.types import MessageParam
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
            self.generate_json_structure_prompt(),
            ClaudeManager.create_claude_input_message("assistant",
                                                      "Sí, he entendido las instrucciones. Pásame las respuestas para evaluar."),
            ClaudeManager.create_claude_input_message("user", query)
        ]

    def generate_json_structure_prompt(self) -> MessageParam:
        """
        Generate a string representing the JSON format expected, based on the structured_output_class.
        """
        justified_answers = self.structured_output_class is EvaluatedJustifiedAnswers
        comment_additional_field = 'and "comment" (required string), this is a justification of the score' if justified_answers else ''
        comment_bad_answer = '"comment": "1+1 is not equal to 3"' if justified_answers else ''
        comment_good_answer = '"comment": "1+1 is equal to 2"' if justified_answers else ''
        prompt = f"""
            Analyze these answers and output in JSON format (escape special chars properly) with keys: 
            "results" (list of dicts with "index" (0, 1, 2, ..., this is the index of the answer in the list), 
            "score" (float)
            {comment_additional_field}.
            Remember to escape special characters in JSON, such as " (double quotes), \ (backslash), \n (newline), \t (tab), and \r (carriage return), 
            to ensure proper syntax and avoid parsing errors.
            Example of expected output:
            {{
                "results": [
                    {{
                        "index": 0,
                        "score": 0
                        {comment_bad_answer}
                    }},
                    {{
                        "index": 1,
                        "score": 1
                        {comment_good_answer}
                    }}
                ]
            }}            
        """
        return self.create_claude_input_message("user", prompt)

    @staticmethod
    def is_valid_json(input_string: str) -> bool:
        try:
            json.loads(input_string)
            return True
        except json.JSONDecodeError:
            return False

    @staticmethod
    def parse_claude_response(raw: str) -> str:
        if ClaudeManager.is_valid_json(raw):
            return raw
        raw = raw.strip('"')
        if ClaudeManager.is_valid_json(raw):
            return raw
        else:
            raise ValueError("No se pudo parsear correctamente la respuesta de Claude.")

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