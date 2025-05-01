from dataclasses import dataclass


@dataclass
class LanguageModelResponse:
    def __init__(self, response: str, elapsed_time: float, input_tokens: int, output_tokens: int):
        self.response: str = response
        self.elapsed_time: float = elapsed_time
        self.input_tokens: int = input_tokens
        self.output_tokens: int = output_tokens