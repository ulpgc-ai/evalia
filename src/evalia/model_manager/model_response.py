class ModelResponse:
    def __init__(self, response: str):
        self.response: str = response
        self.elapsed_time: float = 0.0
        self.input_tokens: int = 0
        self.output_tokens: int = 0