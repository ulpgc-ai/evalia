import time
from datetime import datetime, timezone
from typing import List

from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages import MessageBatch, batch_create_params
from anthropic.types.messages.batch_create_params import Request

from evalia.llm import BatchManager, LanguageModelTask, LanguageModelResponse
from evalia.llm.claude import ClaudeManager



class ClaudeBatchManager(ClaudeManager, BatchManager):

    def __init__(self, model: str, temperature: float = 0.0):
        super().__init__(model=model, temperature = temperature)

    def start_task(self, query_id: str, initial_prompt: str, query_list: List[str], system_context: str = "") -> LanguageModelTask:
        if self.task is not None:
            return self.task
        requests: List[batch_create_params.Request] = []
        for query in query_list:
            requests.append(
                Request(
                    custom_id=query_id + "_" + str(len(requests) + 1),
                    params=MessageCreateParamsNonStreaming(
                        model=self.model,
                        max_tokens=1024,
                        messages=self.convert_to_claude_messages(query, initial_prompt),
                        system=system_context,
                        temperature=self.temperature
                    )
                )
            )
        message_batch: MessageBatch = self.client.messages.batches.create(requests= requests)
        self.task = LanguageModelTask(id=message_batch.id)
        return self.task

    def get_response(self, task: LanguageModelTask, timeout: int, retry: int) -> List[LanguageModelResponse]:
        if self.responses is not None:
            return self.responses
        batch_retrieved: MessageBatch = self.client.messages.batches.retrieve(task.id)
        while batch_retrieved.processing_status != "ended":
            print(f"Batch task {task.id} status is {batch_retrieved.processing_status}. Waited {ClaudeBatchManager.calculate_waited_time_seconds(batch_retrieved.created_at)} seconds...")
            time.sleep(retry)
            batch_retrieved = self.client.messages.batches.retrieve(task.id)
        llm_responses: List[LanguageModelResponse] = []
        for batch_result in self.client.messages.batches.results(task.id):
            match batch_result.result.type:
                case "succeeded":
                    llm_responses.append(LanguageModelResponse(
                        response = ClaudeManager.parse_claude_response(batch_result.result.message),
                        elapsed_time = ClaudeBatchManager.calculate_waited_time_seconds(batch_retrieved.created_at, batch_retrieved.ended_at),
                        input_tokens = batch_result.result.message.usage.input_tokens,
                        output_tokens= batch_result.result.message.usage.output_tokens
                    ))
                case "errored":
                    if batch_result.result.error.type == "invalid_request":
                        print(f"Validation error {batch_result.custom_id}")
                    else:
                        print(f"Server error {batch_result.custom_id}")
                case "expired":
                    print(f"Request expired {batch_result.custom_id}")
        self.responses = llm_responses
        return self.responses

    @staticmethod
    def calculate_waited_time_seconds(created_at: datetime, ended_at: datetime):
        return round((ended_at - created_at).total_seconds(), 3)

    def cancel_task(self, task: LanguageModelTask):
        self.client.messages.batches.cancel(task.id)

    def save_task(self, task: LanguageModelTask, filename: str):
        with open(filename, "w") as f:
            f.write(task.id)
        print(f"Task {task.id} saved to {filename}")

    def load_task(self, filename: str) -> LanguageModelTask:
        with open(filename, "r") as f:
            batch_id = f.read()
        return LanguageModelTask(batch_id)