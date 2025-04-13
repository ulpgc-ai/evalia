""" 
Diálogo mediante la Batch API de OpenAI.
Habla con GPT a través de la API de OpenAI. Se le entrega una lista
de consultas, cada una de ellas en el formato JSON de OpenAI. 
El sistema devuelve las respuestas de GPT.
"""

from typing import List
from openai.types.chat.chat_completion import ChatCompletion
from evalia.llm import LanguageModelTask, LanguageModelResponse, BatchManager
import os
import json
import time

from evalia.llm.gpt import GPTManager
from evalia.logs import get_logger

# OpenAI limits
MAX_REQUESTS_PER_BATCH = 50_000
MAX_BATCH_SIZE = 100_000_000

# Logging
logger = get_logger(__name__)

class GPTBatchManager(GPTManager, BatchManager):
    """
    GPTManager that uses the OpenAI Batch API.
    Sends queries in batches for deferred response.
    """

    def __init__(self,model=""):
        super().__init__(model=model)
        self.batch_api = True
        logger.info("-----------------------------------")
        logger.info(f"GPTBatchManager started. Model: {self.model}")

    def generate_text(self, query_id: str, initial_prompt: str, query_list: List[str], system_context: str = "", temperature: float = 0.0) -> List[LanguageModelResponse]:
        task = self.start_task(query_id, query_list, temperature)
        return self.get_response(task)

    def start_task(self, query_id: str, query_list: List[str], temperature: float) -> LanguageModelTask:
        # Prepare the JSONL file with the batch
        batch_jsonl_file = self._build_jsonl_file(
            query_id, query_list, temperature
            )

        # Upload the file to OpenAI file storage
        with open(batch_jsonl_file, 'rb') as f:
            batch_input_file = self.client.files.create(
                file=f,
                purpose="batch"
            )

        # Remove the JSONL file
        os.remove(batch_jsonl_file)

        # Create the batch in OpenAI
        batch = self.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": query_id
            },
        )

        logger.info(f'Batch task created. Internal ID: "{query_id}". OpenAI ID: {batch.id}')
        logger.info(f'Batch ID "{query_id}". queries={len(query_list)} temperature={temperature}')

        # return the task
        return LanguageModelTask(batch.id)

    def _build_jsonl_file(self,query_id, query_list, temperature) -> str:
            """
            Build the JSONL file for the batch task.
            :param query_id: the ID of the query
            :param query_list: list of queries to process
            :param temperature: temperature for the model
            :return: JSONL filename
            """
            assert len(query_list) <= MAX_REQUESTS_PER_BATCH, \
                f"Number of requests {len(query_list)} exceeds limit {MAX_REQUESTS_PER_BATCH}"

            JSONL_TEMPLATE = (
                '{{ "custom_id": "{custom_id}", "method": "POST", '
                '"url": "/v1/chat/completions", '
                '"body": '
                  '{{ "model": "{model}", "temperature": {temperature}, '
                     '"messages": {messages} }} '
                '}}'
            )

            jsonl_filename = f"{query_id}.jsonl"
            query_counter = 1
            with open(jsonl_filename, "w") as f:
                for query in query_list:
                    json_line = JSONL_TEMPLATE.format(
                        custom_id=query_counter,
                        model=self.model,
                        temperature=temperature,
                        messages=json.dumps(query)
                    )
                    f.write(json_line)
                    f.write("\n")
                    query_counter += 1
            # get file size
            file_size = os.path.getsize(jsonl_filename)
            assert file_size <= MAX_BATCH_SIZE, \
                f"File size {file_size} exceeds limit {MAX_BATCH_SIZE}"
            return jsonl_filename

    def cancel_task(self, task: LanguageModelTask):
        self.client.batches.cancel(task.id)

    def save_task(self, task: LanguageModelTask, filename: str):
        with open(filename, "w") as f:
            f.write(task.id)

    def load_task(self, filename: str) -> LanguageModelTask:
        with open(filename, "r") as f:
            batch_id = f.read()
        return LanguageModelTask(batch_id)
    
    # TODO: ordenar las respuestas según "custom_id" y 
    # poner respuestas nulas en las omitidas
    def get_response(self, task: LanguageModelTask, timeout=0, retry=1) -> List[LanguageModelResponse]:
        start_time = time.time()
        batch_id = task.id
        batch_object = self.client.batches.retrieve(batch_id)

        # Wait until batch reaches a final state or timeout is reached
        running_states = ["validating", "in_progress", "finalizing", "cancelling"]
        while batch_object.status in running_states:
            if timeout > 0 and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Timeout reached for batch {batch_id}.")
            print((
                f"Batch {batch_id} is {batch_object.status}. "
                f"{batch_object.request_counts.completed}/{batch_object.request_counts.total} completed. "
                f"Waited {(time.time() - batch_object.created_at):.2f} seconds..."
            ))
            if retry > 0:
                time.sleep(retry)
                batch_object = self.client.batches.retrieve(batch_id)
            else:
                break

        if batch_object.status != "completed":
            raise Exception(f"Batch {batch_id} failed: {batch_object.errors}")
        
        batch_file = self.client.files.content(batch_object.output_file_id)

        # extract the GPT responses from the batch output file
        lines = batch_file.content.splitlines()
        dict_responses = [ json.loads(l)['response']['body'] for l in lines ]
        openai_responses = [ 
            ChatCompletion.model_validate(r) 
            for r in dict_responses
            ]
        
        # calculate stats
        final_times = [ 
            batch_object.cancelled_at, 
            batch_object.completed_at,
            batch_object.expired_at,
            batch_object.failed_at,
        ]
        end_time = max([ t for t in final_times if t is not None ])
        elapsed_time = end_time - batch_object.created_at

        lm_responses = []
        for r in openai_responses:
            lm_responses.append(LanguageModelResponse(
                response=r.choices[0].message.content,
                elapsed_time=elapsed_time,
                input_tokens=r.usage.prompt_tokens,
                output_tokens=r.usage.completion_tokens,
            ))
        return lm_responses