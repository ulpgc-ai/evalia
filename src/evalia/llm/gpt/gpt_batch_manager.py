""" 
Diálogo mediante la Batch API de OpenAI.
Habla con GPT a través de la API de OpenAI. Se le entrega una lista
de consultas, cada una de ellas en el formato JSON de OpenAI. 
El sistema devuelve las respuestas de GPT.
"""

from typing import List

from openai.types import Batch
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

    def start_task(self, query_id: str, initial_prompt: str, query_list: List[str], system_context: str = "") -> LanguageModelTask:
        if self.task is not None:
            return self.task
        # Prepare the JSONL file with the batch
        batch_jsonl_file = self._build_jsonl_file(query_id, initial_prompt, query_list, system_context)

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
        logger.info(f'Batch ID "{query_id}". queries={len(query_list)} temperature={self.temperature}')

        self.task = LanguageModelTask(batch.id)
        # return the task
        return LanguageModelTask(batch.id)

    def _build_jsonl_file(self, query_id: str, initial_prompt: str, query_list: List[str], system_context: str = "") -> str:
            """
            Build the JSONL file for the batch task.
            :param query_id: the ID of the query
            :param query_list: list of queries to process
            :return: JSONL filename
            """
            assert len(query_list) <= MAX_REQUESTS_PER_BATCH, \
                f"Number of requests {len(query_list)} exceeds limit {MAX_REQUESTS_PER_BATCH}"

            # Generate the schema from the structured_output_class
            schema = self.structured_output_class.model_json_schema()
            schema["additionalProperties"] = False
            schema_dict = {
                "name": self.structured_output_class.__name__,
                "strict": True,
                "schema": schema
            }


            JSONL_TEMPLATE = (
                '{{ '
                    '"custom_id": "{custom_id}", '
                    '"method": "POST", '
                    '"url": "/v1/chat/completions", '
                    '"body": {{ '
                        '"model": "{model}", '
                        '"temperature": {temperature}, '
                        '"messages": {messages}, '
                            '"response_format": {{ '
                                '"type": "json_schema", '
                                '"json_schema": {json_schema} '
                            '}} '
                    '}} '
                '}}'
            )

            jsonl_filename = f"{query_id}.jsonl"
            query_counter = 1
            with open(jsonl_filename, "w") as f:
                for query in query_list:
                    json_line = JSONL_TEMPLATE.format(
                        custom_id=query_counter,
                        model=self.model,
                        temperature=self.temperature,
                        messages=json.dumps(GPTManager.convert_to_gpt_messages(query, initial_prompt, system_context)),
                        json_schema = json.dumps(schema_dict)
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
        if self.responses is not None:
            return self.responses
        start_time = time.time()
        batch_id = task.id
        batch_object: Batch = self.client.batches.retrieve(batch_id)

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

        elapsed_time = batch_object.completed_at - batch_object.created_at

        lm_responses = []
        for r in openai_responses:
            lm_responses.append(LanguageModelResponse(
                response=r.choices[0].message.content,
                elapsed_time=elapsed_time,
                input_tokens=r.usage.prompt_tokens,
                output_tokens=r.usage.completion_tokens,
            ))
        self.responses = lm_responses
        return lm_responses