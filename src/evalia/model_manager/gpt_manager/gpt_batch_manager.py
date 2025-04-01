""" 
Diálogo mediante la Batch API de OpenAI.
Habla con GPT a través de la API de OpenAI. Se le entrega una lista
de consultas, cada una de ellas en el formato JSON de OpenAI. 
El sistema devuelve las respuestas de GPT.

Métodos:

- initialize(): pone a punto el sistema
- count_tokens(): no implementado
- send_queries(): envía un lote de peticiones a GPT
"""

from typing import Tuple
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from . import GPTManager, GPTTask
import os
import json
import time
from src.evalia.logs import get_logger

# OpenAI limits
MAX_REQUESTS_PER_BATCH = 50_000
MAX_BATCH_SIZE = 100_000_000

# Logging
logger = get_logger(__name__)

class GPTBatchManager(GPTManager):
    '''
    GPTManager que utiliza la Batch API de OpenAI.
    Envía las consultas en lotes de respuesta diferida.
    '''

    class GPTBatchTask(GPTTask):
        '''
        Tarea de GPTBatchManager.
        '''
        def __init__(self, batch_id):
            self.batch_id = batch_id

    def __init__(self,model=""):
        self.initialize(model)
        self.batch_api = True
        logger.info(f"-----------------------------------")
        logger.info(f"GPTBatchManager started. Model: {self.model}")

    def initialize(self,model=""):
        """Inicializa el sistema de GPT"""
        self.client = OpenAI()
        self.model = model

    def count_tokens(self,messages):
        """No implementado"""
        return -1

    def start_task(self, query_id, query_list, temperature) -> GPTTask:
        '''
        Envía una lista de consultas a GPT. Devuelve un id de batch.
        '''

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
        return self.GPTBatchTask(batch.id)

    def _build_jsonl_file(self,query_id, query_list, temperature):
            '''
            Prepara un archivo JSONL con las consultas.
            '''
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

    def cancel_task(self, task: GPTBatchTask):
        '''Cancela una tarea.'''
        self.client.batches.cancel(task.batch_id)

    def save_task(self, task: GPTTask, filename: str):
        '''Guarda una tarea en un archivo.'''
        with open(filename, "w") as f:
            f.write(task.batch_id)

    def load_task(self, filename: str) -> GPTTask:
        '''Carga una tarea desde un archivo.'''
        with open(filename, "r") as f:
            batch_id = f.read()
        return self.GPTBatchTask(batch_id)
    
    # TODO: ordenar las respuestas según "custom_id" y 
    # poner respuestas nulas en las omitidas
    def get_response(self, task: GPTBatchTask,
                     timeout=0, retry=1) -> Tuple[list,dict]:
        '''
        Obtiene la respuesta de una tarea.

        Args:
            task: tarea de GPTBatchManager.
            timeout: tiempo máximo de espera (segundos). Si es cero, se espera indefinidamente.
            retry: tiempo de espera entre intentos (segundos). Si es cero, solo intenta una vez.
        '''
        start_time = time.time()
        batch_id = task.batch_id
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
                         
        stats = {
            "input_tokens": 
            sum([ r.usage.prompt_tokens for r in openai_responses ]),

            "output_tokens": 
            sum([ r.usage.completion_tokens for r in openai_responses ]),

            "elapsed_time": elapsed_time,
        }

        # bye bye
        return openai_responses, stats

    
    def send_queries(self, query_id, query_list, temperature) -> Tuple[list, dict]:
        task = self.start_task(query_id, query_list, temperature)
        return self.get_response(task)

# SOME TESTS

def test_gpt_batch():
    gpt_manager = GPTBatchManager(model="gpt-4o")
    query_id = "test_query"
    query_list_simple = [
        [{"role": "user", "content": "Hola"}],
        [{"role": "user", "content": "¿Cómo estás?"}]
    ]

    idiomas = [ "español", "inglés", "francés", "alemán", "italiano", "portugués" ]
    query_list_200 = [ 
        [{"role": "user", 
          "content": 
          f"dame una palabra en {idiomas[i%len(idiomas)]} de {2+i%20} letras"}]
        for i in range(0,200)
     ]
    temperature = 0.5

    task = gpt_manager.start_task(query_id=query_id, 
                           query_list=query_list_200, 
                           temperature=temperature
                           )
    
    messages, stats = gpt_manager.get_response(task)

def get_response_from_task(batch_id):
    gpt_manager = GPTBatchManager()
    task = GPTBatchManager.GPTBatchTask(batch_id)

    messages, stats = gpt_manager.get_response(task)

    text_responses = [ m.choices[0].message.content for m in messages ]
    for i,r in enumerate(text_responses, start=1):
        print(f"{i}. {r}")


if __name__ == "__main__":
   #test_gpt_batch()
   get_response_from_task("batch_6720b8474d988190bb6f5d95caa05619")
   pass