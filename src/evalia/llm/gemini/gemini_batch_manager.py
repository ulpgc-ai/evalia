import json
import os
import time
from typing import List, Type

from google.cloud.storage import Client, Bucket, Blob
from google.genai.types import CreateBatchJobConfig, BatchJob, JobState
from google.cloud import storage
from pydantic import BaseModel

from evalia.llm import BatchManager, LanguageModelTask, LanguageModelResponse
from evalia.llm.evaluated_answer import EvaluatedJustifiedAnswers
from evalia.llm.gemini import GeminiManager


class GeminiBatchManager(GeminiManager, BatchManager):

    completed_states = {
        JobState.JOB_STATE_SUCCEEDED,
        JobState.JOB_STATE_FAILED,
        JobState.JOB_STATE_CANCELLED,
        JobState.JOB_STATE_PAUSED,
    }

    def __init__(self, model: str, temperature: float = 0.0, structured_output_class: Type[BaseModel] = EvaluatedJustifiedAnswers):
        super().__init__(model=model, temperature=temperature, structured_output_class=structured_output_class, use_vertex=True)
        self.storage_client: Client = storage.Client()
        self.bucket_name = os.getenv("GOOGLE_CLOUD_STORAGE_BUCKET_NAME") # for example "evalia-test"
        self.destination_uri = os.getenv("GOOGLE_CLOUD_STORAGE_JSONL_DESTINATION_URI") # for example "data/input". This is where the JSONL file will be saved
        self.output_uri = os.getenv("GOOGLE_CLOUD_STORAGE_BATCH_OUTPUT_URI") # for example "data/output". This is where the batch results will be saved

    def start_task(self, query_id: str, initial_prompt: str, query_list: List[str], system_context: str = "") -> LanguageModelTask:
        if self.task is not None:
            return self.task
        jsonl_file_name = self.create_jsonl_file(query_id, initial_prompt, query_list, self.temperature, self.structured_output_class, system_context)
        gcs_uri = self.upload_jsonl_to_google_cloud_storage(jsonl_file_name)
        self.delete_jsonl_file(jsonl_file_name)
        job: BatchJob = self.client.batches.create(
            model=self.model,
            src=gcs_uri,
            config = CreateBatchJobConfig(dest=self.build_uri(self.bucket_name, self.output_uri, query_id))
        )
        print(f"Job name: {job.name}")
        print(f"Job state: {job.state}")
        self.task = LanguageModelTask(id=job.name)
        return LanguageModelTask(id=job.name)


    @staticmethod
    def build_uri(bucket_name: str, destination_uri: str, query_id: str) -> str:
        return f"gs://{bucket_name}/{destination_uri}/{query_id}"

    @staticmethod
    def create_jsonl_file(query_id: str, initial_prompt: str, query_list: List[str], temperature: float, structured_output_class: Type[BaseModel], system_context: str = "") -> str:
        file_name = f"{query_id}.jsonl"
        with open(file_name, 'w') as f:
            for i in range(1, len(query_list) + 1):
                prompt_text = initial_prompt + query_list[i - 1]
                request_obj = {
                    "id": i,
                    "request": {
                        "contents": [
                            {
                                "parts": {
                                    "text": prompt_text
                                },
                                "role": "user"
                            }
                        ],
                        "generationConfig": {
                            "temperature": temperature,
                            "responseMimeType": "application/json",
                            "responseSchema": GeminiBatchManager.convert_to_vertex_compatible_schema(structured_output_class)
                        }
                    }
                }
                if system_context:
                    request_obj["request"]["system_instruction"] = {
                        "parts": [
                            {"text": system_context}
                        ]
                    }
                f.write(json.dumps(request_obj) + "\n")
        return file_name

    @staticmethod
    def convert_to_vertex_compatible_schema(structured_output_class: Type[BaseModel]) -> dict:
        raw_schema = structured_output_class.model_json_schema()
        resolved_schema = raw_schema.copy()
        if "$defs" in resolved_schema:
            if "properties" in resolved_schema and "results" in resolved_schema["properties"]:
                if "items" in resolved_schema["properties"]["results"] and "$ref" in \
                        resolved_schema["properties"]["results"]["items"]:
                    ref_path = resolved_schema["properties"]["results"]["items"]["$ref"]
                    ref_name = ref_path.split("/")[-1]
                    if ref_name in resolved_schema["$defs"]:
                        resolved_schema["properties"]["results"]["items"] = resolved_schema["$defs"][ref_name]
            del resolved_schema["$defs"]
        return resolved_schema

    def upload_jsonl_to_google_cloud_storage(self, local_file_name: str) -> str:
        try:
            bucket: Bucket = self.storage_client.bucket(self.bucket_name)
            blob: Blob = bucket.blob(self.destination_uri + "/" + local_file_name)
            blob.upload_from_filename(local_file_name)
            gcs_uri = f"gs://{self.bucket_name}/{self.destination_uri}/{local_file_name}"
            print(f"Archivo {local_file_name} subido a {gcs_uri}")
            return gcs_uri

        except Exception as e:
            print(f"Error al subir el archivo: {e}")
            print("Asegúrate de que el bucket existe y tienes permisos de escritura.")
            print("Asegúrate de que la ruta del archivo local es correcta.")

    def download_file_from_google_cloud_storage(self, remote_file_name: str) -> str:
        bucket: Bucket = self.storage_client.bucket(self.bucket_name)
        blob: Blob = bucket.blob(self.output_uri + "/" + remote_file_name)
        jsonl = "resultado_batch.jsonl"
        blob.download_to_filename(jsonl)
        print(f"Archivo {remote_file_name} descargado a {jsonl}")
        return jsonl

    def delete_jsonl_file(self, local_file_name: str):
        try:
            os.remove(local_file_name)
            print(f"Archivo {local_file_name} eliminado.")
        except Exception as e:
            print(f"Error al eliminar el archivo: {e}")
            print("Asegúrate de que la ruta del archivo local es correcta.")

    def get_response(self, task: LanguageModelTask, timeout: int, retry: int) -> List[LanguageModelResponse]:
        if self.responses is not None:
            return self.responses
        job: BatchJob = self.client.batches.get(name=task.id)
        while job.state not in self.completed_states:
            if 0 < timeout < (time.time() - job.create_time.timestamp()):
                raise TimeoutError(f"Timeout reached for batch {task.id}.")
            print((
                f"Batch {task.id} is {job.state}. "
                f"The batch job was recently modified at {job.update_time}. "
                f"Waited {(time.time() - job.create_time.timestamp()):.2f} seconds..."
            ))
            if retry > 0:
                time.sleep(retry)
                job = self.client.batches.get(name=task.id)
            else:
                break
        if job.state != JobState.JOB_STATE_SUCCEEDED:
            raise Exception(f"Batch {task.id} failed: {job.errors}")
        batch_elapsed_time = round(job.end_time.timestamp() - job.create_time.timestamp(), 3)
        llm_responses: List[LanguageModelResponse] = []
        json_list = self.find_json_list_in_gcs()
        for json_dict in json_list:
            llm_responses.append(LanguageModelResponse(
                response = json_dict['response']['candidates'][0]['content']['parts'][0]['text'],
                elapsed_time = batch_elapsed_time / len(json_list),
                input_tokens = json_dict['response']['usageMetadata']['promptTokenCount'],
                output_tokens = json_dict['response']['usageMetadata']['candidatesTokenCount'],
            ))
        self.responses = llm_responses
        return llm_responses

    def find_json_list_in_gcs(self):
        bucket: Bucket = self.storage_client.bucket(self.bucket_name)
        blobs = bucket.list_blobs(prefix=self.output_uri)
        json_list = []
        for blob in blobs:
            if blob.name.endswith(".jsonl"):
                content = blob.download_as_text()
                for line in content.splitlines():
                    try:
                        json_list.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Error al decodificar JSON en la línea: {line} - {e}")
            if len(json_list) > 0:
                print(f"Encontrado {len(json_list)} archivos JSONL en el bucket {self.bucket_name} con prefijo {self.output_uri}.")
                break
        return json_list

    def cancel_task(self, task: LanguageModelTask):
        self.client.batches.cancel(name=task.id)
        self.task = None
        self.responses = None

    def save_task(self, task: LanguageModelTask, filename: str):
        with open(filename, "w") as f:
            f.write(task.id)

    def load_task(self, filename: str) -> LanguageModelTask:
        with open(filename, "r") as f:
            batch_id = f.read()
        return LanguageModelTask(batch_id)

    def __getstate__(self):
        state = super().__getstate__()
        state["storage_client"] = None
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self.storage_client = storage.Client()
