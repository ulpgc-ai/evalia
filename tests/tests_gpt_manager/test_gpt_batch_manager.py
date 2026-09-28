import unittest
from unittest.mock import patch, mock_open

from evalia.gpt_manager.gpt_batch_manager import GPTBatchManager
from .utils import high_cost

import time
import random
import ast
import json
import os

QUERIES_FILE = "query-4ESO-17-deunaenuna.txt"

class TestGPTBatchManager(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Me sitúo en la misma carpeta que el script
        # para poder leer los ficheros de datos
        directorio_del_script = os.path.dirname(os.path.abspath(__file__))
        os.chdir(directorio_del_script)

    def test_build_jsonl_file(self):
        gpt_manager = GPTBatchManager(model="gpt-4o-mini")
        query_id = "test_build_jsonl_file"
        query_list = [
            [{"role": "user", "content": "Hola"}],
            [{"role": "user", "content": "How do you do?"}],
        ]
        temperature = 0.7

        expected_jsonl_content = (
            '{ "custom_id": "1", "method": "POST", '
            '"url": "/v1/chat/completions", '
            '"body": '
            '{ "model": "gpt-4o-mini", "temperature": 0.7, '
            '"messages": [{"role": "user", "content": "Hola"}] } }\n'
            '{ "custom_id": "2", "method": "POST", '
            '"url": "/v1/chat/completions", '
            '"body": '
            '{ "model": "gpt-4o-mini", "temperature": 0.7, '
            '"messages": [{"role": "user", "content": "How do you do?"}] } }\n'
        )

        # Solo se simula open() dentro de gpt_batch_manager: parchear
        # builtins.open afectaría también a otras bibliotecas (p. ej. al
        # cliente de OpenAI, que lee ficheros del sistema al importarse).
        with patch("evalia.gpt_manager.gpt_batch_manager.open", mock_open(), create=True) as mock_file, \
             patch("os.path.getsize", return_value=1000):
            jsonl_filename = gpt_manager._build_jsonl_file(query_id, query_list, temperature)

        mock_file.assert_called_once_with(f"{query_id}.jsonl", "w")
        mock_file().write.assert_any_call(expected_jsonl_content.split('\n')[0])
        mock_file().write.assert_any_call(expected_jsonl_content.split('\n')[1])
        self.assertEqual(jsonl_filename, f"{query_id}.jsonl")


    def test_build_jsonl_file_without_temperature(self):
        # PARCHE TEMPORAL (ver temperatura_2026.md): a los modelos que solo
        # admiten la temperatura por defecto no se les envía
        gpt_manager = GPTBatchManager(model="gpt-5.5")
        query_id = "test_build_jsonl_file_without_temperature"
        query_list = [[{"role": "user", "content": "Hola"}]]

        with patch("evalia.gpt_manager.gpt_batch_manager.open", mock_open(), create=True) as mock_file, \
             patch("os.path.getsize", return_value=1000):
            gpt_manager._build_jsonl_file(query_id, query_list, 0.7)

        json_line = mock_file().write.call_args_list[0].args[0]
        body = json.loads(json_line)["body"]
        self.assertEqual(body["model"], "gpt-5.5")
        self.assertEqual(body["messages"], query_list[0])
        self.assertNotIn("temperature", body)


    def test_start_task(self):
        gpt_manager = GPTBatchManager(model="gpt-4o-mini")
        query_id = "test_start_task"
        query_list = [
            [{"role": "user", "content": "Hola"}],
            [{"role": "user", "content": "How do you do?"}],
        ]
        temperature = 0.7

        task: GPTBatchManager.GPTBatchTask = gpt_manager.start_task(
            query_id, query_list, temperature
            )

        self.assertIsInstance(task, GPTBatchManager.GPTBatchTask)
        self.assertIsInstance(task.batch_id, str)
        #self.assertEqual(task.batch_id.endpoint, "/v1/chat/completions")
        #self.assertEqual(task.batch_id.completion_window, "24h")
        #self.assertEqual(task.batch_id.metadata, {"description": query_id})

        # tear down
        gpt_manager.cancel_task(task)
    
    @high_cost
    def test_start_task_complex(self):
        with open(QUERIES_FILE,'r',encoding='iso-8859-1') as query_file:
            queries = query_file.read()
            queries = ast.literal_eval(queries)
        gpt_manager = GPTBatchManager(model="gpt-4o-mini")
        query_id = "test_start_task_complex"

        task = gpt_manager.start_task(query_id, queries, temperature=0.7)

    @high_cost
    def test_get_response(self):
        with open(QUERIES_FILE,'r',encoding='iso-8859-1') as query_file:
            queries = query_file.read()
            queries = ast.literal_eval(queries)
        gpt_manager = GPTBatchManager(model="gpt-4o-mini")

        query_id = "test_query_20"
        task = gpt_manager.start_task(query_id, queries, temperature=0.7)
        response = gpt_manager.get_response(task)
        print(response)


class GPTExceptionError(Exception):
    pass

def mock_chat_completion_create(model, messages):
    time.sleep(2)
    print("------------------------------------------------------------------------")
    print(" Mocking chat completion create ")
    print("------------------------------------------------------------------------")
    json = {
            "choices": [
                {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "\u00a1Hola! \u00bfEn qu\u00e9 puedo ayudarte hoy?",
                    "role": "assistant"
                }
                }
            ],
            "created": 1697920381,
            "id": "chatcmpl-8CCwXw1ilJeHwUJ8KQHl8zxQVsskB",
            "model": "gpt-3.5-turbo-0613",
            "object": "chat.completion",
            "usage": {
                "completion_tokens": 11,
                "prompt_tokens": 8,
                "total_tokens": 19
            }
    }
    if random.random() < 0.5:
        raise GPTExceptionError("Ha habido una excepción desde la API de GPT")
    return json
        

@unittest.skip("Skip this class for now")
class TestGPTManager(unittest.TestCase):

    @patch('openai.chat.completions.create', side_effect=mock_chat_completion_create)
    def test_basic_interaction(self, mock_chat_completion_create):
        gpt_manager = GPTBatchManager(model="gpt-4")
        for _ in range(10):
            response = gpt_manager.query([{"role": "user", "content": "Hola"}])
        
        # check that the response contains the word "hola"
        self.assertIn("Hola", response[0].choices[0].message.content)
        self.assertEqual(response[0].choices[0].finish_reason, "stop")

    def test_bad_message(self):
        gpt_manager = GPTBatchManager(model="gpt-4")
        with self.assertRaises(Exception) as context:
            response = gpt_manager.query("this is a wrongly formatted message")

    def test_send_queries(self):
        gpt_manager = GPTBatchManager(model="gpt-4")
        query_list = [
            [{"role" : "user", "content" : "Hola"}],
            [{"role" : "user", "content" : "Hola, dime hola."}],
        ]
        responses, stats = gpt_manager.send_queries(
            query_id="test",
            query_list=query_list,
            temperature=0.0
        )
        self.assertEqual(len(responses), 2)
        # check that stats is a dict with a key "input_tokens" with a numeric value
        self.assertIsInstance(stats, dict)
        self.assertIn("input_tokens", stats)
        self.assertIn("output_tokens", stats)
        self.assertIn("elapsed_time", stats)

    def test_bad_queries(self):
        gpt_manager = GPTBatchManager(model="gpt-4")
        with self.assertRaises(Exception) as context:
            response = gpt_manager.send_queries(
                query_id="test",
                query_list= None,
                temperature=0.0       
                )
        with self.assertRaises(Exception) as context:
            response = gpt_manager.send_queries(
                query_id="test",
                query_list="bad query, should be an OpenAI message or a list of OpenAI messages",
                temperature=0.0
                )
        with self.assertRaises(Exception) as context:
            response = gpt_manager.send_queries(
                query_id="test",
                query_list=[{"role" : "user", "content" : "Hola"}, "bad query"],
               temperature=0.0
                 )

       
if __name__ == '__main__':
    unittest.main()