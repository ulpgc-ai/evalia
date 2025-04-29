import pickle
import unittest
from unittest.mock import patch, mock_open

from evalia.llm.gpt import GPTBatchManager

from evalia.llm import LanguageModelTask
from .utils import high_cost

import time
import random
import ast
import os

QUERIES_FILE = "query-4ESO-17-deunaenuna.txt"

class TestGPTBatchManager(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Me sitúo en la misma carpeta que el script
        # para poder leer los ficheros de datos
        directorio_del_script = os.path.dirname(os.path.abspath(__file__))
        os.chdir(directorio_del_script)

    @patch("os.path.getsize", return_value=1000)
    @patch("builtins.open", new_callable=mock_open)
    def test_build_jsonl_file(self, mock_file, mock_getsize):
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

        # Mocked file writing and checking file size
        jsonl_filename = gpt_manager._build_jsonl_file(query_id, query_list, temperature)


        mock_file.assert_called_once_with(f"{query_id}.jsonl", "w")
        mock_file().write.assert_any_call(expected_jsonl_content.split('\n')[0])
        mock_file().write.assert_any_call(expected_jsonl_content.split('\n')[1])
        self.assertEqual(jsonl_filename, f"{query_id}.jsonl")


    def test_start_task(self):
        gpt_manager = GPTBatchManager(model="gpt-4o-mini")
        query_id = "test_start_task"
        query_list = [
            [{"role": "user", "content": "Hola"}],
            [{"role": "user", "content": "How do you do?"}],
        ]
        temperature = 0.7

        task: LanguageModelTask = gpt_manager.start_task(query_id, query_list, temperature)

        self.assertIsInstance(task, LanguageModelTask)
        self.assertIsInstance(task.id, str)
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
        

class TestGPTManager(unittest.TestCase):

    def test_serialize(self):
        gpt_batch_manager = GPTBatchManager(model="gpt-4o-mini")
        gpt_batch_manager.task = LanguageModelTask("test_task")
        gpt_file_name = 'data.pkl'
        self.addCleanup(lambda: os.remove(gpt_file_name) if os.path.exists(gpt_file_name) else None)


        with open(gpt_file_name, 'wb') as file:
            pickle.dump(gpt_batch_manager, file)
        self.assertTrue(os.path.exists(gpt_file_name))

        with open(gpt_file_name, 'rb') as file:
            loaded_manager = pickle.load(file)

        self.assertIsInstance(loaded_manager, GPTBatchManager)
        self.assertEqual(loaded_manager.model, "gpt-4o-mini")
        self.assertIsNotNone(loaded_manager.client)
        self.assertIsInstance(loaded_manager.task, LanguageModelTask)
        self.assertEqual(loaded_manager.task.id, "test_task")


    @unittest.skip("Skip this test for now")
    @patch('openai.chat.completions.create', side_effect=mock_chat_completion_create)
    def test_basic_interaction(self, mock_chat_completion_create):
        gpt_manager = GPTBatchManager(model="gpt-4")
        for _ in range(10):
            response = gpt_manager.query([{"role": "user", "content": "Hola"}])
        
        # check that the response contains the word "hola"
        self.assertIn("Hola", response[0].choices[0].message.content)
        self.assertEqual(response[0].choices[0].finish_reason, "stop")

    @unittest.skip("Skip this test for now")
    def test_bad_message(self):
        gpt_manager = GPTBatchManager(model="gpt-4")
        with self.assertRaises(Exception) as context:
            response = gpt_manager.query("this is a wrongly formatted message")

    @unittest.skip("Skip this test for now")
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

    @unittest.skip("Skip this test for now")
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