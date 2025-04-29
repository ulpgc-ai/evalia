import os
import pickle
import unittest
from typing import List
from unittest.mock import patch

from evalia.llm import LanguageModelResponse
from evalia.llm.gpt import Request, RequestQueue, GPTSmartManager
import time
import random


class GPTExcepcionError(Exception):
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
        raise GPTExcepcionError("Ha habido una excepción desde la API de GPT")
    return json
        

class TestRequest(unittest.TestCase):
    def test_request(self):
        request = Request(1234)
        self.assertEqual(request.tokens, 1234)


class TestRequestQueue(unittest.TestCase):

    # Check that request not exceeding the maximum tokens is accepted
    def test_requestqueue_1(self):
        rq = RequestQueue("gpt-4")
        with self.assertRaises(Exception) as context:
            rq.add(Request(150_000_000))
        self.assertTrue("tokens exceeds the maximum" in str(context.exception))

    # Check that the current minute tokens are calculated correctly
    @unittest.skip("Not neccesary")
    def test_requestqueue_2(self):
        rq = RequestQueue("gpt-4")
        rq.add(Request(1000))
        rq.add(Request(2000))
        rq.add(Request(3000))
        self.assertEqual(rq.current_minute_tokens, 6000)


class TestGPTManager(unittest.TestCase):

    @patch('openai.chat.completions.create', side_effect=mock_chat_completion_create)
    def test_basic_interaction(self, mock_chat_completion_create):
        gpt_manager = GPTSmartManager(model="gpt-4")
        for _ in range(10):
            response = gpt_manager.query([{"role": "user", "content": "Hola"}])
        
        # check that the response contains the word "hola"
        self.assertIn("Hola", response[0].choices[0].message.content)
        self.assertEqual(response[0].choices[0].finish_reason, "stop")

    def test_bad_message(self):
        gpt_manager = GPTSmartManager(model="gpt-4")
        with self.assertRaises(Exception):
            gpt_manager.query("this is a wrongly formatted message")

    def test_send_queries(self):
        gpt_manager = GPTSmartManager(model="gpt-4")
        query_list = ["Hola", "Hola, dime hola."]
        llm_responses: List[LanguageModelResponse] = gpt_manager.generate_text(
            query_id="test",
            initial_prompt="",
            query_list=query_list,
            temperature=0.0
        )
        self.assertEqual(len(llm_responses), 2)
        self.assertIsInstance(llm_responses, list)
        for response in llm_responses:
            self.assertIsInstance(response, LanguageModelResponse)

    def test_bad_queries(self):
        gpt_manager = GPTSmartManager(model="gpt-4")
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

    def test_serialize(self):
        gpt_smart_manager = GPTSmartManager(model="gpt-4o-mini")
        gpt_file_name = 'data.pkl'
        self.addCleanup(lambda: os.remove(gpt_file_name) if os.path.exists(gpt_file_name) else None)

        with open(gpt_file_name, 'wb') as file:
            pickle.dump(gpt_smart_manager, file)
        self.assertTrue(os.path.exists(gpt_file_name))

        with open(gpt_file_name, 'rb') as file:
            loaded_manager = pickle.load(file)

        self.assertIsInstance(loaded_manager, GPTSmartManager)
        self.assertEqual(loaded_manager.model, "gpt-4o-mini")
        self.assertIsNotNone(loaded_manager.client)
        self.assertIsNotNone(loaded_manager.encoding)

        
if __name__ == '__main__':
    unittest.main()