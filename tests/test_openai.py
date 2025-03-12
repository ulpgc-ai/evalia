import unittest

class TestOpenAI(unittest.TestCase):

    def test_ChatCompletion_class(self):
        '''check that ChatCompletion internal class is defined'''
        try:
            from openai.types.chat.chat_completion import ChatCompletion
        except ImportError:
            self.fail("OpenAI ChatCompletion class not defined. "
                      "Check the OpenAI Python library!!!")

    def test_ChatCompletion_from_JSON(self):
        '''check that a JSON can be converted to ChatCompletion'''
        from openai.types.chat.chat_completion import ChatCompletion
        import json
        # A typical JSON string response from OpenAI
        response_string = '''
        {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-4o-mini",
        "system_fingerprint": "fp_44709d6fcb",
        "choices": [{
            "index": 0,
            "message": {
            "role": "assistant",
            "content": "Hello there, how may I assist you today?"
            },
            "logprobs": null,
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 9,
            "completion_tokens": 12,
            "total_tokens": 21,
            "completion_tokens_details": {
            "reasoning_tokens": 0
            }
        }
        }
        '''
        json_response = json.loads(response_string)
        try:
            response = ChatCompletion.model_validate(json_response)
        except Exception as e:
            self.fail(
                f"Error converting JSON to ChatCompletion: {e}. "
                "Check the OpenAI Python library!!!")
        self.assertIsInstance(response, ChatCompletion)
        self.assertEqual(response.id, "chatcmpl-123")
        self.assertEqual(response.choices[0].message.role, "assistant")
        self.assertEqual(response.usage.prompt_tokens, 9)

