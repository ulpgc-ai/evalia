import os
import pickle
import unittest

from evalia.llm import LanguageModelResponse
from evalia.llm.gemini import GeminiManager


class TestGeminiManager(unittest.TestCase):

    def test_serialization(self):
        gemini_manager = GeminiManager(model="gemini-2.0-flash")
        fake_response = LanguageModelResponse(response="Hola", elapsed_time=0, input_tokens=1, output_tokens=1)
        gemini_manager.responses = [fake_response]
        gemini_file_name = 'data.pkl'
        self.addCleanup(lambda: os.remove(gemini_file_name) if os.path.exists(gemini_file_name) else None)

        with open(gemini_file_name, 'wb') as file:
            pickle.dump(gemini_manager, file)
        self.assertTrue(os.path.exists(gemini_file_name))

        with open(gemini_file_name, 'rb') as file:
            loaded_manager = pickle.load(file)


        self.assertIsInstance(loaded_manager, GeminiManager)
        self.assertEqual(loaded_manager.model, "gemini-2.0-flash")
        self.assertEqual(len(loaded_manager.responses), 1)
        self.assertEqual(loaded_manager.responses[0].response, "Hola")
        self.assertIsNotNone(loaded_manager.client)
