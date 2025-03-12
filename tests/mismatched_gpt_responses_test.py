import pandas as pd
from io import StringIO
import copy

import sys
sys.path.append('.')

from evaluators import BaseEvaluator

# Estructura similar a un ítem del dataset de la ACCUEE
mock_dataframe = '''
"Respuesta 8"  "Calificación 8.1"
 0      Texto-0                 0
 1      Texto-1                 0
 3      Texto-2                 0
 7      Texto-3                 1
 8      Texto-4                 0
12      Texto-5                 0
63      Texto-6                 1
'''

mock_response_ok = """0. 1lsakdfjkalsdf
1. 0laskfdjlasf
3. 1laksjdflasfdssdf
7. 1lksdjfowqer
8. 0oiqjweofif asdfsad
12. 1lasjdfas
63. 0lkjqwr qwoeiroqwer
"""

mock_response_missing = """0. 1lsakdfjkalsdf
1. 0laskfdjlasf
3. 1laksjdflasfdssdf
8. 0oiqjweofif asdfsad
63. 0lkjqwr qwoeiroqwer
"""

mock_response_extra = """0. 1lsakdfjkalsdf
1. 0laskfdjlasf
3. 1laksjdflasfdssdf

7. 1lksdjfowqer
8. 0oiqjweofif asdfsad
9. 0ESTE NO EXISTE EN EL DF

63. 0lkjqwr qwoeiroqwer
"""

# Mock taken from OpenAI API docs
chat_completion_mock = {
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "gpt-3.5-turbo-0613",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "\n\nHello there, how may I assist you today?",
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 9,
    "completion_tokens": 12,
    "total_tokens": 21
  }
}

class TestItem(BaseEvaluator):

    def read_sample_answers(self):
        self.sample_answers = pd.read_csv(
            StringIO(mock_dataframe), 
            delim_whitespace=True, quoting=2,
            index_col=0, dtype={0:int})
        return self.sample_answers
    
    def receive_gpt_responses(self):
        chat_completion = copy.deepcopy(chat_completion_mock)
        chat_completion['choices'][0]['message']['content'] = mock_response_extra
        return [chat_completion]

# NOTA: el nombre del ítem es irrelevante, aunque tiene que existir    
myitem = TestItem("4ESO-8.1")

myitem.process_gpt_responses()
pass