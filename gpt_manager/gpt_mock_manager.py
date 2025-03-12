""" 
Versión mock de GPTManager.
Procesa los mensajes sin dialogar realmente con GPT.
"""
import sys
sys.path.append('.')

from gpt_manager import GPTManager, GPTTask
from typing import Tuple
import re
import copy
import openai.types.chat.chat_completion as chat
from logs import get_logger

logger = get_logger(__name__)

# Mock taken from OpenAI API docs
chat_completion_mock = chat.ChatCompletion(
    id='chatcmpl-9KvPGltIJXhLOOFaTz4QUkOKLvlxD', 
    choices=[
        chat.Choice(
            finish_reason='stop', 
            index=0, 
            logprobs=None, 
            message=chat.ChatCompletionMessage(
                content='0. 1\n1. 0\n2. 0\n3. 0\n4. 0\n5. 1\n6. 0', 
                role='assistant', 
                function_call=None, 
                tool_calls=None)
                )
      ], 
    created=1714774258,
    model='gpt-3.5-turbo-0125',
    object='chat.completion',
    system_fingerprint='fp_a450710239',
    usage=chat.CompletionUsage(
        completion_tokens=34, 
        prompt_tokens=404, 
        total_tokens=438
        )
)

mock_stats = {
    "elapsed_time": 0.1,
    "input_tokens": 1000,  
    "output_tokens": 100
}



class GPTMockManager(GPTManager):
    """Versión mock de GPTManager. Devuelve respuestas ficticias."""

    def __init__(self):
        super().__init__()
        self.model = "mock"

    def initialize(self,model="mock"):
        logger.info(f"-----------------------------------")
        logger.info(f"GPTMockManager started. Model: {self.model}")
        pass

    def count_tokens(self,messages):
        """No implementado"""
        return -1
    
    def send_queries(self, query_id,query_list,
                     temperature=1):
        """
        Simula un diálogo con GPT.
        Devuelve una lista de respuestas falsas, similares a las de GPT.
        """

        def single_dialog(query):
            '''devolvemos los números encontrados en la última frase'''
            last_message = query[-1]['content']
            # detecta los ítems (líneas que empiezan por un número)
            # y les añade una respuesta ficticia (un 1)
            item_numbers = re.findall(r'^(\d+\.)', last_message, re.MULTILINE)
            mock_answer = ' 1\n'
            response = mock_answer.join(item_numbers) + mock_answer

            # modela la respuesta de GPT
            chat_completion = copy.deepcopy(chat_completion_mock)
            chat_completion.choices[0].message.content = response
            return chat_completion

        if not isinstance(query_list,list):
            return single_dialog(query_list), mock_stats
        else:
            return [ single_dialog(q) for q in query_list ], mock_stats

    class GPTMockTask(GPTTask):
        def __init__(self, responses, stats):
            self.responses = responses
            self.stats = stats

    def start_task(self, query_id, query_list, temperature) -> GPTTask:
        responses, stats = self.send_queries(query_id, query_list, temperature)
        return self.GPTMockTask(responses, stats)

    def cancel_task(self, task: GPTTask):
        pass

    def save_task(self, task: GPTTask, filename: str):
        raise NotImplementedError("Esta clase no permite persistir el estado de una tarea.")
    
    def load_task(self, filename: str) -> GPTTask:
        raise NotImplementedError("Esta clase no permite persistir el estado de una tarea.")
    
    def get_response(self, task: GPTMockTask, timeout=0, retry=0) -> Tuple[list, dict]:
        return task.responses, task.stats
