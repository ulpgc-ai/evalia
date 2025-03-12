""" 
Habla con GPT a través de la API de OpenAI. Se le entrega una lista
de consultas, cada una de ellas en el formato JSON de OpenAI. 
El sistema devuelve las respuestas de GPT.

Métodos:

- initialize(): pone a punto el sistema
- count_tokens(): no implementado
- send_queries(): envía un lote de peticiones a GPT
"""

import openai
import os
from gpt_manager import GPTManager

class GPTBasicManager(GPTManager):

    def __init__(self,model=""):
        self.model = model

    def initialize(self,model=""):
        """Inicializa el sistema de GPT"""
        openai.api_key = os.getenv('OPENAI_API_KEY')
        if model != "":
            self.model = model
    
    def count_tokens(self,messages):
        """No implementado"""
        return -1
    
    def send_queries(self, query_id,query_list,
                     temperature=1):
        """
        Dialoga con GPT y obtiene una respuesta.

        - query_id: no se utiliza.
        - query_list: una lista con peticiones para GPT.
        - temperature: se pasa directamente a la API de OpenAI.

        Retorna la lista de respuestas que produce GPT.
        """

        def single_dialog(query):
            response = openai.chat.completions.create(
                model = self.model,
                temperature = temperature,
                messages = query
            )
            return response

        mock_stats = {
            "elapsed_time": 0.1,
            "input_tokens": 1000,  
            "output_tokens": 100
        }

        if not isinstance(query_list,list):
            return single_dialog(query_list), mock_stats
        else:
            return [ single_dialog(q) for q in query_list ], mock_stats

