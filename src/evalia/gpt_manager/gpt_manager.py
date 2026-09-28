'''
Clase abstracta que modela una interfaz para dialogar con GPT
'''

from abc import ABC, abstractmethod
from typing import Tuple

class GPTTask(ABC):
  '''Clase abstracta que modela una tarea solicitada a GPT.

  Cada GPTManager debe implementar una clase que herede de GPTTask.
  '''
  pass


class GPTManager(ABC):

  # Si un GPTManager concreto necesita sobrevivir a un reinicio del proceso
  # (p. ej. porque delega en una tarea externa de larga duración, como la
  # Batch API de OpenAI), debe declarar requires_persistence = True.
  # Es un atributo de clase (no depende de __init__) para que el Evaluator
  # pueda consultarlo sin necesidad de conocer la implementación concreta
  # del manager, y sin depender de que el manager llame a super().__init__().
  requires_persistence: bool = False

  def __init__(self):
    self.model = ""
    self.batch_api = False
  
  @abstractmethod
  def initialize(self,model=""):
    '''
    Inicializa el sistema de diálogo con GPT.

    model: una string con el modelo GPT que usará este GPTManager.
    '''
    self.model = model
    pass

  @abstractmethod
  def count_tokens(self,messages):
    '''Cuenta el número de tokens que hay en un diálogo para GPT.'''
    pass

  # TODO: remove or rename this legacy method
  @abstractmethod
  def send_queries(self,query_id,query_list,
                   temperature) -> Tuple[list,dict]:
    '''
    Envía una lista de consultas a GPT y devuelve una lista de respuestas.

    Args:
      query_id: Identificador de la consulta (una string).
      query_list: Lista de consultas (una lista de JSON en formato OpenAI).
      temperature: Número entre 0 y 2. Se pasa directamente a GPT.

    Return:
      Una tupla con dos elementos:
      - una lista con las respuestas de GPT.
      - un diccionario con estadísticas de la consulta:
        - elapsed_time: tiempo que tardó GPT en responder (en segundos).
        - input_tokens: número de tokens de entrada.
        - output_tokens: número de tokens de salida.
    '''
    pass

  @abstractmethod 
  def start_task(self, query_id, query_list, temperature) -> GPTTask:
      '''
      Envía una lista de consultas a GPT y devuelve un objeto respuesta (GPTTask).
      '''
      pass
  
  @abstractmethod
  def cancel_task(self, task: GPTTask):
    '''Cancela una tarea.'''
    pass
  
  @abstractmethod
  def save_task(self, task: GPTTask, filename: str):
    '''Guarda una tarea en un archivo.'''
    pass

  @abstractmethod
  def load_task(self, filename: str) -> GPTTask:
    '''Carga una tarea desde un archivo.'''
    pass
  
  @abstractmethod
  def get_response(self, task: GPTTask,
                   timeout=0.0, retry=0.0) -> Tuple[list,dict]:
    '''Devuelve la respuesta de GPT.
    
    Args:
      Si timeout es 0, se intenta una respuesta inmediata.
      Si timeout>0, se espera un máximo de "timeout" segundos.
      retry: si la respuesta no está disponible, reintenta cada 'retry' segundos,
      hasta que se venza 'timeout'. Si retry=0, no reintenta.
    '''
    pass

