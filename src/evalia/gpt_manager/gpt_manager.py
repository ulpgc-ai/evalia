'''
Clase abstracta que modela una interfaz para dialogar con GPT
'''

from abc import ABC, abstractmethod
from typing import Tuple

# PARCHE TEMPORAL: los modelos de razonamiento
# (gpt-5, gpt-5.5, gpt-6...) solo admiten la temperatura por defecto (1) y
# devuelven un error 400 con cualquier otro valor. 
# Para evitar problemas, solo se envía la temperatura a los modelos 
# que se ha comprobado que la aceptan; el resto usa su valor por defecto. 
# gpt-5.1 la acepta mientras no razone, y Evalia
# no activa su razonamiento, al menos de momento.
MODELS_ACCEPTING_TEMPERATURE = ("gpt-3.5", "gpt-4", "gpt-5.1")

def accepts_temperature(model: str) -> bool:
  '''True si el modelo admite una temperatura distinta de la de por defecto.'''
  return model.startswith(MODELS_ACCEPTING_TEMPERATURE)

class GPTTask(ABC):
  '''Clase abstracta que modela una tarea solicitada a GPT.

  Cada GPTManager debe implementar una clase que herede de GPTTask.
  '''
  pass


class GPTManager(ABC):

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

