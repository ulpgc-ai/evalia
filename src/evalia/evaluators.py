"""
class AbstractEvaluator
-----------------------
Clase abstracta para la evaluación de un ítem.

Métodos abstractos:

- build_prompt_preamble(): produce un JSON con la conversación inicial que se 
  repite en todas las interacciones con GPT.
- read_sample_answers(): lee una muestra de respuestas desde una fuente de datos (un DataFrame).
- build_gpt_queries(): convierte la muestra en una colección de textos para el GPT.
- receive_gpt_responses(): recibe las respuestas de GPT en el JSON de OpenAI.
- process_gpt_responses(): transforma las respuestas de GPT en un DataFrame listo para explotar.
- get_stats(): devuelve estadísticas de la interacción con GPT.

Estos métodos se orquestan en el siguiente método plantilla:

- run(): ejecuta todos los pasos anteriores. Devuelve las respuestas, un DataFrame y
  el tiempo consumido

class Evaluator
---------------
Clase concreta con la implementación básica para procesar un ítem típico y
proporcionar una evaluación.
Esta clase sirve para la mayoría de los prompts. Tiene un par de métodos
para tratar las respuestas antes y después de ser procesados por GPT:

- preprocess_one_answer(text) -> devuelve un texto listo para GPT
- postprocess_one_gpt_response(text) -> nuevo texto listo para el DataFrame de respuesta

Por defecto, estos métodos no transforman nada. 
Para ítems más complicados o con particularidades, se pueden ir especializando
estos métodos.

"""

from abc import ABC, abstractmethod
from typing import Callable, Type, TypeVar
import inspect
import time
import pandas as pd
import os
import re
import json
import pickle
import warnings
from openai.types.chat.chat_completion import ChatCompletion

from evalia.logs import get_logger
from evalia.prompts import PromptSource
from evalia.gpt_manager import GPTManager, GPTTask, gpt_factory
from evalia.gpt_responses import GPTResponse, GPTResponseOneLine
from evalia.config import cache_dir

DEFAULT_TEMPERATURE = 0.0

# Columnas que añade el evaluador automático al DataFrame de respuestas
COLNAME_AI_GRADES = "evaluación IA"
COLNAME_AI_FULL_EVALUATIONS = "respuesta completa IA"

# Tiempo de espera para recibir respuesta de GPT (segundos)
GPT_TIMEOUT = 0

# Tiempo de espera para reintentar obtener respuesta de GPT (segundos)
GPT_RETRY = 1

# Tipo de la clase concreta (Evaluator o subclase) en create_or_resume()
EvaluatorT = TypeVar("EvaluatorT", bound="Evaluator")


class AbstractEvaluator(ABC):
    '''
    Clase abstracta para la evaluación de un ítem
    '''
   
    def __init__(self):
      self.temperature = DEFAULT_TEMPERATURE
   
    @abstractmethod
    def build_prompt_preamble(self):
        '''return a dict list with a chat.completion conversation'''
        pass
   
    @abstractmethod
    def read_sample_answers(self):
        '''return a DataFrame with a sample of student answers'''
        pass

    @abstractmethod
    def build_gpt_queries(self):
        '''
        Transform the sample answers into a list of requests to be sent to GPT.
        Return the GPT-enabled list.
        '''
        pass

    @abstractmethod
    def send_gpt_queries(self):
        '''send the queries to GPT'''
        pass

    # TODO: rename this method as receive_gpt_responses()
    @abstractmethod
    def receive_gpt_responses_new(self):
        '''receive responses from GPT. Return a list of responses.'''
        pass

    # TODO: rename this method as dialog_with_gpt()
    @abstractmethod
    def receive_gpt_responses(self):
        '''(LEGACY) return a list of GPT responses'''

    @abstractmethod
    def get_stats(self):
        '''return a dict with statistics'''
        pass

    @abstractmethod
    def process_gpt_responses(self):
        '''
        Post-process the gpt response after it is received.
        Return a DataFrame with added columns with the gpt
        responses for each student answer.
        '''
        pass

    def run(self):
        '''template method that runs the whole evaluation process'''
        self.build_gpt_queries()
        self.send_gpt_queries()
        self.receive_gpt_responses_new()
        df_result = self.process_gpt_responses()
        return df_result 


# Logging
logger = get_logger(__name__)

# --- clase base para casi cualquier ítem

class Evaluator(AbstractEvaluator):
    '''Clase base para la mayoría de los ítems'''

    def __init__ (self, 
                  evaluator_id='',
                  student_responses: pd.DataFrame | None = None,
                  responses_column: int | str = 0,
                  prompt: PromptSource | None = None,
                  sample_selector = None,
                  gpt_manager: str | GPTManager | None = None,
                  model: str | None = None,
                  batch_api: bool = False,
                  query_batch_length=20,
                  gpt_response_class: Type[GPTResponse] = GPTResponseOneLine,
                  autosave_gpt_responses: bool = True
                  ):
        '''
        Args:

        - evaluator_id: identificador del evaluador automático. Normalmente será una string.
        - student_responses: un DataFrame con al menos una columna con las respuestas de los estudiantes.
        - responses_column: índice de la columna de respuestas en el DataFrame.
          Puede ser un entero o una string. Por defecto es la primera columna del DataFrame.
        - sample_selector: el rango de respuestas que se seleccionarán del DataFrame. Puede ser
          un slice (ej. slice(0,15)), una lista de índices (ej. [1,7,99]), un entero N que servirá para
          tomar una muestra aleatoria de N respuestas, o un objeto Callable
          (ej. una expresión lambda). Si se deja a None, se seleccionan todas las respuestas.
        - gpt_manager: se puede aportar un objeto GPTManager ya inicializado. Si no se aporta, se
          puede especificar el modelo de GPT que se usará
        - model: el modelo de GPT que se usará, ej. 'gpt-4o' (argumento alternativo a gpt_manager).
        - batch_api: si es True, se usará la API de lotes (Batch API) de OpenAI.
        - query_batch_length: número de respuestas que se empaquetarán en cada consulta a GPT. 
          Vale cualquier valor entero de 1 en adelante.
        - gpt_response_class: modalidad de respuesta de GPT (una línea o varias líneas).
        - autosave_gpt_responses: si es True (por defecto), al recibir las respuestas de GPT
          guarda una copia local en un JSON, en el directorio EVALIA_CACHE_DIR.
          Sirve para diagnosticar problemas o para volver
          a procesar las respuestas con rerun_gpt_responses() sin invocar de nuevo a GPT.
        '''
        super().__init__()

        # Set GPT manager
        self._gpt_manager = None
        self._model = None
        self._batch_api = batch_api
        if gpt_manager is not None and model is not None:
            raise ValueError(
            "No se pueden especificar al mismo tiempo gpt_manager y model."
            )
        if model is not None:
            self.gpt_manager = model
        elif gpt_manager is not None:
            self.gpt_manager = gpt_manager
        else:
            self._gpt_manager = None
        
        # Set other attributes
        self.evaluator_id = evaluator_id
        self.student_responses = student_responses
        self.responses_column = responses_column
        self.prompt = prompt
        self.sample_selector = sample_selector
        self.query_batch_length = query_batch_length
        self.gpt_response_class = gpt_response_class
        self.autosave_gpt_responses = autosave_gpt_responses

        # Reset execution state variables
        self.reset()

        logger.info(f'"{self.evaluator_id}" created')

    # --- getters and setters for the GPTManager attributes

    def _change_gpt_manager(self):
        if self._model is not None:
            self._gpt_manager = gpt_factory.get_manager(
                model=self._model, batch_api=self._batch_api
                )
        else:
            self._gpt_manager = None

    @property
    def gpt_manager(self) -> GPTManager:
        return self._gpt_manager
    
    @gpt_manager.setter
    def gpt_manager(self,gpt_manager):
        if isinstance(gpt_manager,str):
            self._model = gpt_manager
            self._change_gpt_manager()
        else:
            manager:GPTManager = gpt_manager
            self._gpt_manager = manager
            self._model = manager.model
            self._batch_api = manager.batch_api

    @property
    def model(self) -> str:
        return self._model
    
    @model.setter
    def model(self,model:str):
        try:
            old_model = self._model
        except:
            old_model = None
        if model != old_model:
            self._model = model
            self._change_gpt_manager()
            
    @property
    def batch_api(self) -> bool:
        return self._batch_api
    
    @batch_api.setter
    def batch_api(self,batch_api:bool):
        try:
            old_batch_api = self._batch_api
        except:
            old_batch_api = None
        if batch_api != old_batch_api:
            self._batch_api = batch_api
            self._change_gpt_manager()

    # --- end of getters and setters

    def reset(self):
        self.sample_answers = None
        self.queries = None
        self.task: GPTTask = None
        self.gpt_responses = None
        self.stats = None
        self.result = None

    # Load a serialized evaluator from a pickle file
    # so you can continue the evaluation process.
    # The object was serialized after executing send_gpt_queries()

    @classmethod
    def pickle_filename(cls, evaluator_id):
        return os.path.join(cache_dir(), evaluator_id + ".pkl")
    
    @classmethod
    def load_from_file(cls, evaluator_id) -> AbstractEvaluator:
        '''Load a serialized evaluator from a pickle file'''
        filename = cls.pickle_filename(evaluator_id)
        # if filename exists, load the evaluator from the pickle
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                evaluator = pickle.load(f)
            logger.debug(f'"{evaluator_id}" loaded from file "{filename}"')
        else:
            evaluator = None
        return evaluator
    
    # Internal function called in send_gpt_queries()
    def _persist_evaluator(self):
        filename = self.pickle_filename(self.evaluator_id)
        with open(filename,"wb") as f:
            pickle.dump(self,f)
        logger.debug(f'"{self.evaluator_id}" saved to file "{filename}"')
        

    # Redefine serialization. GPTSmartManager cannot be serialized
    # because it uses TikToken (a non-serializable module).
    def __getstate__(self):
        gpt_manager = self._gpt_manager
        self._gpt_manager = gpt_manager.model
        self.model = gpt_manager.model
        self.batch_api = gpt_manager.batch_api
        state = self.__dict__.copy()
        self._gpt_manager = gpt_manager
        return state
    
    def __setstate__(self,state):
        # Evaluadores guardados con versiones anteriores a autosave_gpt_responses
        state.setdefault('autosave_gpt_responses', True)
        self.__dict__.update(state)
        self.gpt_manager = state['_gpt_manager']
        self.model = self.gpt_manager.model

    # CONSTRUCTOR DE OBJETO PERSISTENTE: 
    # recupera el evaluador guardado en disco o, si no hay ninguno, crea uno nuevo. 
    @classmethod
    def create_or_resume(
        cls: type[EvaluatorT],
        evaluator_id: str,
        factory: Callable[[str], EvaluatorT] | None = None
    ) -> EvaluatorT:
        '''Recupera el evaluador guardado con este evaluator_id o, si no hay
        ninguno, crea uno nuevo llamando a factory(evaluator_id).

        Args:

        - evaluator_id: identificador del evaluador. También da nombre al
          fichero donde se guarda, así que debe ser único para cada evaluación.
        - factory: recibe el id y devuelve un evaluador nuevo (una función,
          una lambda, un functools.partial o una clase). Solo se llama si no
          hay nada guardado. Si se omite, se usa la propia clase, que en ese
          caso debe tener un constructor propio al que le baste con el id.

        Al reanudar se recupera el ESTADO guardado (prompt, respuestas, modelo,
        tarea en curso...), pero los MÉTODOS son los del código actual: un
        cambio en postprocess_one_gpt_response() se aplica, pero un cambio en
        el prompt o en los datos no. Para empezar de cero, discard_saved().

        El evaluador solo se guarda si su manager lo necesita (p. ej. la
        Batch API). Con un manager síncrono no se guarda nada.

        Ejemplos:

            evaluador = Evaluador.create_or_resume("capitales")

            evaluador = Evaluator.create_or_resume(
                "capitales",
                lambda id: Evaluator(id, prompt=..., student_responses=...))
        '''
        filename = cls.pickle_filename(evaluator_id)
        try:
            loaded = cls.load_from_file(evaluator_id)
        except Exception as e:
            raise RuntimeError(
                f'No se pudo recuperar "{evaluator_id}" de "{filename}": {e!r}. '
                f'Si ya no lo necesitas: Evaluator.discard_saved("{evaluator_id}")'
                ) from e
        if loaded is not None:
            if not isinstance(loaded, cls):
                raise TypeError(
                    f'"{evaluator_id}" está guardado como {type(loaded).__name__}, '
                    f'no como {cls.__name__}. '
                    f'¿Dos evaluadores distintos usan el mismo id?'
                    )
            print(f'"{evaluator_id}": reanudado desde "{filename}"')
            return loaded

        if factory is not None:
            evaluator = factory(evaluator_id)
        else:
            cls._check_id_only_constructor()
            evaluator = cls(evaluator_id)
        if not isinstance(evaluator, cls):
            raise TypeError(
                f'La factory devolvió un {type(evaluator).__name__}, '
                f'no un {cls.__name__}'
                )
        if evaluator.evaluator_id != evaluator_id:
            raise ValueError(
                f'El evaluador creado tiene evaluator_id="{evaluator.evaluator_id}" '
                f'en lugar de "{evaluator_id}": no se podría reanudar'
                )
        return evaluator

    @classmethod
    def _check_id_only_constructor(cls):
        '''Comprueba que la clase tiene constructor propio y le basta con el id.'''
        if cls.__init__ is not Evaluator.__init__:
            try:
                inspect.signature(cls).bind("id")
                return
            except TypeError:
                pass
        raise TypeError(
            f'{cls.__name__} necesita algo más que el id para crearse. '
            f'Pasa una factory: '
            f'{cls.__name__}.create_or_resume(id, lambda id: {cls.__name__}(id, ...))'
            )

    @classmethod
    def discard_saved(cls, evaluator_id: str) -> bool:
        '''Borra el evaluador guardado con este id, para que la próxima vez
        create_or_resume() cree uno nuevo. Devuelve True si había uno.

        No cancela su tarea si sigue en curso en el proveedor (p. ej. un
        batch de OpenAI): seguiría ejecutándose y facturándose.'''
        filename = cls.pickle_filename(evaluator_id)
        if os.path.exists(filename):
            os.remove(filename)
            logger.debug(f'"{evaluator_id}" discarded: file "{filename}" removed')
            return True
        return False

    # DECORADOR OBSOLETO: se mantiene para no romper los programas que lo
    # usan. La función decorada, func(id) -> evaluador, es una factory.
    @classmethod
    def persistent(cls, func):
        '''Obsoleto: usa create_or_resume(id, factory).'''
        warnings.warn(
            "Evaluator.persistent está obsoleto: usa Evaluator.create_or_resume(id, factory)",
            DeprecationWarning, stacklevel=2)
        return lambda item_id: Evaluator.create_or_resume(item_id, func)

    ### --- end of persistence section

    # --- copia de las respuestas de GPT
    # Independiente de la persistencia: discard_saved() no borra la copia
    # y delete_gpt_responses() no borra el evaluador guardado.

    @classmethod
    def gpt_responses_filename(cls, evaluator_id):
        '''Fichero con la copia de las respuestas de GPT, en el directorio de caché.'''
        return os.path.join(cache_dir(), evaluator_id + "_gpt_responses.json")

    def save_gpt_responses(self, filename=None):
        '''Guarda una copia de las respuestas de GPT en un JSON. Por defecto,
        en gpt_responses_filename(evaluator_id), que es donde se guardan
        automáticamente si autosave_gpt_responses es True.

        Se puede volver a procesar con rerun_gpt_responses(), sin invocar
        de nuevo a GPT.'''
        if self.gpt_responses is None:
            raise ValueError(
                f'"{self.evaluator_id}": todavía no hay respuestas de GPT que guardar'
                )
        if filename is None:
            filename = self.gpt_responses_filename(self.evaluator_id)
        dictlist = [ x.model_dump() for x in self.gpt_responses ]
        with open(filename,"w") as f:
            json.dump(dictlist,f,indent=2)
        logger.debug(f'"{self.evaluator_id}" GPT responses saved to file "{filename}"')

    # Internal function called when GPT responses are received.
    # Solo se llama cuando ya han llegado TODAS las respuestas: si la petición
    # falla a medias, no se guarda nada (todo o nada). Si no se puede guardar,
    # solo se avisa en el log, para no interrumpir una evaluación ya pagada.
    def _autosave_gpt_responses(self):
        if not self.autosave_gpt_responses:
            return
        try:
            self.save_gpt_responses()
        except Exception as e:
            logger.warning(f'"{self.evaluator_id}" GPT responses not saved: {e!r}')

    @classmethod
    def delete_gpt_responses(cls, evaluator_id: str) -> bool:
        '''Borra la copia de las respuestas de GPT guardada con este id.
        Devuelve True si había una.

        No borra el evaluador guardado: para eso, discard_saved().'''
        filename = cls.gpt_responses_filename(evaluator_id)
        if os.path.exists(filename):
            os.remove(filename)
            logger.debug(f'"{evaluator_id}" GPT responses deleted: file "{filename}" removed')
            return True
        return False

    # --- end of GPT responses section

    def build_prompt_preamble(self):
        assert self.prompt is not None
        return self.prompt.get_prompt()
        
    def read_sample_answers(self):
        '''Lee una muestra de respuestas de los estudiantes'''
        assert self.student_responses is not None
        if self.sample_answers is None:
            if self.sample_selector is None:
                self.sample_answers = self.student_responses
            elif isinstance(self.sample_selector,int):
                self.sample_answers = self.student_responses.sample(n=self.sample_selector,random_state=42)
            elif isinstance(self.sample_selector,slice):
                self.sample_answers = self.student_responses[self.sample_selector]
            elif isinstance(self.sample_selector,list):
                intersection = self.student_responses.index.intersection(self.sample_selector)
                self.sample_answers = self.student_responses.iloc[intersection]
            elif callable(self.sample_selector):
                self.sample_answers = self.sample_selector(self.student_responses)
            else:
                raise TypeError("Tipo de selector no soportado.")
        return self.sample_answers
    
    def preprocess_one_answer(self,text):
        '''(override this method as needed)
        transform one student answer from the dataframe
        representation into a text to be delivered to GPT
        '''
        return text

    def gpt_input_text(self,index,student_answer):
        '''format one answer as GPT text: a JSON list'''
        processed_answer = self.preprocess_one_answer(student_answer)
        json_student_answer = json.dumps(processed_answer)
        json_list = f'[ {index}, {json_student_answer} ]'
        return json_list
            
    def build_gpt_queries(self):
        if self.queries is not None:
            return self.queries
        prompt_preamble = self.build_prompt_preamble()
        dataset_answers = self.read_sample_answers()

        if isinstance(self.responses_column,int):
            getcol = lambda x: x.iloc[self.responses_column]
        elif isinstance(self.responses_column,str):
            getcol = lambda x: x.loc[self.responses_column]
        else:
            raise TypeError("Tipo de columna de respuestas no soportado.")

        if self.query_batch_length > 1:
            # partition the dataset into batches of self.query_batch_length consecutive answers
            # each batch will be appended to query_list
            # the last batch may be smaller than self.query_batch_length
            query_list = []
            for i in range(0,len(dataset_answers),self.query_batch_length):
                batch_slice = dataset_answers.iloc[i:i+self.query_batch_length,:]
                gpt_input_list = [ 
                    self.gpt_input_text(index,getcol(row))
                    for index,row in batch_slice.iterrows() 
                    ]
                batch_message = {
                    'role': 'user',
                    'content': '\n'.join(gpt_input_list)
                }
                query_slice = prompt_preamble + [batch_message]
                query_list.append(query_slice)
        else:

            def one_query(index,text):
                return prompt_preamble + [{
                    'role': 'user',
                    'content': self.gpt_input_text(index,text)
                }]
                     
            query_list = [ one_query(index,getcol(row)) 
                           for index,row in dataset_answers.iterrows() 
                         ]

        self.queries = query_list
        return self.queries

    def send_gpt_queries(self):
        if self.task is None:
            self.build_gpt_queries()
            self.task = self.gpt_manager.start_task(
                self.evaluator_id,
                self.queries,
                self.temperature)
            # Solo persiste si el manager lo necesita (p. ej. Batch API):
            # es una tarea externa de larga duración que puede sobrevivir
            # a un reinicio del proceso. El manager síncrono no lo necesita.
            if self.gpt_manager.requires_persistence:
                self._persist_evaluator()
        return self.task

    def receive_gpt_responses_new(self):
        if self.gpt_responses is None:
            self.send_gpt_queries()
            self.gpt_responses, self.stats = self.gpt_manager.get_response(
                self.task,
                GPT_TIMEOUT, GPT_RETRY
                )
            self._autosave_gpt_responses()
        return self.gpt_responses

    def receive_gpt_responses(self):
        if self.gpt_responses is None:
            self.read_sample_answers()
            queries = self.build_gpt_queries()
            self.gpt_responses, self.stats = self.gpt_manager.send_queries(
                query_id=self.evaluator_id,
                query_list=queries,
                temperature=self.temperature
                )
            self._autosave_gpt_responses()
        return self.gpt_responses
        
    def get_stats(self):
        return self.stats
    
    def postprocess_one_gpt_response(self,text):
        '''(override this method as needed)
        transform a GPT response to an answer into a usable text
        '''
        return text

    def process_gpt_responses(self):
        '''
        Recupera la respuesta de GPT, la procesa y
        extrae las evaluaciones correspondientes a cada respuesta
        de la muestra evaluada.
        Devuelve un DataFrame que es el mismo de la muestra,
        añadiendo dos columnas al final: 
        - una columna con la calificación 
        - una columna con la descripción de la evaluación
        '''
        df = self.read_sample_answers().copy()
        gpt_responses = self.receive_gpt_responses()
        gpt_text_messages = [ x.choices[0].message.content
                              for x in gpt_responses ]

        # me obliga a usar la clase dos veces: como objeto y también como argumento
        extractor = self.gpt_response_class
        gpt_responses = extractor.extract_responses(extractor,gpt_text_messages)

        # a partir de gpt_lines, obtener listas indexadas de respuestas y evaluaciones
        get_score = lambda x: self.postprocess_one_gpt_response(x.get_assessment())
        indexed_assessments = { x.get_index():get_score(x) for x in gpt_responses }
        indexed_responses = { x.get_index():x.get_full_response() for x in gpt_responses }
        # añadir columnas al dataframe, vinculadas por el índice
        # NOTA: puede haber índices faltantes por errores en la respuesta de GPT
        # por eso hay que usar .loc e .index.map
        df.loc[:,COLNAME_AI_GRADES] = df.index.map(indexed_assessments)
        df.loc[:,COLNAME_AI_FULL_EVALUATIONS] = df.index.map(indexed_responses)

        self.result = df
        logger.info(f'"{self.evaluator_id}" run successfully')
        return df
    
    def rerun_gpt_responses (self,gpt_responses_file):
        '''
        Vuelve a procesar las respuestas de GPT recibidas en una
        anterior ejecución, sin invocar de nuevo a GPT.
        '''
        with open(gpt_responses_file,"r") as f:
            gpt_responses = json.load(f)
        # El JSON guarda diccionarios: se reconstruyen los objetos de OpenAI
        self.gpt_responses = [ ChatCompletion.model_validate(x) for x in gpt_responses ]
        df_result = self.process_gpt_responses()
        return df_result


def extract_indicators(gpt_text_answer):
    '''
    Extrae indicadores de una respuesta de GPT
    Los indicadores vendrán como parejas indicador_alfanumérico.valor_numérico,
    por ejemplo: "a.1 b.2 c.3 d.4 e.5 f.6".
    Ojo: normalizamos indicadores a minúsculas.
    Retorna un dict con los indicadores y sus valores, ej. {'a':1,'b':2,...}
    '''
    try:
        matches = re.findall(r'([a-z0-9]+\.\d+)',str.lower(gpt_text_answer))
        pairs = [ m.split('.') for m in matches ]
        indicators = { k:int(v) for k,v in pairs }
    except:
        indicators = {}
    return indicators


def save_excel(df, ITEM, output_dir = None):
    '''
    Guarda un DataFrame (resultado) en un Excel
    con el mismo nombre que el ítem evaluado.
    El directorio de salida por defecto es el CWD.
    '''
    if output_dir is None:
        base_path = ITEM
    else:
        base_path = os.path.join(output_dir,ITEM)
    extension = ".xlsx"
    counter = 0
    while True:
      if counter == 0:
        pathname = base_path + extension
      else:
        pathname = f"{base_path}-{counter}{extension}"
      if not os.path.exists(pathname):
        df.to_excel(pathname)
        break
      counter += 1

def simple_report(item : Evaluator,
                  gpt_response, df_result, elapsed_time):
    print(f"elapsed time (secs): {elapsed_time}")
    print("tokens spent: ")
    print(gpt_response["usage"])

    print("Answer: ")
    gpt_text_answer : str = gpt_response["choices"][0]["message"]["content"]
    print(gpt_text_answer)

    save_excel(df_result,item.evaluator_id)

def save_result(eva : Evaluator):
    '''
    Imprime las estadísticas de la última evaluación
    y genera un archivo Excel con los resultados
    '''
    stats = eva.get_stats()
    print(stats)
    save_excel(eva.result,eva.evaluator_id)