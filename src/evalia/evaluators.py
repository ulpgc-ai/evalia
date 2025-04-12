"""
class Evaluator
---------------
Clase para procesar un ítem típico y proporcionar una evaluación.
Esta clase sirve para la mayoría de los prompts. Tiene un par de métodos
para tratar las respuestas antes y después de ser procesados por GPT:

- build_prompt_preamble(): produce un JSON con la conversación inicial que se
  repite en todas las interacciones con GPT.
- build_gpt_queries(): convierte la muestra en una colección de textos para el GPT.
- evaluate_answers(): transforma las respuestas de cada LLM en un DataFrame listo para explotar.
- get_stats(): devuelve estadísticas de la interacción con GPT.

Estos métodos se orquestan en el siguiente método plantilla:

- run(): ejecuta todos los pasos anteriores. Devuelve las respuestas, un DataFrame y
  el tiempo consumido

- preprocess_one_answer(text) -> devuelve un texto listo para GPT
- postprocess_one_gpt_response(text) -> nuevo texto listo para el DataFrame de respuesta

Por defecto, estos métodos no transforman nada. 
Para ítems más complicados o con particularidades, se pueden ir especializando
estos métodos.

"""

from typing import Type, List
import pandas as pd
import os
import re
import json
import pickle

from evalia.logs import get_logger
from evalia.prompts import PromptSource
from evalia.llm import LanguageModelManager, LanguageModelResponse
from evalia.api_response import APIResponse, APIResponseOneLine
from evalia.config import cache_dir

# Columnas que añade el evaluador automático al DataFrame de respuestas
COLNAME_AI_GRADES = "evaluación "
COLNAME_AI_FULL_EVALUATIONS = "respuesta completa "

# Logging
logger = get_logger(__name__)
class Evaluator:
    '''Clase base para la mayoría de los ítems'''

    def __init__ (self,
                  evaluator_id='',
                  student_responses: pd.DataFrame = None,
                  responses_column: int | str = 0,
                  prompt: PromptSource = None,
                  sample_selector = None,
                  query_batch_length=20,
                  gpt_response_class: Type[APIResponse] = APIResponseOneLine,
                  temperature: float = 0.0
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
        - query_batch_length: número de respuestas que se empaquetarán en cada consulta a GPT.
          Vale cualquier valor entero de 1 en adelante.
        - gpt_response_class: modalidad de respuesta de GPT (una línea o varias líneas).
        '''
        super().__init__()

        self.managers: List[LanguageModelManager] = []

        # Set other attributes
        self.evaluator_id = evaluator_id
        self.student_responses = student_responses
        self.responses_column = responses_column
        self.prompt = prompt
        self.sample_selector = sample_selector
        if self.sample_selector is None:
            self.sample_answers = self.student_responses
        elif isinstance(self.sample_selector, int):
            self.sample_answers = self.student_responses.sample(n=self.sample_selector, random_state=42)
        elif isinstance(self.sample_selector, slice):
            self.sample_answers = self.student_responses[self.sample_selector]
        elif isinstance(self.sample_selector, list):
            intersection = self.student_responses.index.intersection(self.sample_selector)
            self.sample_answers = self.student_responses.iloc[intersection]
        elif callable(self.sample_selector):
            self.sample_answers = self.sample_selector(self.student_responses)
        else:
            raise TypeError("Tipo de selector no soportado.")
        self.query_batch_length = query_batch_length
        self.gpt_response_class = gpt_response_class
        self.temperature = temperature

        # Reset execution state variables
        self.gpt_responses = None
        self.stats = None
        self.result = None

        logger.info(f'"{self.evaluator_id}" created')

    def add_manager(self, manager: LanguageModelManager):
        self.managers.append(manager)
        return self

    # Load a serialized evaluator from a pickle file
    # so you can continue the evaluation process.
    # The object was serialized after executing send_gpt_queries()
    @classmethod
    def pickle_filename(cls, evaluator_id):
        return os.path.join(cache_dir(), evaluator_id + ".pkl")

    @classmethod
    def load_from_file(cls, evaluator_id):
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
        self.__dict__.update(state)
        self.gpt_manager = state['_gpt_manager']
        self.model = self.gpt_manager.model

    # PERSISTENCE DECORATOR
    @classmethod
    def persistent(cls,func):
        '''Decorator that loads the evaluator from a pickle file if it exists'''
        def f(item_id):
            eva = Evaluator.load_from_file(item_id)
            if eva is None:
                return func(item_id)
            else:
                return eva
        return f

    ### --- end of persistence section

    def build_prompt_preamble(self):
        return self.prompt.get_prompt()

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

    def build_llm_queries(self):
        if isinstance(self.responses_column, int):
            responses = self.sample_answers.iloc[:, self.responses_column]
        elif isinstance(self.responses_column, str):
            responses = self.sample_answers[self.responses_column]
        else:
            raise TypeError("Tipo de columna de respuestas no soportado.")
        return [self.build_prompt_preamble() + batch for batch in self.partition_batches(responses)]

    def partition_batches(self, responses):
        """ Divides the responses into batches according to `self.query_batch_length` """
        total_responses = len(responses)
        batches = []

        for i in range(0, total_responses, self.query_batch_length):
            batch = responses.iloc[i:i + self.query_batch_length]
            batch_messages = [
                {
                    'role': 'user',
                    'content': self.gpt_input_text(index, self.preprocess_one_answer(answer))
                }
                for index, answer in batch.items()
            ]
            batches.append(batch_messages)

        return batches

    def get_stats(self):
        return self.stats

    def postprocess_one_gpt_response(self,text):
        '''(override this method as needed)
        transform a GPT response to an answer into a usable text
        '''
        return text

    def save_llm_responses(self, llm_responses: List[LanguageModelResponse], llm_name: str):
        try:
            filename = os.path.join(cache_dir(), self.evaluator_id + f"_${llm_name}_responses.json")
            dictlist = [x.dict() for x in llm_responses]
            with open(filename,"w") as f:
                json.dump(dictlist,f,indent=2)
        except:
            pass

    def evaluate_answers(self):
        '''
        Recupera la respuesta del LLM, la procesa y
        extrae las evaluaciones correspondientes a cada respuesta
        de la muestra evaluada.
        Devuelve un DataFrame que es el mismo de la muestra,
        añadiendo dos columnas al final:
        - una columna con la calificación
        - una columna con la descripción de la evaluación
        '''
        df = self.sample_answers.copy()
        for manager in self.managers:
            llm_responses = manager.generate_text(query_id = self.evaluator_id,
                                                  query_list= self.build_llm_queries(),
                                                  system_context = self.build_prompt_preamble(),
                                                  temperature = self.temperature)
            self.save_llm_responses(llm_responses, manager.get_llm_name())
            text_messages = [x.response for x in llm_responses]

            # me obliga a usar la clase dos veces: como objeto y también como argumento
            extractor = self.gpt_response_class
            gpt_responses = extractor.extract_responses(text_messages)

            # a partir de gpt_lines, obtener listas indexadas de respuestas y evaluaciones
            get_score = lambda x: self.postprocess_one_gpt_response(x.get_assessment())
            indexed_assessments = { x.get_index():get_score(x) for x in gpt_responses }
            indexed_responses = { x.get_index():x.get_full_response() for x in gpt_responses }
            # añadir columnas al dataframe, vinculadas por el índice
            # NOTA: puede haber índices faltantes por errores en la respuesta de GPT
            # por eso hay que usar .loc e .index.map
            df.loc[:,COLNAME_AI_GRADES + manager.get_llm_name()] = df.index.map(indexed_assessments)
            df.loc[:,COLNAME_AI_FULL_EVALUATIONS + manager.get_llm_name()] = df.index.map(indexed_responses)

        self.result = df
        logger.info(f'"{self.evaluator_id}" run successfully')
        return df

    def rerun_gpt_responses (self,gpt_responses_file):
        '''
        Vuelve a procesar las respuestas de GPT recibidas en una 
        anterior ejecución
        '''
        with open(gpt_responses_file,"r") as f:
            gpt_responses = json.load(f)
        self.gpt_responses = gpt_responses
        df_result = self.evaluate_answers()
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