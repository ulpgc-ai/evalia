from dataclasses import asdict
from typing import Type, List, Self, Callable
import pandas as pd
import os
import json
import pickle

from evalia.logs import get_logger
from evalia.prompts import PromptSource
from evalia.llm import LanguageModelManager, LanguageModelResponse
from evalia.api_response import APIResponse, APIResponseOneLine
from evalia.config import cache_dir

# Columns added by the automatic evaluator to the response DataFrame
COLNAME_AI_GRADES = "evaluación "
COLNAME_AI_FULL_EVALUATIONS = "respuesta completa "

logger = get_logger(__name__)
class Evaluator:
    def __init__ (self,
                  evaluator_id: str = '',
                  managers: List[LanguageModelManager] = None,
                  student_responses: pd.DataFrame = None,
                  responses_column: int | str = 0,
                  system_context: PromptSource = None,
                  sample_selector: int | slice | list | Callable = None,
                  query_batch_length: int = 20,
                  api_response_class: Type[APIResponse] = APIResponseOneLine,
                  temperature: float = 0.0,
                  postprocess_one_llm_response: Callable[[str], str] = lambda text: text,
                  persistent: bool = False
                  ):
        """
        Constructor of the Evaluator class.
        :param evaluator_id: the identifier of the evaluator.
        :param managers: the list of managers to use for the evaluation.
        :param student_responses: the DataFrame with the student responses. At least one column with the student responses is required.
        :param responses_column: the index of the column with the student responses in the DataFrame. It can be an integer or a string. By default, it is the first column of the DataFrame.
        :param system_context: context for the LLM. Normally containing instructions for the LLM.
        :param sample_selector: the range of responses to be selected from the DataFrame. It can be a slice (e.g. slice(0,15)), a list of indices (e.g. [1,7,99]),
        an integer N that will be used to take a random sample of N responses, or a Callable object (e.g. a lambda expression). If set to None, all responses are selected.
        :param query_batch_length: the number of responses that will be packed in each query to each LLM. Any integer value greater than 0 is valid.
        :param api_response_class: the modality of the response from the LLM (one line or multiple lines).
        :param temperature: the temperature to use for each LLM. A value of 0.0 means deterministic responses.
        :param postprocess_one_llm_response: a function to postprocess the response from the LLM. It should take a string as input and return a string as output.
        :param persistent: if True, the evaluator will be saved to a file after executing the tasks, and can be loaded in a subsequent execution.
        """
        if persistent:
            loaded_evaluator = self.load_from_file(evaluator_id)
            if loaded_evaluator:
                self.__dict__.update(loaded_evaluator.__dict__)
                return
        self.managers: List[LanguageModelManager] = managers if managers is not None else []

        self.evaluator_id = evaluator_id
        self.student_responses = student_responses
        self.responses_column = responses_column
        self.prompt = system_context
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
        self.api_response_class = api_response_class
        self.temperature = temperature
        self.postprocess_one_llm_response = postprocess_one_llm_response
        self.persistent = persistent

        logger.info(f'"{self.evaluator_id}" created')

    def add_manager(self, manager: LanguageModelManager):
        self.managers.append(manager)
        return self

    @classmethod
    def pickle_filename(cls, evaluator_id):
        return os.path.join(cache_dir(), evaluator_id + ".pkl")

    @classmethod
    def load_from_file(cls, evaluator_id: str) -> Self | None:
        """
        Load a serialized evaluator from a pickle file
        :param evaluator_id: The ID of the evaluator to load
        :return: The loaded evaluator or None if the file does not exist
        """
        filename = cls.pickle_filename(evaluator_id)
        # if filename exists, load the evaluator from the pickle
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                evaluator = pickle.load(f)
            logger.debug(f'"{evaluator_id}" loaded from file "{filename}"')
        else:
            evaluator = None
        return evaluator

    def _persist_evaluator(self):
        filename = self.pickle_filename(self.evaluator_id)
        with open(filename,"wb") as f:
            pickle.dump(self,f)
        logger.debug(f'"{self.evaluator_id}" saved to file "{filename}"')

    def __getstate__(self):
        state = self.__dict__.copy()
        state['postprocess_one_llm_response'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.postprocess_one_llm_response = lambda x: x

    def build_prompt_preamble(self):
        return self.prompt.get_prompt()

    def build_llm_queries(self):
        if isinstance(self.responses_column, int):
            responses = self.sample_answers.iloc[:, self.responses_column]
        elif isinstance(self.responses_column, str):
            responses = self.sample_answers[self.responses_column]
        else:
            raise TypeError("Tipo de columna de respuestas no soportado.")
        return self.partition_batches(responses)

    def partition_batches(self, responses) -> List[str]:
        """ Divides the responses into batches according to `self.query_batch_length` """
        total_responses = len(responses)
        batches = []

        for i in range(0, total_responses, self.query_batch_length):
            batch = responses.iloc[i:i + self.query_batch_length]
            message = ""
            for index, answer in batch.items():
                message += f'[ {index}, {answer} ]\n'
            batches.append(message)

        return batches

    def print_stats(self):
        for manager in self.managers:
            print(f"Stats for {manager.get_llm_name()}:")
            print(f"Input tokens: {sum(response.input_tokens for response in manager.task.responses)}")
            print(f"Output tokens: {sum(response.output_tokens for response in manager.task.responses)}")
            print(f"Elapsed time: {sum(response.elapsed_time for response in manager.task.responses)} seconds")
            print("=============================================")

    def save_llm_responses(self, llm_responses: List[LanguageModelResponse], llm_name: str):
        try:
            filename = os.path.join(cache_dir(), self.evaluator_id + f"_${llm_name}_responses.json")
            json_llm_responses = [asdict(x) for x in llm_responses]
            with open(filename,"w") as f:
                json.dump(json_llm_responses,f,indent=2)
        except Exception as e:
            logger.error(f"Error saving LLM responses: {e}")
            raise

    def evaluate_answers(self) -> pd.DataFrame:
        """
        Evaluates the student responses using each LLM manager.
        For each manager, it builds the prompt and sends the requests to the LLM.
        The responses are processed and the evaluations are extracted.
        :return: a DataFrame which is the same as the sample, but with two additional columns at the end: a column with the score and a column with the evaluation description.
        """
        df = self.sample_answers.copy()
        for manager in self.managers:
            system_context, initial_prompt = self.build_prompt_preamble()
            first_time_executed: bool = manager.task is None
            manager.start_task(query_id = self.evaluator_id,
                                                  initial_prompt = initial_prompt,
                                                  query_list= self.build_llm_queries(),
                                                  system_context = system_context,
                                                  temperature = self.temperature)
            if first_time_executed and self.persistent:
                self._persist_evaluator()

        for manager in self.managers:
            llm_responses = manager.get_response(manager.task, timeout=0, retry=3)
            self.save_llm_responses(llm_responses, manager.get_llm_name())
            self._add_evaluation_columns_to_dataframe(df, llm_responses, manager.get_llm_name())

        logger.info(f'"{self.evaluator_id}" run successfully')
        return df

    def _add_evaluation_columns_to_dataframe(self, df: pd.DataFrame, llm_responses: List[LanguageModelResponse], manager_name: str):
        text_messages = [x.response for x in llm_responses]
        responses = self.api_response_class.extract_responses(self.api_response_class.extract_responses, text_messages)
        get_score = lambda x: self.postprocess_one_llm_response(x.get_assessment())
        indexed_assessments = {x.get_index(): get_score(x) for x in responses}
        indexed_responses = {x.get_index(): x.get_full_response() for x in responses}
        df.loc[:, COLNAME_AI_GRADES + manager_name] = df.index.map(indexed_assessments)
        df.loc[:, COLNAME_AI_FULL_EVALUATIONS + manager_name] = df.index.map(indexed_responses)


def save_excel(df: pd.DataFrame, file_name: str, output_dir: str | None = None):
    """
    Saves a DataFrame to an Excel file with the same name as the evaluated item.
    :param df: the DataFrame to save
    :param file_name: the name of the file (without extension)
    :param output_dir: the directory to save the file. If None, it will be saved in the current working directory.
    :return:
    """
    if output_dir is None:
        base_path = file_name
    else:
        base_path = os.path.join(output_dir, file_name)
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