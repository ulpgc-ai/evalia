from dataclasses import asdict
from typing import List, Self, Callable
import pandas as pd
import os
import json
import pickle

from evalia.llm.evaluated_answer import EvaluatedAnswer
from evalia.logs import get_logger
from evalia.prompts import PromptSource
from evalia.llm import LanguageModelManager, LanguageModelResponse, EvaluatedJustifiedAnswer
from evalia.config import cache_dir

# Columns added by the automatic evaluator to the response DataFrame
COLNAME_AI_GRADES = "evaluación "
COLNAME_AI_FULL_EVALUATIONS = "respuesta completa "

logger = get_logger(__name__)
class Evaluator:
    def __init__ (self,
                  evaluator_id: str = '',
                  managers: List[LanguageModelManager] = None,
                  answers_dataframe: pd.DataFrame = None,
                  responses_column: int | str = 0,
                  prompt: PromptSource = None,
                  sample_selector: int | slice | list | Callable = None,
                  query_batch_length: int = 20,
                  persistent: bool = False
                  ):
        """
        Constructor of the Evaluator class.
        :param evaluator_id: the identifier of the evaluator.
        :param managers: the list of managers to use for the evaluation.
        :param answers_dataframe: the DataFrame with the student responses. At least one column with the student responses is required.
        :param responses_column: the index of the column with the student responses in the DataFrame. It can be an integer or a string. By default, it is the first column of the DataFrame.
        :param prompt: context for the LLM. Normally containing instructions for the LLM.
        :param sample_selector: the range of responses to be selected from the DataFrame. It can be a slice (e.g. slice(0,15)), a list of indices (e.g. [1,7,99]), an integer N that will be used to take a random sample of N responses, or a Callable object (e.g. a lambda expression). If set to None, all responses are selected.
        :param query_batch_length: the number of responses that will be packed in each query to each LLM. Any integer value greater than 0 is valid.
        :param persistent: if True, the evaluator will be saved to a file after executing the tasks, and can be loaded in a subsequent execution.
        """
        if persistent:
            loaded_evaluator = self.load_from_file(evaluator_id)
            if loaded_evaluator:
                self.__dict__.update(loaded_evaluator.__dict__)
                return
        self.managers: List[LanguageModelManager] = managers if managers is not None else []

        self.evaluator_id = evaluator_id
        self.answers_dataframe = answers_dataframe
        self.responses_column = responses_column
        self.prompt = prompt
        self.sample_selector = sample_selector

        if isinstance(self.sample_selector, int):
            self.answers_dataframe = self.answers_dataframe.sample(n=self.sample_selector, random_state=42)
        elif isinstance(self.sample_selector, slice):
            self.answers_dataframe = self.answers_dataframe[self.sample_selector]
        elif isinstance(self.sample_selector, list):
            intersection = self.answers_dataframe.index.intersection(self.sample_selector)
            self.answers_dataframe = self.answers_dataframe.iloc[intersection]
        elif callable(self.sample_selector):
            self.answers_dataframe = self.sample_selector(self.answers_dataframe)
        elif self.sample_selector is not None:
            raise TypeError("Tipo de selector no soportado.")

        if isinstance(self.responses_column, int):
            self.student_answers = self.answers_dataframe.iloc[:, self.responses_column]
        elif isinstance(self.responses_column, str):
            self.student_answers = self.answers_dataframe[self.responses_column]
        else:
            raise TypeError("Tipo de columna de respuestas no soportado.")

        self.query_batch_length = query_batch_length
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
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def partition_batches(self, responses) -> List[str]:
        """ Divides the responses into batches according to `self.query_batch_length` """
        total_responses = len(responses)
        batches = []

        for i in range(0, total_responses, self.query_batch_length):
            batch = responses.iloc[i:i + self.query_batch_length]
            message = ""
            for index, answer in batch.items():
                message += f'[ index: {index}, answer: "{answer}" ]\n'
            batches.append(message)

        return batches

    def print_stats(self):
        for manager in self.managers:
            print(f"Stats for {manager.get_llm_name()}:")
            print(f"Input tokens: {sum(response.input_tokens for response in manager.responses)}")
            print(f"Output tokens: {sum(response.output_tokens for response in manager.responses)}")
            print(f"Elapsed time: {sum(response.elapsed_time for response in manager.responses)} seconds")
            print("=============================================")

    def _save_llm_responses(self, llm_responses: List[LanguageModelResponse], llm_name: str):
        try:
            filename = os.path.join(cache_dir(), self.evaluator_id + f"_{llm_name}_responses.json")
            json_llm_responses = [vars(x) for x in llm_responses]
            with open(filename, "w") as f:
                json.dump(json_llm_responses, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving LLM responses: {e}")

    def evaluate_answers(self) -> pd.DataFrame:
        """
        Evaluates the student responses using each LLM manager.
        For each manager, it builds the prompt and sends the requests to the LLM.
        The responses are processed and the evaluations are extracted.
        :return: a DataFrame which is the same as the sample, but with two additional columns at the end: a column with the score and a column with the evaluation description.
        """
        if self.managers is None or len(self.managers) == 0:
            raise ValueError("No managers have been added to the evaluator. Please add at least one manager before running the evaluation.")
        df = self.answers_dataframe.copy()
        for manager in self.managers:
            system_context, initial_prompt = self.prompt.get_prompt()
            manager.start_task(query_id = self.evaluator_id,
                              initial_prompt = initial_prompt,
                              query_list= self.partition_batches(self.student_answers),
                              system_context = system_context)
            if self.persistent:
                self._persist_evaluator()

        for manager in self.managers:
            llm_responses = manager.get_response(manager.task, timeout=0, retry=3)
            self._save_llm_responses(llm_responses, manager.get_llm_name())
            evaluated_answers = self._convert_to_evaluated_answers(llm_responses, manager.structured_output_class)
            self._add_evaluation_columns_to_dataframe(df, evaluated_answers, manager.get_llm_name())

        logger.info(f'"{self.evaluator_id}" run successfully')
        return df

    import pandas as pd

    @staticmethod
    def _add_evaluation_columns_to_dataframe(df: pd.DataFrame, evaluated_answers: List[EvaluatedAnswer], manager_name: str):
        indexed_scores = {x.index: x.score for x in evaluated_answers}
        df[COLNAME_AI_GRADES + manager_name] = df.index.map(indexed_scores)
        if all(isinstance(x, EvaluatedJustifiedAnswer) for x in evaluated_answers):
            indexed_comments = {x.index: x.comment for x in evaluated_answers}
            df[COLNAME_AI_FULL_EVALUATIONS + manager_name] = df.index.map(indexed_comments)

    @staticmethod
    def _convert_to_evaluated_answers(llm_responses: List[LanguageModelResponse], structured_output_class):
        evaluated_answers: List[EvaluatedAnswer] = []
        text_messages = [x.response for x in llm_responses]
        for text_message in text_messages:
            parsed_evaluated_answers = structured_output_class(**json.loads(text_message))
            evaluated_answers.extend(parsed_evaluated_answers.results)
        return evaluated_answers



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