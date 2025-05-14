from abc import ABC, abstractmethod
from typing import List
import re
import json

class EvaluatedAnswer(ABC):

    @abstractmethod
    def get_full_response(self) -> str:
        '''get the whole text of the response'''
        pass

    @abstractmethod
    def get_index(self):
        '''find the index of the answer being evaluated'''
        pass

    @abstractmethod
    def get_assessment(self):
        '''extract the assessment/scoring part of this response'''
        pass

    @classmethod
    def extract_evaluated_answers(cls, llm_text_responses: List[str]):
        """
        Extract a list of evaluated answers from the LLM text responses.
        :param llm_text_responses: list of strings with the LLM responses
        :return: list of EvaluatedAnswerExtractor objects
        """
        pass

class EvaluatedAnswerOneLine(EvaluatedAnswer):
    '''
    Single line responses that are easy to handle.
    Example: "17. 2.5"
    '''

    def __init__(self, response: str):
        self.response = response

    def get_full_response(self) -> str:
        return self.response

    def get_index(self):
        # find a number at the beginning of the string followed by a dot
        # skip initial non-digit characters
        id = re.search(r'^\D*(\d+)\.', self.response)
        if id:
            return int(id.group(1))
        else:
            return None
    
    def get_assessment(self):
        return self.response

    @classmethod
    def extract_evaluated_answers(cls, llm_text_responses: List[str]) -> List[EvaluatedAnswer]:
        '''flatten the messages into a list of lines'''
        evaluated_responses = [EvaluatedAnswerOneLine(line)
                               for text in llm_text_responses
                               for line in text.splitlines()
                               ]
        return evaluated_responses
    
class EvaluatedAnswerJSON(EvaluatedAnswer):
    '''
    Each response is a JSON list: [ index, score, comment... ]
    Lists are separated by a separator string
    '''
    separator = '\n'

    def __init__(self, response: str):
        self.raw_response = response
        try:
            self.json_response = json.loads(response)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            self.json_response = []

    def get_full_response(self) -> str:
        return self.json_response

    def get_index(self):
        '''
        return the first element of the list.
        It must be an integer.'''
        try:
            return int(self.json_response[0])
        except ValueError:
            print(f"Error parsing index from response: {self.raw_response}")
            return None
        
    def get_assessment(self):
        try:
            return self.json_response[1:]
        except IndexError:
            print(f"Error parsing assessment from response: {self.raw_response}")
            return None

    @classmethod
    def extract_evaluated_answers(cls, llm_text_responses: List[str]) -> List[EvaluatedAnswer]:
        '''JSON responses are separated by a separator string'''
        evaluated_responses = [EvaluatedAnswerJSON(block)
                               for text in llm_text_responses
                               for block in text.split(EvaluatedAnswerJSON.separator)
                               if block.strip()  # Filter out empty blocks
                               ]
        return evaluated_responses

    
class EvaluatedAnswerMultiLine(EvaluatedAnswer):

    separator = '#RESP#'

    def __init__(self, response: str):
        self.response = response
        self.search_is_done = False
        self.assessment = None

    def get_full_response(self) -> str:
        return self.response

    def get_index(self):
        # find a number at the beginning of the string followed by a dot
        # skip initial non-digit characters
        try:
            id = re.search(r'^\D*(\d+)\.', self.get_assessment())
            if id:
                return int(id.group(1))
            else:
                return None
        except ValueError:
            print(f"Error parsing index from response: {self.response}")
            return None
        
    def get_assessment(self):
        if self.search_is_done:
            return self.assessment
        else:
            # the assessment should be the last ocurrence of a
            # number followed by a dot and any characters until
            # the end of that line
            self.search_is_done = True
            try:
                reversed_lines = self.response.split('\n')[::-1]
                for line in reversed_lines:
                    assessment = re.search(r'^\D*(\d+\..*$)', line)
                    if assessment:
                        self.assessment = assessment.group(1)
                        return self.assessment
                return None
            except:
                return None

    @classmethod
    def extract_evaluated_answers(cls, llm_text_responses: List[str]) -> List[EvaluatedAnswer]:
        '''multi-line responses are separated by a blank line'''
        evaluated_responses = [EvaluatedAnswerMultiLine(block)
                               for text in llm_text_responses
                               for block in text.split(EvaluatedAnswerMultiLine.separator)
                               ]
        return evaluated_responses

# quick test
if __name__ == '__main__':
    raw_response = (
    "1. This is the first line\n"
    "2. This is the second line\n"
    "\n"
    "3. This is the third line\n"
    "This is the last line\n"
    )
    print(raw_response)
    gpt_response = EvaluatedAnswerOneLine(raw_response)
    print(f"id: {gpt_response.get_index()}")
    print(f"assessment: {gpt_response.get_assessment()}")
    print(f"response: {gpt_response.get_full_response()}")
    print('-'*20)
    gpt_response = EvaluatedAnswerMultiLine(raw_response)
    print(f"id: {gpt_response.get_index()}")
    print(f"assessment: {gpt_response.get_assessment()}")
    print(f"response: {gpt_response.get_full_response()}")
