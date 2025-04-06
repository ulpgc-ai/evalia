'''
class GPTResponse: abstracción de la respuesta de una API de un modelo de lenguaje
'''

from abc import ABC, abstractmethod
from typing import List
import re
import json

class APIResponse(ABC):

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
    def extract_responses(cls,gpt_messages_list) -> List['APIResponse']:
        '''get a list of OpenAI response messages and transform them
        into a list of GPTResponse objects.
        Every input message should come from an API response, in the
        path response['choices'][0]['message']['content'].
        '''
        pass

class APIResponseOneLine(APIResponse):
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

    def extract_responses(cls,gpt_messages_list) -> List[APIResponse]:
        '''flatten the messages into a list of lines'''
        gpt_responses = [APIResponseOneLine(line)
                         for text in gpt_messages_list
                         for line in text.splitlines()
                         ]
        return gpt_responses	
    
class APIResponseJSON(APIResponse):
    '''
    Each response is a JSON list: [ index, score, comment... ]
    Lists are separated by a separator string
    '''
    separator = '\n'

    def __init__(self, response: str):
        self.raw_response = response
        try:
            self.json_response = json.loads(response)
        except:
            self.json_response = None

    def get_full_response(self) -> str:
        return self.json_response

    def get_index(self):
        '''
        return the first element of the list.
        It must be an integer.'''
        try:
            return int(self.json_response[0])
        except:
            return None
        
    def get_assessment(self):
        try:
            return self.json_response[1:]
        except:
            return None
            
    def extract_responses(cls,gpt_messages_list) -> List[APIResponse]:
        '''JSON responses are separated by a separator string'''
        gpt_responses = [APIResponseJSON(block)
                         for text in gpt_messages_list
                         for block in text.split(cls.separator)
                         ]
        return gpt_responses

    
class APIResponseMultiLine(APIResponse):

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
        except:
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
            
    def extract_responses(cls,gpt_messages_list) -> List[APIResponse]:
        '''multi-line responses are separated by a blank line'''
        gpt_responses = [APIResponseMultiLine(block)
                         for text in gpt_messages_list
                         for block in text.split(cls.separator)
                         ]
        return gpt_responses

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
    gpt_response = APIResponseOneLine(raw_response)
    print(f"id: {gpt_response.get_index()}")
    print(f"assessment: {gpt_response.get_assessment()}")
    print(f"response: {gpt_response.get_full_response()}")
    print('-'*20)
    gpt_response = APIResponseMultiLine(raw_response)
    print(f"id: {gpt_response.get_index()}")
    print(f"assessment: {gpt_response.get_assessment()}")
    print(f"response: {gpt_response.get_full_response()}")
