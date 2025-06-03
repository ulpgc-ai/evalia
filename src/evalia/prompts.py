""" 
Clases que modelan la fuente de un prompt inicial.
Tenemos estas clases:

- PromptFromString: el prompt viene de una cadena de texto.
- PromptFromTextFile: el prompt viene de un fichero de texto plano.
- PromptFromJSON: el prompt viene de un fichero JSON (compatible con OpenAI).
- PromptFromTemplate: el prompt viene de un fichero de texto plano con 
  parámetros (suplidos como JSON).
"""

from abc import ABC, abstractmethod
import ast

class PromptSource(ABC):
    """
    Base class for any source that produces an initial prompt.
    """

    @abstractmethod
    def get_prompt(self):
        """
        Gets a prompt suitable for input into a language model

        :return: a tuple (system_context, initial_prompt)
        """
        '''devuelve una cadena con un prompt apto para ingresar en GPT'''
        return None

class PromptFromString(PromptSource):
    """
    The prompt source is a string in memory.
    """

    def __init__(self, prompt_text: str):
        self.prompt_text = prompt_text

    def get_prompt(self):
        system_context = (
            "Eres un asistente colaborador. "
            "En tus respuestas cíñete estrictamente a las instrucciones dadas. "
            "No aportes explicaciones ni justificaciones adicionales."
        )
        initial_prompt = self.prompt_text
        return system_context, initial_prompt

class PromptFromTextFile(PromptSource):
    """
    The prompt source is a plain text file.
    """

    def __init__(self, prompt_filename: str):
        self.prompt_filename = prompt_filename

    def get_prompt(self):
        with open(self.prompt_filename, 'r', encoding="utf-8") as prompt_file:
            prompt_text = prompt_file.read()
        prompt_preamble = PromptFromString(prompt_text).get_prompt()
        return prompt_preamble

class PromptFromStringCoT(PromptSource):
    """
    The prompt source is a string in memory. Specially designed for Chain of Thought (CoT).
    """

    def __init__(self,prompt_text):
        self.prompt_text = prompt_text

    def get_prompt(self):
        system_context = (
            "Actúa como un evaluador de pruebas académicas. "
            "En tus evaluaciones, desarrolla los razonamientos que llevan a tus conclusiones. "
            "Nunca alcances una conclusión sin haber explicitado el razonamiento previo. "
            "Cumple estrictamente las instrucciones sobre los formatos de tu respuesta."
        )
        initial_prompt = self.prompt_text
        return system_context, initial_prompt


class PromptFromTextFileCoT(PromptSource):
    """
    The same as PromptFromTextFile, but with a special prompt for Chain of Thought (CoT).
    """
    def __init__(self,prompt_filename):
        self.prompt_filename = prompt_filename
    
    def get_prompt(self):
        with open(self.prompt_filename, 'r') as prompt_file:
            prompt_text = prompt_file.read()
        prompt_preamble = PromptFromStringCoT(prompt_text).get_prompt()
        return prompt_preamble

class PromptFromJSON(PromptSource):
    '''El prompt está escrito en el formato JSON de OpenAI'''

    def __init__(self,prompt_filename):
        self.prompt_filename = prompt_filename

    def get_prompt(self):
        with open(self.prompt_filename, 'r', 
                encoding='UTF-8', errors='ignore') as prompt_file:
            main_prompt = prompt_file.read()
            prompt_preamble = ast.literal_eval(main_prompt)
        return prompt_preamble

class PromptFromTemplate(PromptSource):
    """
    The prompt comes from a plain text file with parameters. The parameters are filled in from a separate JSON file.
    """
    def __init__(self,prompt_filename,args_filename):
        self.prompt_filename = prompt_filename
        self.args_filename = args_filename

    def get_prompt(self):
        with open(self.prompt_filename, 'r',encoding='utf-8') as prompt_file:
            template_text = prompt_file.read()
        with open(self.args_filename, 'r',encoding='utf-8') as args_file:
            arguments_raw = args_file.read()
        arguments = ast.literal_eval(arguments_raw)
        prompt_text = template_text
        for key,value in arguments.items():
            prompt_text = prompt_text.replace(key,value)      
        prompt_preamble = PromptFromString(prompt_text).get_prompt()
        return prompt_preamble

