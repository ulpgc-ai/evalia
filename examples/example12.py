'''
EJEMPLO 1-2: Evaluación de oraciones compuestas
Versión con tratamiento posterior de las respuestas de GPT
'''

# añadir la ruta actual para poder importar los módulos
import pandas as pd
import os
import re

# Evaluador automático
from evalia.evaluators import Evaluator, OUTPUT_DIR

# Para leer prompts desde ficheros de texto
from evalia.prompt_sources import PromptFromTextFile

# Para configurar el modelo GPT que se va a usar
MODELO_GPT = 'gpt-4o-mini'

respuestas_estudiantes = {
    "respuesta": [
        "El deporte femenino es el que está de moda",
        "El deporte es femenino y está de moda",
        "El deporte es femenino, pero poco conocido",
        "El deporte a sido lo mas conocido del mundo",
        "Aunque no es muy conocido, el deporte femenino está de moda"
        ],
    "calificación": [1, 1, 0, 0, 1]
}

PROMPT_FILE = "./examples/prompt_oraciones_compuestas.txt"
prompt = PromptFromTextFile(PROMPT_FILE)

class EvaluadorOracionesCompuestas(Evaluator):

    # constructor que pasa todos los argumentos a la clase base
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

    # GPT devuelve por cada respuesta un texto elaborado.
    # El método postprocess_one_gpt_response() extrae una calificación
    # numérica a partir del texto de GPT. 
    def postprocess_one_gpt_response(self,text):
        """
        Calcula si una respuesta de GPT está OK o no:
        - al menos 2 verbos
        - al menos 1 conector
        Ejemplos de textos:
        20. 2 verbos (piensa, está), 1 conector (que).
        21. 1 verbo (estaba), 0 conectores.
        """
        try:
            text=str.lower(text)
            str_verbos = re.findall(r'(\d+)\s+(?:verbo)', text)
            nverbos = 0 if not str_verbos else int(str_verbos[0])
            str_conectores = re.findall(r'(\d+)\s+(?:conector)', text)
            nconectores = 0 if not str_conectores else int(str_conectores[0])
            return int(nverbos>=2 and nconectores>=1)
        except:
            return None
        

evaluador = EvaluadorOracionesCompuestas(
    evaluator_id = "oraciones_compuestas",
    student_responses = pd.DataFrame(respuestas_estudiantes),
    prompt_source = prompt,
    gpt_manager = MODELO_GPT,
    query_batch_length=20
)

df_result = evaluador.run()

print(evaluador.get_stats())

# Guarda el resultado en un Excel
EXCEL_FILE = os.path.join(OUTPUT_DIR, "oraciones_compuestas.xlsx")
df_result.to_excel(EXCEL_FILE)
