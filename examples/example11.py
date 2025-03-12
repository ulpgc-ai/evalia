'''
EJEMPLO 1-1: Evaluación de oraciones compuestas
Versión sencilla, sin tratamiento posterior de las respuestas de GPT
'''

import pandas as pd
import os

# Clase para evaluador automático
from evalia.evaluators import BaseEvaluator, OUTPUT_DIR

# Para leer prompts desde ficheros de texto
from evalia.prompt_sources import PromptFromTextFile

# Para configurar el modelo GPT que se va a usar
MODELO_GPT = 'gpt-4'
#MODELO_GPT = 'gpt-3.5-turbo'

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

evaluador = BaseEvaluator(
    evaluator_id = "oraciones_compuestas",
    student_responses = pd.DataFrame(respuestas_estudiantes),
    prompt_source = prompt,
    gpt_manager = MODELO_GPT,
    query_batch_length=20
)

df_result = evaluador.run()

print(evaluador.get_stats())

# Guarda el resultado en un Excel
EXCEL_FILE = os.path.join(OUTPUT_DIR, "example11.xlsx")
df_result.to_excel(EXCEL_FILE)

