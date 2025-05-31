'''
EJEMPLO 1-2: Evaluación de oraciones compuestas
Versión con tratamiento posterior de las respuestas del LLM
'''

# añadir la ruta actual para poder importar los módulos
import pandas as pd
import os
import re

# Evaluador automático
from evalia.evaluator import Evaluator
from evalia.llm.gpt import GPTSmartManager

# Para leer prompts desde ficheros de texto
from evalia.prompts import PromptFromTextFile

# Para configurar el modelo GPT que se va a usar
MODELO_GPT = 'gpt-4o-mini'

# Directorio de salida
OUTPUT_DIR = "./output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

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

PROMPT_FILE = "./prompt_oraciones_compuestas.txt"
prompt = PromptFromTextFile(PROMPT_FILE)

evaluator = Evaluator(
    evaluator_id = "oraciones_compuestas",
    answers_dataframe= pd.DataFrame(respuestas_estudiantes),
    prompt= prompt,
    managers = [
        GPTSmartManager(model=MODELO_GPT)
    ],
    query_batch_length=20
)

df_result = evaluator.evaluate_answers()

evaluator.print_stats()

# Guarda el resultado en un Excel
EXCEL_FILE = os.path.join(OUTPUT_DIR, "oraciones_compuestas.xlsx")
df_result.to_excel(EXCEL_FILE)
