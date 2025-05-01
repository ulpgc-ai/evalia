'''
EJEMPLO 3. Evaluar capitales europeas.
Partiendo del ejemplo 2 (repuesta JSON con comentarios),
usamos el método Evaluator.postprocess_one_gpt_response() para
extraer la calificación numérica y depositarla en la columna
"Calificación GPT".

Este ejemplo utiliza la API por lotes de OpenAI (Batch API).
'''

import pandas as pd
import os

from evalia.api_response import APIResponseJSON
# Clase para evaluador automático
from evalia.evaluators import Evaluator
from evalia.llm.gpt import GPTBatchManager

# Para leer prompts desde cadenas de texto
from evalia.prompts import PromptFromString

# Para trabajar con respuestas en formato JSON

# Para configurar el modelo GPT que se va a usar
MODELO_GPT = 'gpt-4o-mini'

# Directorio de salida
OUTPUT_DIR = "./output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

PROMPT = '''
Eres un evaluador de geografía y te han pedido que evalúes si los estudiantes 
conocen las capitales de los países europeos.
Al estudiante se le da la siguiente instrucción:
"escribe una lista de cinco capitales europeas".
A continuación te pasaré una lista de respuestas de estudiantes. 
Cada respuesta tiene un número de índice y la lista de ciudades.
Tienes que calificar cada respuesta de la siguiente forma:
1 = hay al menos cinco nombres y todos son capitales europeas.
0 = cualquier otro caso.
No importan las faltas de ortografía, por ejemplo considera correctas "Berlin" y "Verlin".
Tu calificación debe venir en este formato: [ <número de respuesta>, <calificación>, "<justificación de la calificación>" ]
'''

respuestas_estudiantes = {
    "respuesta": [
        "Madrid, París, Berlín, Roma, Lisboa",
        "Madriz, Paris, Verlin, Rroma, Lisbona", # con faltas de ortografía
        "París, Londres",
        "Pekín, Tokio, París, Roma, Copenhague, Londres",
        "esto es una respuesta inválida",
        "Madrid, París, Berlín, Roma, Lisboa, Londres",
        "Barcelona, Londres, Rotterdam, Salzburgo"
        ],
    "calificación real": [1, 1, 0, 0, 0, 1, 0]
}

evaluator_id = "capitales europeas"
evaluator = Evaluator(
            evaluator_id = evaluator_id,
            student_responses = pd.DataFrame(respuestas_estudiantes),
            system_context= PromptFromString(PROMPT),
            api_response_class=APIResponseJSON,
            query_batch_length=20,
            postprocess_one_llm_response= lambda text: text[0],
            managers = [
                GPTBatchManager(model=MODELO_GPT),
            ],
            persistent = True,
)
df_result = evaluator.evaluate_answers()

print("Resultados:")
print(df_result)
print(evaluator.get_stats())

EXCEL_OUTPUT = os.path.join(OUTPUT_DIR, "example03.xlsx")
df_result.to_excel(EXCEL_OUTPUT)

