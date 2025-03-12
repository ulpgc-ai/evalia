'''
EJEMPLO 1. Evaluar capitales europeas
'''

import pandas as pd
import os

# Clase para evaluador automático
from evaluators import BaseEvaluator, OUTPUT_DIR

# Para leer prompts desde cadenas de texto
from prompt_sources import PromptFromString

# Para configurar el modelo GPT que se va a usar
MODELO_GPT = 'gpt-4o-mini'

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
Tu calificación debe venir en este formato: <número de respuesta>. <calificación>
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


evaluador = BaseEvaluator(
    evaluator_id = "capitales europeas",
    student_responses = pd.DataFrame(respuestas_estudiantes),
    prompt_source = PromptFromString(PROMPT),
    gpt_manager = MODELO_GPT,
    query_batch_length=20 # envía a GPT las respuestas en lotes de 20
)

# Ejecuta la evaluación y devuelve un dataframe con el resultado
df_result = evaluador.run()

print("Resultados:")
print(df_result)

# Imprime estadísticas: tokens y tiempo consumido
print(evaluador.get_stats())

# Guarda el resultado en un Excel
EXCEL_FILE = os.path.join(OUTPUT_DIR, "example01.xlsx")
df_result.to_excel(EXCEL_FILE)

