'''
EJEMPLO 0. Evaluar capitales europeas
'''

import pandas as pd
import os

# Clase para evaluador automático
from evalia.evaluators import Evaluator

# Para leer prompts desde cadenas de texto
from evalia.prompts import PromptFromString

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


# Función fábrica: recibe el id y crea un evaluador nuevo.
# create_or_resume() solo la llama si no hay ningún evaluador guardado con
# ese id; si lo hay, lo recupera tal cual, sin llamarla.
def nuevo_evaluador(id):
    return Evaluator(
        evaluator_id = id,
        student_responses = pd.DataFrame(respuestas_estudiantes),
        prompt = PromptFromString(PROMPT),
        gpt_manager = MODELO_GPT,
        batch_api=True,
        query_batch_length=20 # envía a GPT las respuestas en lotes de 20
    )

# Con la Batch API, el evaluador se guarda en un archivo tras lanzar el lote,
# así que la ejecución se reanuda volviendo a ejecutar este mismo programa.
# Si se desea volver a empezar desde cero:
# - en código, llamar a: Evaluator.discard_saved("ejemplo01-capitales")
# - a mano, borrar el archivo "ejemplo01-capitales.pkl" en el dir. de cachés
evaluador = Evaluator.create_or_resume("ejemplo01-capitales", nuevo_evaluador)


# Ejecuta la evaluación y devuelve un dataframe con el resultado
df_result = evaluador.run()

print("Resultados:")
print(df_result)

# Imprime estadísticas: tokens y tiempo consumido
print(evaluador.get_stats())

# Guarda el resultado en un Excel
EXCEL_FILE = os.path.join(OUTPUT_DIR, "example01.xlsx")
df_result.to_excel(EXCEL_FILE)

