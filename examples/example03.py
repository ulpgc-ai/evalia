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

# Clase para evaluador automático
from evalia.evaluators import Evaluator

# Para leer prompts desde cadenas de texto
from evalia.prompts import PromptFromString

# Para trabajar con respuestas en formato JSON
from evalia.gpt_responses import GPTResponseJSON

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


# Definimos una nueva clase, ya que vamos a crear un método nuevo
class Evaluador(Evaluator):
    def __init__(self, id):
        super().__init__(
            evaluator_id = id,
            student_responses = pd.DataFrame(respuestas_estudiantes),
            prompt = PromptFromString(PROMPT),
            gpt_response_class=GPTResponseJSON,
            gpt_manager = MODELO_GPT,
            batch_api=True,
            query_batch_length=20
        )

    def postprocess_one_gpt_response(self, text):
        '''Extrae la calificación numérica de la respuesta de GPT.
        Lo que devuelve esta función es lo que irá
        a la columna "Calificación GPT".
        La respuesta de GPT es una lista: [ <calificación>, <'justificación'> ]
        (ojo, el número de índice se suprime)
        Es importante usar manejadores de excepciones para defenderse de
        posibles errores de GPT.'''
        try:
            score = text[0]
            return score
        except:
            return None

# El decorador "persistent" guarda el estado del objeto en un archivo
# y permite reanudar la ejecución en otro momento.
# La reanudación se hace volviendo a ejecutar este mismo programa.
@Evaluator.persistent
def evaluador_persistente(id_evaluador):
    return Evaluador(id_evaluador)

evaluador = evaluador_persistente("capitales europeas")
df_result = evaluador.run()

print("Resultados:")
print(df_result)
print(evaluador.get_stats())

EXCEL_OUTPUT = os.path.join(OUTPUT_DIR, "example03.xlsx")
df_result.to_excel(EXCEL_OUTPUT)

