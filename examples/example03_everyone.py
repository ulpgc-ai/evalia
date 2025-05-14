import os

import pandas as pd

from evalia import Evaluator
from evalia.evaluated_answer import EvaluatedAnswerJSON
from evalia.llm.claude.claude_batch_manager import ClaudeBatchManager
from evalia.llm.gemini.gemini_batch_manager import GeminiBatchManager
from evalia.llm.gpt import GPTBatchManager
from evalia.prompts import PromptFromString

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
No añadas frases adicionales del estilo \"aquí tienes la respuesta\" o \"aquí tienes la evaluación\".
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

evaluator_id = "example03_everyone"
evaluator = Evaluator(
            evaluator_id = evaluator_id,
            student_answers= pd.DataFrame(respuestas_estudiantes),
            system_context= PromptFromString(PROMPT),
            evaluated_answer_extractor=EvaluatedAnswerJSON,
            query_batch_length=20,
            postprocess_one_llm_response= lambda text: text[0],
            managers = [
                GeminiBatchManager(model="gemini-2.0-flash-001"),
                ClaudeBatchManager(model="claude-3-5-haiku-latest"),
                GPTBatchManager(model="gpt-4o-mini"),
            ],
            persistent = True,
)
df_result = evaluator.evaluate_answers()

print("Resultados:")
print(df_result)
evaluator.print_stats()

EXCEL_OUTPUT = os.path.join(OUTPUT_DIR, "example03.xlsx")
df_result.to_excel(EXCEL_OUTPUT)