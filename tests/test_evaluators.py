import unittest

import pandas as pd
from evalia.evaluators import Evaluator
from evalia.prompts import PromptFromString
from evalia.llm.gpt import GPTMockManager

# Usamos un modelo mock para no generar interacción con OpenAI
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


def new_evaluator():
    return Evaluator(
        evaluator_id = "capitales europeas",
        student_responses = pd.DataFrame(respuestas_estudiantes),
        system_context= PromptFromString(PROMPT),
        query_batch_length=20
    ).add_manager(GPTMockManager())


class TestEvaluator(unittest.TestCase):

    def test_init_gpt_manager(self):
        evaluator1 = Evaluator()
        self.assertEqual(evaluator1.managers, [])

    def test_add_llm_manager(self):
        evaluator = Evaluator()
        evaluator.add_manager(GPTMockManager())
        self.assertEqual(len(evaluator.managers), 1)

    def test_df_salida(self):
        evaluator = new_evaluator()
        df_result = evaluator.evaluate_answers()
        self.assertIsInstance(df_result, pd.DataFrame)
        self.assertEqual(df_result.shape[0], 7)
        self.assertEqual(df_result.shape[1], 4)
        columnas_resultado = df_result.columns
        self.assertIn("respuesta", columnas_resultado)
        self.assertIn("calificación real", columnas_resultado)
        self.assertIn("evaluación GPT", columnas_resultado)
        self.assertIn("respuesta completa GPT", columnas_resultado)

    def test_openai(self):
        evaluator = new_evaluator()
        evaluator.gpt_manager = MODELO_GPT
        df_result = evaluator.evaluate_answers()
        self.assertIsInstance(df_result, pd.DataFrame)
        self.assertEqual(df_result.shape[0], 7)
        self.assertEqual(df_result.shape[1], 4)
        columnas_resultado = df_result.columns
        self.assertIn("respuesta", columnas_resultado)
        self.assertIn("calificación real", columnas_resultado)
        self.assertIn("evaluación GPT", columnas_resultado)
        self.assertIn("respuesta completa GPT", columnas_resultado)


if __name__ == '__main__':
    unittest.main()