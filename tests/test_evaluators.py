import unittest

import pandas as pd
from evalia.evaluators import Evaluator
from evalia.prompts import PromptFromString
from evalia.gpt_manager import GPTManager, GPTSmartManager, GPTBatchManager

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

class TestEvaluator(unittest.TestCase):

    def new_evaluator(self):
        return Evaluator(
            evaluator_id = "capitales europeas",
            student_responses = pd.DataFrame(respuestas_estudiantes),
            prompt = PromptFromString(PROMPT),
            gpt_manager = 'mock',
            query_batch_length=20
        )
    
    def test_init_gpt_manager(self):
        evaluator1 = Evaluator()
        self.assertEqual(evaluator1.gpt_manager, None)
        evaluator1.batch_api = True
        self.assertEqual(evaluator1.batch_api, True)
        evaluator1.model = 'gpt-4'
        self.assertEqual(evaluator1.model, 'gpt-4')
        self.assertIsInstance(evaluator1.gpt_manager, GPTBatchManager)

        evaluator2 = Evaluator(gpt_manager='gpt-4')
        self.assertIsInstance(evaluator2.gpt_manager, GPTManager)
        self.assertEqual(evaluator2.gpt_manager.model, 'gpt-4')
        self.assertEqual(evaluator2.gpt_manager.batch_api, False)
        evaluator2.model = 'gpt-4o'
        self.assertEqual(evaluator2.model, 'gpt-4o')
        self.assertEqual(evaluator2.gpt_manager.model, 'gpt-4o')    
        self.assertEqual(evaluator2.gpt_manager.batch_api, False)

        evaluator3 = Evaluator(model='gpt-4o')
        self.assertIsInstance(evaluator3.gpt_manager, GPTSmartManager)
        self.assertEqual(evaluator3.gpt_manager.model, 'gpt-4o')
        self.assertEqual(evaluator3.gpt_manager.batch_api, False)
        evaluator3.batch_api = True
        self.assertIsInstance(evaluator3.gpt_manager, GPTBatchManager)


    def test_df_salida(self):
        evaluator = self.new_evaluator()
        df_result = evaluator.run()
        self.assertIsInstance(df_result, pd.DataFrame)
        self.assertEqual(df_result.shape[0], 7)
        self.assertEqual(df_result.shape[1], 4)
        columnas_resultado = df_result.columns
        self.assertTrue("respuesta" in columnas_resultado)
        self.assertTrue("calificación real" in columnas_resultado)
        self.assertTrue("evaluación IA" in columnas_resultado)
        self.assertTrue("respuesta completa IA" in columnas_resultado)

    def test_openai(self):
        evaluator = self.new_evaluator()
        evaluator.gpt_manager = MODELO_GPT
        df_result = evaluator.run()
        self.assertIsInstance(df_result, pd.DataFrame)
        self.assertEqual(df_result.shape[0], 7)
        self.assertEqual(df_result.shape[1], 4)
        columnas_resultado = df_result.columns
        self.assertTrue("respuesta" in columnas_resultado)
        self.assertTrue("calificación real" in columnas_resultado)
        self.assertTrue("evaluación IA" in columnas_resultado)
        self.assertTrue("respuesta completa IA" in columnas_resultado)


if __name__ == '__main__':
    unittest.main()