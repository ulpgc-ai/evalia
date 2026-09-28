import io
import os
import unittest
from contextlib import redirect_stdout
from functools import partial
from unittest.mock import Mock, patch

import pandas as pd
from evalia.evaluators import Evaluator
from evalia.prompts import PromptFromString
from evalia.gpt_manager import GPTManager, GPTSmartManager, GPTBatchManager, GPTMockManager
from evalia.gpt_manager.gpt_smart_manager import RequestQueue

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
    
    # GPTSmartManager descubre sus límites reales en el constructor (una
    # petición mínima a OpenAI): se mockea para que este test de "cableado"
    # (qué clase de gestor se instancia según los parámetros) siga sin tocar
    # la red, tal y como indica el comentario de MODELO_GPT más arriba.
    @patch.object(RequestQueue, '_discover_limits', return_value=(500, 10_000))
    def test_init_gpt_manager(self, _mock_discover_limits):
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


# Definida a nivel de módulo (no dentro de un test) a propósito: pickle
# necesita poder importar la clase por su nombre cualificado para
# reconstruirla, y una clase local a una función no es importable.
class SubEvaluadorParaTest(Evaluator):
    def postprocess_one_gpt_response(self, text):
        return "custom:" + str(text)


class EvaluadorSoloId(Evaluator):
    '''Constructor de un solo argumento: create_or_resume() no necesita factory.'''
    inits = 0

    def __init__(self, id):
        EvaluadorSoloId.inits += 1
        super().__init__(
            evaluator_id=id,
            student_responses=pd.DataFrame(respuestas_estudiantes),
            prompt=PromptFromString(PROMPT),
            gpt_manager='mock',
        )


class EvaluadorCurso(Evaluator):
    '''Constructor que necesita algo más que el id: create_or_resume() pide una factory.'''
    def __init__(self, id, curso):
        super().__init__(
            evaluator_id=id,
            student_responses=pd.DataFrame(respuestas_estudiantes),
            prompt=PromptFromString(PROMPT),
            gpt_manager='mock',
        )
        self.curso = curso


class TestPersistence(unittest.TestCase):
    """create_or_resume() y la persistencia condicional de send_gpt_queries()."""

    def setUp(self):
        self._pickle_ids = []

    def tearDown(self):
        for evaluator_id in self._pickle_ids:
            try:
                os.remove(Evaluator.pickle_filename(evaluator_id))
            except OSError:
                pass

    def _fresh_id(self, name):
        '''Id de evaluador único por test, y ya registrado para limpieza.'''
        self._pickle_ids.append(name)
        try:
            os.remove(Evaluator.pickle_filename(name))
        except OSError:
            pass
        return name

    def _new_evaluator(self, evaluator_id, **kwargs):
        return Evaluator(
            evaluator_id=evaluator_id,
            student_responses=pd.DataFrame(respuestas_estudiantes),
            prompt=PromptFromString(PROMPT),
            gpt_manager='mock',
            **kwargs,
        )

    def _factory_spy(self, **kwargs):
        '''Factory que crea un Evaluator con el mock manager y registra sus llamadas.'''
        return Mock(side_effect=lambda id: self._new_evaluator(id, **kwargs))

    def test_create_or_resume_creates_fresh_when_no_pickle(self):
        evaluator_id = self._fresh_id("test_create_or_resume_fresh")
        factory = self._factory_spy(query_batch_length=20)
        evaluator = Evaluator.create_or_resume(evaluator_id, factory)
        factory.assert_called_once_with(evaluator_id)
        self.assertEqual(evaluator.evaluator_id, evaluator_id)
        self.assertIsInstance(evaluator.gpt_manager, GPTMockManager)
        self.assertEqual(evaluator.query_batch_length, 20)

    def test_create_or_resume_recovers_from_pickle_without_calling_factory(self):
        evaluator_id = self._fresh_id("test_create_or_resume_resume")
        self._new_evaluator(evaluator_id, query_batch_length=5)._persist_evaluator()

        # La factory crearía un evaluador distinto (query_batch_length=999),
        # pero ya hay uno guardado con este evaluator_id: se recupera tal cual,
        # sin llamar a la factory, y se avisa por consola.
        factory = self._factory_spy(query_batch_length=999)
        with redirect_stdout(io.StringIO()) as out:
            resumed = Evaluator.create_or_resume(evaluator_id, factory)
        factory.assert_not_called()
        self.assertEqual(resumed.query_batch_length, 5)
        self.assertIn("reanudado", out.getvalue())

    def test_create_or_resume_accepts_partial_as_factory(self):
        evaluator_id = self._fresh_id("test_create_or_resume_partial")
        evaluator = Evaluator.create_or_resume(evaluator_id, partial(
            Evaluator,
            student_responses=pd.DataFrame(respuestas_estudiantes),
            prompt=PromptFromString(PROMPT),
            gpt_manager='mock',
        ))
        self.assertEqual(evaluator.evaluator_id, evaluator_id)

    def test_create_or_resume_without_factory_uses_id_only_constructor(self):
        evaluator_id = self._fresh_id("test_create_or_resume_id_only")
        EvaluadorSoloId.inits = 0
        created = EvaluadorSoloId.create_or_resume(evaluator_id)
        self.assertIsInstance(created, EvaluadorSoloId)
        self.assertEqual(EvaluadorSoloId.inits, 1)

        created._persist_evaluator()
        with redirect_stdout(io.StringIO()):
            resumed = EvaluadorSoloId.create_or_resume(evaluator_id)
        self.assertIsInstance(resumed, EvaluadorSoloId)
        # Al reanudar no se vuelve a ejecutar el constructor
        self.assertEqual(EvaluadorSoloId.inits, 1)

    def test_create_or_resume_with_factory_for_complex_constructor(self):
        evaluator_id = self._fresh_id("test_create_or_resume_complex")
        evaluator = EvaluadorCurso.create_or_resume(
            evaluator_id, lambda id: EvaluadorCurso(id, curso="4ESO"))
        self.assertIsInstance(evaluator, EvaluadorCurso)
        self.assertEqual(evaluator.curso, "4ESO")

    def test_create_or_resume_requires_factory_if_id_is_not_enough(self):
        # Ni Evaluator ni una subclase cuyo constructor pida algo más que el
        # id se pueden crear sin factory: error que sugiere pasar una.
        for cls in (Evaluator, EvaluadorCurso):
            with self.subTest(cls=cls.__name__):
                evaluator_id = self._fresh_id(f"test_create_or_resume_no_factory_{cls.__name__}")
                with self.assertRaisesRegex(TypeError, "factory"):
                    cls.create_or_resume(evaluator_id)

    def test_create_or_resume_rejects_evaluator_with_other_id(self):
        # Se guardaría con otro nombre de fichero y nunca se podría reanudar
        evaluator_id = self._fresh_id("test_create_or_resume_other_id")
        with self.assertRaises(ValueError):
            Evaluator.create_or_resume(
                evaluator_id, lambda id: self._new_evaluator("otro id"))

    def test_create_or_resume_rejects_factory_of_other_class(self):
        evaluator_id = self._fresh_id("test_create_or_resume_factory_other_class")
        with self.assertRaises(TypeError):
            EvaluadorSoloId.create_or_resume(evaluator_id, self._factory_spy())

    def test_create_or_resume_rejects_saved_evaluator_of_other_class(self):
        # Dos evaluadores distintos con el mismo id: el guardado es un
        # Evaluator, pero se pide un EvaluadorSoloId.
        evaluator_id = self._fresh_id("test_create_or_resume_saved_other_class")
        self._new_evaluator(evaluator_id)._persist_evaluator()
        with self.assertRaisesRegex(TypeError, "mismo id"):
            EvaluadorSoloId.create_or_resume(evaluator_id)

    def test_create_or_resume_reports_unreadable_pickle(self):
        evaluator_id = self._fresh_id("test_create_or_resume_unreadable")
        filename = Evaluator.pickle_filename(evaluator_id)
        with open(filename, "wb") as f:
            f.write(b"esto no es un pickle")
        # Error claro, sin crear un evaluador nuevo ni borrar el fichero
        factory = self._factory_spy()
        with self.assertRaises(RuntimeError):
            Evaluator.create_or_resume(evaluator_id, factory)
        factory.assert_not_called()
        self.assertTrue(os.path.exists(filename))

    def test_persistent_decorator_is_deprecated_but_still_works(self):
        # Compatibilidad con los programas que usan el decorador: la función
        # decorada hace de factory de create_or_resume().
        evaluator_id = self._fresh_id("test_persistent_decorator")
        factory = self._factory_spy()
        with self.assertWarns(DeprecationWarning):
            evaluador_persistente = Evaluator.persistent(factory)

        created = evaluador_persistente(evaluator_id)
        factory.assert_called_once_with(evaluator_id)
        created._persist_evaluator()

        with redirect_stdout(io.StringIO()):
            resumed = evaluador_persistente(evaluator_id)
        factory.assert_called_once()  # al reanudar no se vuelve a llamar
        self.assertEqual(resumed.evaluator_id, evaluator_id)

    def test_discard_saved(self):
        evaluator_id = self._fresh_id("test_discard_saved")
        self._new_evaluator(evaluator_id)._persist_evaluator()
        self.assertTrue(Evaluator.discard_saved(evaluator_id))
        self.assertFalse(os.path.exists(Evaluator.pickle_filename(evaluator_id)))
        self.assertFalse(Evaluator.discard_saved(evaluator_id))

    def test_create_or_resume_preserves_subclass(self):
        evaluator_id = self._fresh_id("test_create_or_resume_subclass")
        original = SubEvaluadorParaTest(
            evaluator_id=evaluator_id,
            student_responses=pd.DataFrame(respuestas_estudiantes),
            prompt=PromptFromString(PROMPT),
            gpt_manager='mock',
        )
        original._persist_evaluator()

        # Invocado desde la clase BASE, no desde la subclase: debe recuperar
        # igualmente un SubEvaluadorParaTest, porque pickle conserva el tipo
        # real, no el tipo por el que se invoca create_or_resume.
        resumed = Evaluator.create_or_resume(evaluator_id=evaluator_id)
        self.assertIsInstance(resumed, SubEvaluadorParaTest)

    def test_send_gpt_queries_does_not_persist_with_mock_manager(self):
        evaluator_id = self._fresh_id("test_send_gpt_queries_mock")
        evaluator = Evaluator(
            evaluator_id=evaluator_id,
            student_responses=pd.DataFrame(respuestas_estudiantes),
            prompt=PromptFromString(PROMPT),
            gpt_manager='mock',
        )
        evaluator.send_gpt_queries()
        self.assertFalse(os.path.exists(Evaluator.pickle_filename(evaluator_id)))

    @patch(
        'evalia.gpt_manager.gpt_batch_manager.GPTBatchManager.start_task',
        return_value=GPTBatchManager.GPTBatchTask("fake_batch_id"),
    )
    def test_send_gpt_queries_persists_with_batch_manager(self, mock_start_task):
        evaluator_id = self._fresh_id("test_send_gpt_queries_batch")
        evaluator = Evaluator(
            evaluator_id=evaluator_id,
            student_responses=pd.DataFrame(respuestas_estudiantes),
            prompt=PromptFromString(PROMPT),
            gpt_manager=GPTBatchManager(model="gpt-4o-mini"),
        )
        evaluator.send_gpt_queries()
        mock_start_task.assert_called_once()
        self.assertTrue(os.path.exists(Evaluator.pickle_filename(evaluator_id)))


if __name__ == '__main__':
    unittest.main()