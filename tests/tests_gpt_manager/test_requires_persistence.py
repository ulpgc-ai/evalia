import unittest

from evalia.gpt_manager import GPTManager, GPTBatchManager, GPTSmartManager, GPTMockManager


class TestRequiresPersistence(unittest.TestCase):
    """`requires_persistence` es lo que usa Evaluator.send_gpt_queries() para
    decidir si persiste tras enviar una tarea, sin conocer la implementación
    concreta de cada manager. Es un atributo de clase: estas comprobaciones
    no instancian ningún manager ni tocan la red."""

    def test_default_is_false(self):
        # Cualquier GPTManager futuro que no lo declare cae en el valor
        # seguro por defecto: no persiste.
        self.assertFalse(GPTManager.requires_persistence)

    def test_batch_manager_requires_persistence(self):
        # Su tarea vive en OpenAI hasta 24h, independiente del proceso.
        self.assertTrue(GPTBatchManager.requires_persistence)

    def test_smart_manager_does_not_require_persistence(self):
        # Es síncrono: la respuesta ya está completa cuando start_task() vuelve.
        self.assertFalse(GPTSmartManager.requires_persistence)

    def test_mock_manager_does_not_require_persistence(self):
        self.assertFalse(GPTMockManager.requires_persistence)


if __name__ == '__main__':
    unittest.main()
