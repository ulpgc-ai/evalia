import unittest
from unittest.mock import patch
from openai import AuthenticationError, BadRequestError
from openai.resources.chat.completions import Completions
from openai.types.chat import ChatCompletion
from evalia.gpt_manager.gpt_smart_manager import Request, RequestQueue, HistoryRecord, GPTSmartManager
from .utils import high_cost


# Límites (rpm, tpm) simulados para los tests que no ponen a prueba el
# descubrimiento de límites en sí (eso lo cubre TestRequestQueue): así no
# dependen de la red ni de una clave de API real.
FAKE_LIMITS = (500, 10_000)


def _fake_chat_completion(content="¡Hola! ¿En qué puedo ayudarte hoy?"):
    """Respuesta de OpenAI simulada, usando el propio tipo del SDK para que
    se comporte como una respuesta real (acceso por atributos, no por claves)."""
    return ChatCompletion(
        id="chatcmpl-test",
        object="chat.completion",
        created=1697920381,
        model="gpt-3.5-turbo-0613",
        choices=[{
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": content},
        }],
        usage={"prompt_tokens": 8, "completion_tokens": 11, "total_tokens": 19},
    )


def _mock_create(self, *args, **kwargs):
    """Sustituye a `Completions.create` (afecta también a `with_raw_response.create`)."""
    return _fake_chat_completion()


class TestRequest(unittest.TestCase):
    def test_request(self):
        request = Request(1234)
        self.assertEqual(request.tokens, 1234)


class TestRequestQueue(unittest.TestCase):

    # Check that request not exceeding the maximum tokens is accepted
    @patch.object(RequestQueue, '_discover_limits', return_value=FAKE_LIMITS)
    def test_requestqueue_1(self, _mock_discover_limits):
        rq = RequestQueue("gpt-4", client=None)
        with self.assertRaises(Exception) as context:
            rq.add(Request(150_000_000))
        self.assertTrue("tokens exceeds the maximum" in str(context.exception))

    # Check that the current minute tokens are calculated correctly
    @unittest.skip("Not neccesary")
    @patch.object(RequestQueue, '_discover_limits', return_value=FAKE_LIMITS)
    def test_requestqueue_2(self, _mock_discover_limits):
        rq = RequestQueue("gpt-4", client=None)
        rq.add(Request(1000))
        rq.add(Request(2000))
        rq.add(Request(3000))
        self.assertEqual(rq.current_minute_tokens, 6000)


class TestGPTManager(unittest.TestCase):

    @high_cost
    @patch('evalia.gpt_manager.gpt_smart_manager.OPENAI_API_KEY', 'invalid-api-key')
    def test_invalid_api_key_is_managed(self):
        # Con la clave inválida, el propio descubrimiento de límites en el
        # constructor falla al autenticarse contra la API real de OpenAI.
        with self.assertRaises(AuthenticationError):
            GPTSmartManager(model="gpt-4")

    @patch.object(RequestQueue, '_discover_limits', return_value=FAKE_LIMITS)
    @patch.object(Completions, 'create', new=_mock_create)
    def test_basic_interaction(self, _mock_discover_limits):
        gpt_manager = GPTSmartManager(model="gpt-4")
        response = gpt_manager.query([{"role": "user", "content": "Hola"}])

        # check that the response contains the word "hola"
        self.assertIn("Hola", response[0].choices[0].message.content)
        self.assertEqual(response[0].choices[0].finish_reason, "stop")

    @patch.object(RequestQueue, '_discover_limits', return_value=FAKE_LIMITS)
    def test_bad_message(self, _mock_discover_limits):
        gpt_manager = GPTSmartManager(model="gpt-4")
        with self.assertRaises(Exception) as context:
            response = gpt_manager.query("this is a wrongly formatted message")

    @patch.object(RequestQueue, '_discover_limits', return_value=FAKE_LIMITS)
    @patch.object(Completions, 'create', new=_mock_create)
    def test_send_queries(self, _mock_discover_limits):
        gpt_manager = GPTSmartManager(model="gpt-4")
        query_list = [
            [{"role" : "user", "content" : "Hola"}],
            [{"role" : "user", "content" : "Hola, dime hola."}],
        ]
        responses, stats = gpt_manager.send_queries(
            query_id="test",
            query_list=query_list,
            temperature=0.0
        )
        self.assertEqual(len(responses), 2)
        # check that stats is a dict with a key "input_tokens" with a numeric value
        self.assertIsInstance(stats, dict)
        self.assertIn("input_tokens", stats)
        self.assertIn("output_tokens", stats)
        self.assertIn("elapsed_time", stats)

    @patch.object(RequestQueue, '_discover_limits', return_value=FAKE_LIMITS)
    def test_bad_queries(self, _mock_discover_limits):
        gpt_manager = GPTSmartManager(model="gpt-4")
        with self.assertRaises(Exception) as context:
            response = gpt_manager.send_queries(
                query_id="test",
                query_list= None,
                temperature=0.0
                )
        with self.assertRaises(Exception) as context:
            response = gpt_manager.send_queries(
                query_id="test",
                query_list="bad query, should be an OpenAI message or a list of OpenAI messages",
                temperature=0.0
                )
        with self.assertRaises(Exception) as context:
            response = gpt_manager.send_queries(
                query_id="test",
                query_list=[{"role" : "user", "content" : "Hola"}, "bad query"],
               temperature=0.0
                 )


class TestDiscoverLimits(unittest.TestCase):
    """El constructor de GPTSmartManager descubre los límites RPM/TPM con una
    petición mínima real al modelo. Esa petición tiene que ser válida para
    todos los modelos, incluidos los gpt-5*. Usan la API real de OpenAI."""

    def _assert_manager_can_be_created(self, model):
        try:
            gpt_manager = GPTSmartManager(model=model)
        except BadRequestError as e:
            self.fail(f'GPTSmartManager(model="{model}") no se pudo crear: {e}')
        self.assertGreater(gpt_manager.request_queue.rpm, 0)
        self.assertGreater(gpt_manager.request_queue.tpm, 0)

    #@high_cost
    def test_discover_limits_gpt4(self):
        # Control: con un modelo anterior, el descubrimiento funciona
        self._assert_manager_can_be_created("gpt-4o-mini")

    #@high_cost
    def test_discover_limits_gpt5(self):
        for model in ("gpt-5", "gpt-5-mini", "gpt-5.1", "gpt-5.5"):
            with self.subTest(model=model):
                self._assert_manager_can_be_created(model)

    def test_discover_limits_gpt6(self):
        for model in ("gpt-6-sol", "gpt-6-luna", "gpt-6-astra"):
            with self.subTest(model=model):
                self._assert_manager_can_be_created(model)

if __name__ == '__main__':
    unittest.main()
