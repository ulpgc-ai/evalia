import unittest
import os
from functools import wraps

# Decorador para omitir tests de alto coste (en tiempo o dinero)
# Por ejemplo, tests que invocan a la API de OpenAI
def high_cost(test_func):
    @wraps(test_func)
    def wrapper(*args, **kwargs):
        if os.getenv("EVALIA_RUN_HIGH_COST_TESTS", "0") != "1":
            raise unittest.SkipTest("Test de alto coste omitido")
        return test_func(*args, **kwargs)
    return wrapper