# TEST: comprobar que se generan las queries correctamente
# Se toma una muestra desde un dataset real
# Se generan las queries con el método Evaluator.build_gpt_queries()
# Es importante jugar con el parámetro QUERY_BATCH_LENGTH

import sys
sys.path.append('.')
import pandas as pd
from evaluators import Evaluator
from prompts import PromptFromTextFile

MODEL = "gpt-4"
#MODEL = "gpt-3.5-turbo"

test_dataframe = pd.read_excel("tests/dataset_test_4ESO_15.xlsx")
test_prompt=PromptFromTextFile("tests/prompt_test_4ESO_15.txt")
SLICE_RANGE=slice(00,20)

QUERY_BATCH_LENGTH = 7
# probar valores como 1, 7, 20

test_item = Evaluator(
    evaluator_id="test evaluator", 
    student_responses=test_dataframe,
    sample_selector=SLICE_RANGE,
    prompt=test_prompt,
    query_batch_length=QUERY_BATCH_LENGTH
)

test_item.model = MODEL
test_item.temperature = 0.2

queries = test_item.build_gpt_queries()

print(queries)
