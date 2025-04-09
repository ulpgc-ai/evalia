import sys
sys.path.append('.')
import pandas as pd
from evaluators import Evaluator
from prompts import PromptFromTextFile
#from gpt import GPTMockManager as GPTManager
from gpt_manager import GPTSmartManager as GPTManager

#MODEL = "gpt-4"
GPT_MODEL = "gpt-3.5-turbo"

test_dataframe = pd.read_excel("tests/dataset_test_4ESO_15.xlsx")
test_prompt=PromptFromTextFile("tests/prompt_test_4ESO_15.txt")
SLICE_RANGE=slice(00,20)

QUERY_BATCH_LENGTH = 7
# probar valores como 1, 7, 20

gpt = GPTManager(GPT_MODEL)
gpt.initialize()

evaluator = Evaluator(
    evaluator_id="test evaluator", 
    student_responses=test_dataframe,
    sample_selector=SLICE_RANGE,
    prompt=test_prompt,
    gpt_manager=gpt,
    query_batch_length=QUERY_BATCH_LENGTH,
)

evaluator.temperature = 0.2

df = evaluator.evaluate_answers()

print(df)
