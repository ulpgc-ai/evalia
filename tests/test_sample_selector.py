import unittest
import sys
import os
sys.path.append('.')
import pandas as pd
from evalia.evaluators import BaseEvaluator

class TestSampleSelector(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        test_dir = os.path.dirname(__file__)
        data_path = os.path.join(test_dir, "dataset_test_4ESO_15.xlsx")
        cls.test_dataframe = pd.read_excel(data_path)
        cls.test_eva = BaseEvaluator(
            evaluator_id="test evaluator", 
            student_responses=cls.test_dataframe,
        )

    def test_invalid_selector(self):
        self.test_eva.sample_answers = None
        # tipo inválido (str)
        self.test_eva.sample_selector = "invalid selector"
        with self.assertRaises(TypeError):
            self.test_eva.read_sample_answers()

    def test_slice_selector(self):
        self.test_eva.sample_answers = None
        self.test_eva.sample_selector = slice(1,20,3)
        df = self.test_eva.read_sample_answers()
        # Debería haber 7 elementos
        self.assertEqual(df.shape[0], 7)

    def test_random_selector(self):
        self.test_eva.sample_answers = None
        self.test_eva.sample_selector = 10
        df = self.test_eva.read_sample_answers()
        # Debería haber 10 elementos
        self.assertEqual(df.shape[0], 10)

    def test_lambda_selector(self):
        self.test_eva.sample_answers = None
        self.test_eva.sample_selector = lambda df : df.sample(10)
        df = self.test_eva.read_sample_answers()
        # Debería haber 10 elementos
        self.assertEqual(df.shape[0], 10)

    def test_filter_selector(self):
        self.test_eva.sample_answers = None
        self.test_eva.sample_selector = lambda df : df[df["Calificación 15"]==1]
        df = self.test_eva.read_sample_answers()
        # todos los elementos tienen calificación 1
        self.assertTrue(all(df["Calificación 15"]==1))

    def test_list_selector(self):
        self.test_eva.sample_answers = None
        selected_indices = [ 1, 7, 46, 99, 45 ]
        self.test_eva.sample_selector = selected_indices
        df = self.test_eva.read_sample_answers()

        actual_indices = df.index.tolist()
        # Los índices deberían ser los mismos
        self.assertEqual(set(actual_indices), set(selected_indices))
    
if __name__ == '__main__':
    unittest.main()