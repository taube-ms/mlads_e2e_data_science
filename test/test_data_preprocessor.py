import unittest
from mlads_ds.data_preprocessor import DataPreprocessor
import pandas as pd


class TestDataPreprocessor(unittest.TestCase):
    def test_preprocess(self):
        test_data = pd.DataFrame(
            {
                "Age": [30, None, 25],
                "Sex": ["male", "female", "male"],
                "Embarked": ["S", "C", None],
                "Cabin": [None, "C123", None],
            }
        )
        preprocessor = DataPreprocessor(test_data)
        processed_data = preprocessor.preprocess()
        self.assertFalse(
            processed_data.isnull().any().any()
        )  # Check if there are no null values


if __name__ == "__main__":
    unittest.main()
