import unittest
import pandas as pd
from mlads_ds.data_preprocessor import (
    DataPreprocessor,
)  # assuming this is the class where your preprocess method is


class TestDataPreprocessor(unittest.TestCase):
    def setUp(self):
        self.preprocessor = DataPreprocessor()
        self.data = pd.DataFrame(
            {
                "Name": ["John", "Doe", "Foo"],
                "Ticket": ["A23", "B45", "C67"],
                "Cabin": ["C1", "C2", "C3"],
                "Age": [22, pd.np.nan, 30],
                "Embarked": ["S", "C", pd.np.nan],
            }
        )
        self.preprocessor.data = self.data

    def test_preprocess(self):
        processed_data = self.preprocessor.preprocess()
        self.assertNotIn(["Name", "Ticket", "Cabin"], processed_data.columns)
        self.assertFalse(processed_data["Age"].isnull().any())
        self.assertFalse(processed_data["Embarked"].isnull().any())


if __name__ == "__main__":
    unittest.main()
