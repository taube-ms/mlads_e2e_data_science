import unittest
from mlads_ds.feature_selector import FeatureSelector
import pandas as pd


class TestFeatureSelector(unittest.TestCase):
    def test_select_features(self):
        test_data = pd.DataFrame(
            {
                "Pclass": [1, 2, 3],
                "Age": [22, 28, 35],
                "SibSp": [1, 0, 0],
                "Parch": [0, 0, 0],
                "Fare": [7.25, 71.2833, 8.05],
                "Sex_male": [0, 1, 1],
                "Embarked_Q": [0, 0, 0],
                "Embarked_S": [1, 0, 1],
            }
        )
        selector = FeatureSelector(test_data)
        features = selector.select_features()
        self.assertEqual(features.shape[1], 8)  # Assuming you are selecting 8 features


if __name__ == "__main__":
    unittest.main()
