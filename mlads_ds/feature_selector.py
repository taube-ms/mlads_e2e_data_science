import pandas as pd


class FeatureSelector:
    """Select features for the model.

    Args:
        data (pd.DataFrame): Dataframe containing the data

    Returns:
        pd.DataFrame: Dataframe containing the selected features"""

    def __init__(self, data):
        self.data = data

    def select_features(self):
        features = self.data[
            [
                "Pclass",
                "Age",
                "SibSp",
                "Parch",
                "Fare",
                "Sex_male",
                "Embarked_Q",
                "Embarked_S",
            ]
        ]
        return features
