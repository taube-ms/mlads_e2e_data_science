import pandas as pd


class DataPreprocessor:
    """Preprocess the data: handle missing values, encode categorical variables, etc.

    Args:
        data (pd.DataFrame): Dataframe containing the data

    Returns:
        pd.DataFrame: Dataframe containing the preprocessed data
    """

    def __init__(self, data):
        self.data = data

    def preprocess(self):
        # Filling missing values
        self.data["Age"].fillna(self.data["Age"].median(), inplace=True)
        self.data["Embarked"].fillna(self.data["Embarked"].mode()[0], inplace=True)
        self.data.drop(
            "Cabin", axis=1, inplace=True
        )  # Dropping the Cabin column due to high missing values

        # Encoding categorical variables
        self.data = pd.get_dummies(
            self.data, columns=["Sex", "Embarked"], drop_first=True
        )

        return self.data
