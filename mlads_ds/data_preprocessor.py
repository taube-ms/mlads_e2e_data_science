import pandas as pd


class DataPreprocessor:
    """A simple data preprocessor class

    Attributes:
        data: The data to preprocess

    Methods:
        preprocess: Preprocesses the data

    """

    def __init__(self, data):
        self.data = data

    def preprocess(self):
        # Basic preprocessing steps
        self.data = self.data.drop(["Name", "Ticket", "Cabin"], axis=1)
        self.data["Age"].fillna(self.data["Age"].median(), inplace=True)
        self.data["Embarked"].fillna(self.data["Embarked"].mode()[0], inplace=True)
        return self.data
