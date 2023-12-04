from sklearn.preprocessing import LabelEncoder


class FeatureEngineer:
    """A simple feature engineering class

    Attributes:
        data: The data to engineer features for

    Methods:
        engineer_features: Engineer features for the data

    """

    def __init__(self, data):
        self.data = data

    def engineer_features(self):
        # Convert categorical variables to numerical
        label_encoder = LabelEncoder()
        self.data["Sex"] = label_encoder.fit_transform(self.data["Sex"])
        self.data["Embarked"] = label_encoder.fit_transform(self.data["Embarked"])
        return self.data
