from sklearn.metrics import accuracy_score


class ModelEvaluator:
    """Evaluates a model on a test set

    Attributes:
        model: A trained model
        X_test: Test features
        y_test: Test labels

    Methods:
        evaluate: Evaluates the model on the test set

    """

    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

    def evaluate(self):
        """Evaluates the model on the test set

        Returns:
            float: The accuracy of the model on the test set"""
        predictions = self.model.predict(self.X_test)
        return accuracy_score(self.y_test, predictions)
