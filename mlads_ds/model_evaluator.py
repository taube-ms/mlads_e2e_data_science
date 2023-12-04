from sklearn.metrics import classification_report, accuracy_score


class ModelEvaluator:
    """Evaluate the trained model.

    Args:
        model (sklearn.model): Trained model
        X_test (pd.DataFrame): Test set
        y_test (pd.DataFrame): Test labels

    Returns:
        float: Accuracy score
        str: Classification report"""

    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

    def evaluate(self):
        # Predicting the Test set results
        y_pred = self.model.predict(self.X_test)

        # Creating the evaluation report
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)

        return accuracy, report
