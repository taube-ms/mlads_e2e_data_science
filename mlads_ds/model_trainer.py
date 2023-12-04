# Enhancing the data science pipeline to be more robust, allowing for the comparison of different models and hyperparameters
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


class ModelTrainer:
    """Train and compare multiple machine learning models with different hyperparameters.

    Args:
        features (pd.DataFrame): Dataframe containing the features
        labels (pd.DataFrame): Dataframe containing the labels

    Returns:
        dict: Dictionary containing the best models for each algorithm
        dict: Dictionary containing the evaluation results for each algorithm
    """

    def __init__(
        self,
        features,
        labels,
        models={
            "RandomForest": RandomForestClassifier(),
            "SVM": SVC(),
            "LogisticRegression": LogisticRegression(),
        },
        hyperparameters={
            "RandomForest": {
                "n_estimators": [10, 50, 100, 200],
                "max_depth": [None, 10, 20, 30],
            },
            "SVM": {"C": [0.1, 1, 10], "gamma": [1, 0.1, 0.01]},
            "LogisticRegression": {"C": [0.1, 1, 10]},
        },
    ):
        """Initialize the ModelTrainer class.

        Args:
            features (pd.DataFrame): Dataframe containing the features
            labels (pd.DataFrame): Dataframe containing the labels
            models (dict): Dictionary containing the models to be trained
            hyperparameters (dict): Dictionary containing the hyperparameters to be tested

        Returns:
            None"""
        self.features = features
        self.labels = labels
        self.models = models
        self.hyperparameters = hyperparameters

    def train_and_evaluate_models(self):
        """
        Train and evaluate the models with different hyperparameters.

        Returns:
            dict: Dictionary containing the best models for each algorithm
            dict: Dictionary containing the evaluation results for each algorithm
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.labels, test_size=0.2, random_state=0
        )
        best_models = {}
        evaluation_results = {}

        for model_name, model in self.models.items():
            print(f"Training and evaluating {model_name}...")
            grid_search = GridSearchCV(
                model, self.hyperparameters[model_name], cv=5, scoring="accuracy"
            )
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            best_models[model_name] = best_model

            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)

            evaluation_results[model_name] = {
                "Best Parameters": grid_search.best_params_,
                "Accuracy": accuracy,
                "Classification Report": report,
            }

        return best_models, evaluation_results
