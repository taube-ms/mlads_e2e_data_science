from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC


class ModelTrainer:
    """Trains and tunes a logistic regression, random forest, and SVM model
    using the provided features and target. Returns a dictionary of results
    for each model.

    Args:
        features (pandas.DataFrame): Features to train the model on.
        target (pandas.Series): Target to train the model on.

    Returns:
        dict: Dictionary of results for each model.
    """

    def __init__(self, features, target):
        self.features = features
        self.target = target
        self.models = {
            "logistic_regression": LogisticRegression(),
            "random_forest": RandomForestClassifier(),
            "svm": SVC(),
            "knn": KNeighborsClassifier(),
            "decision_tree": DecisionTreeClassifier(),
            "gradient_boosting": GradientBoostingClassifier(),
            "adaboost": AdaBoostClassifier(),
            "xgboost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
            "lightgbm": LGBMClassifier(),
            "naive_bayes": GaussianNB(),
            "linear_svc": LinearSVC(),
            "extra_trees": ExtraTreesClassifier(),
        }
        self.hyperparameters = {
            "logistic_regression": {"model__C": [0.1, 1, 10]},
            "random_forest": {"model__n_estimators": [10, 50, 100]},
            "svm": {"model__C": [0.1, 1, 10], "model__gamma": ["scale", "auto"]},
            "knn": {
                "model__n_neighbors": [3, 5, 11, 19],
                "model__weights": ["uniform", "distance"],
                "model__metric": ["euclidean", "manhattan"],
            },
            "decision_tree": {
                "model__max_depth": [None, 10, 20, 30, 40, 50],
                "model__min_samples_split": [2, 5, 10],
            },
            "gradient_boosting": {
                "model__n_estimators": [100, 200, 300],
                "model__learning_rate": [0.01, 0.1, 0.2],
                "model__max_depth": [3, 5, 10],
            },
            "adaboost": {
                "model__n_estimators": [50, 100, 200],
                "model__learning_rate": [0.01, 0.1, 1],
            },
            "xgboost": {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.01, 0.1],
                "model__max_depth": [3, 5, 7],
            },
            "lightgbm": {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.01, 0.1],
                "model__num_leaves": [31, 50, 100],
            },
            "naive_bayes": {},  # Naive Bayes usually does not need hyperparameter tuning
            "linear_svc": {"model__C": [0.1, 1, 10]},
            "extra_trees": {
                "model__n_estimators": [50, 100, 200],
                "model__max_depth": [None, 10, 20, 30],
            },
        }

    def train_and_tune(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.target, test_size=0.2, random_state=42
        )
        results = {}
        for model_name, model in self.models.items():
            pipeline = Pipeline(
                [
                    (
                        "preprocessor",
                        ColumnTransformer(
                            transformers=[
                                ("num", StandardScaler(), ["Age", "Fare"]),
                                ("cat", OneHotEncoder(), ["Sex", "Embarked"]),
                            ]
                        ),
                    ),
                    ("model", model),
                ]
            )
            grid_search = GridSearchCV(pipeline, self.hyperparameters[model_name], cv=5)
            grid_search.fit(X_train, y_train)
            results[model_name] = {
                "best_params": grid_search.best_params_,
                "best_score": grid_search.best_score_,
                "test_score": grid_search.score(X_test, y_test),
                "confusion_matrix": confusion_matrix(
                    y_test, grid_search.predict(X_test)
                ),
                "classification_report": classification_report(
                    y_test, grid_search.predict(X_test)
                ),
            }
        return results
