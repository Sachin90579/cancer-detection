"""
Model factory: builds the classifiers compared in this project.

Each model is wrapped in a scikit-learn Pipeline with StandardScaler so
feature scaling is handled consistently and can't leak across CV folds
when used with cross_val_predict / GridSearchCV.
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_models(random_state: int = 42) -> dict[str, Pipeline]:
    """Return a dict of {model_name: sklearn Pipeline} ready to fit."""
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                        random_state=random_state)),
        ]),
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=8, class_weight="balanced",
                random_state=random_state,
            )),
        ]),
        "mlp": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(64, 32), activation="relu",
                alpha=1e-3, max_iter=2000, early_stopping=True,
                random_state=random_state,
            )),
        ]),
    }
