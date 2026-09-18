from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_probe(kind: str = "linear", seed: int = 0):
    """Construct a standardized downstream probe for frozen representations."""
    if kind == "linear":
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=seed,
        )
    elif kind == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=1e-3,
            max_iter=600,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=seed,
        )
    else:
        raise ValueError("kind must be 'linear' or 'mlp'")

    return Pipeline([("scale", StandardScaler()), ("clf", clf)])
