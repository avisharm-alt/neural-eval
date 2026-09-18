from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score


def binary_metrics(y_true, probability, threshold: float = 0.5) -> dict[str, float]:
    """Compute clinically interpretable binary classification metrics."""
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    prediction = (probability >= threshold).astype(int)

    cm = confusion_matrix(y_true, prediction, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")

    if len(np.unique(y_true)) == 2:
        auc = roc_auc_score(y_true, probability)
    else:
        auc = float("nan")

    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "roc_auc": float(auc),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
    }


def bootstrap_ci(
    values,
    confidence: float = 0.95,
    n_boot: int = 5000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for a 1D metric vector."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        boot[i] = rng.choice(x, size=len(x), replace=True).mean()

    alpha = 1.0 - confidence
    return (
        float(np.quantile(boot, alpha / 2)),
        float(np.quantile(boot, 1 - alpha / 2)),
    )
