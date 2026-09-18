from __future__ import annotations

import numpy as np


def make_neural_embeddings(
    n_subjects: int = 40,
    trials_per_subject: int = 30,
    n_features: int = 64,
    class_signal: float = 0.6,
    subject_signal: float = 1.2,
    noise: float = 1.0,
    seed: int = 0,
):
    """Create EEG-like trial embeddings with subject-specific nuisance signal.

    Labels are assigned at the subject level. Each trial combines class signal,
    a stable subject fingerprint, and independent noise.
    """
    if n_subjects < 4:
        raise ValueError("n_subjects must be >= 4")
    if n_features < 2:
        raise ValueError("n_features must be >= 2")

    rng = np.random.default_rng(seed)
    subject_labels = np.arange(n_subjects) % 2
    rng.shuffle(subject_labels)
    subject_vectors = rng.normal(0, subject_signal, size=(n_subjects, n_features))

    direction = rng.normal(size=n_features)
    direction /= np.linalg.norm(direction)

    X, y, groups = [], [], []
    for subject in range(n_subjects):
        label = int(subject_labels[subject])
        signed_class = (2 * label - 1) * class_signal * direction
        for _ in range(trials_per_subject):
            x = (
                subject_vectors[subject]
                + signed_class
                + rng.normal(0, noise, size=n_features)
            )
            X.append(x)
            y.append(label)
            groups.append(f"s{subject:03d}")

    return np.asarray(X), np.asarray(y), np.asarray(groups)
