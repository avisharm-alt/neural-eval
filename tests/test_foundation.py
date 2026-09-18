from neural_eval.diagnostics import representation_diagnostics
from neural_eval.foundation import evaluate_probe_suite
from neural_eval.synthetic import make_neural_embeddings


def test_probe_suite_runs_for_linear_and_mlp():
    X, y, groups = make_neural_embeddings(
        n_subjects=24,
        trials_per_subject=5,
        n_features=16,
        seed=21,
    )
    result = evaluate_probe_suite(X, y, groups, n_splits=3, seed=21)
    assert set(result["probe"]) == {"linear", "mlp"}
    assert len(result) == 6
    assert (result["duplicate_feature_rows"] == 0).all()


def test_representation_diagnostics_are_finite():
    X, _, _ = make_neural_embeddings(
        n_subjects=12,
        trials_per_subject=3,
        n_features=10,
        seed=22,
    )
    d = representation_diagnostics(X, seed=22)
    assert d["effective_rank"] > 0
    assert d["mean_l2_norm"] > 0
    assert 0 <= d["cosine_anisotropy"] <= 1
