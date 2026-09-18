# neural-eval

**Evaluation infrastructure for brain foundation models and neural representation learning.**

`neural-eval` is an AI/ML toolkit for asking whether embeddings from EEG, MEG, fMRI, MRI, or multimodal neural encoders actually generalize to unseen people.

The central question is: **does a learned neural representation carry transferable task signal, or is downstream performance inflated by subject identity, preprocessing leakage, or an overpowered probe?**

## AI/ML focus

- **Foundation-model evaluation** — benchmark frozen embeddings from neural encoders and multimodal models
- **Linear probing** — measure linearly accessible information in learned representations
- **Nonlinear probing** — compare a small MLP probe against the linear baseline
- **Subject-held-out generalization** — stratified group CV with hard subject-disjoint assertions
- **Representation diagnostics** — effective rank, embedding norms, and cosine anisotropy
- **Leakage detection** — duplicate-feature checks across train/test folds
- **Statistical controls** — bootstrap confidence intervals and subject-level permutation tests
- **Synthetic stress tests** — controllable task signal + subject fingerprint nuisance signal

## Why this matters for foundation models

A neural foundation model can produce impressive-looking downstream numbers without learning representations that transfer across patients. If the same subject appears in training and test trials, or if a high-capacity probe memorizes idiosyncratic features, reported performance can overstate generalization.

This repository makes the **evaluation contract** explicit.

## Install

```bash
git clone https://github.com/avisharm-alt/neural-eval
cd neural-eval
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quickstart: foundation-model embeddings

Given one embedding per EEG trial or imaging sample:

```python
from neural_eval.foundation import evaluate_probe_suite
from neural_eval.diagnostics import representation_diagnostics

report = evaluate_probe_suite(
    X=embeddings,
    y=labels,
    groups=subject_ids,
    n_splits=5,
    seed=11,
)

print(report)
print(representation_diagnostics(embeddings))
```

The probe suite evaluates both:

1. **Linear probe** — standardized logistic regression
2. **MLP probe** — a compact nonlinear neural-network classifier

A large nonlinear-over-linear gap suggests useful information may be present but not linearly organized. Strong performance from both probes under subject-held-out evaluation is a more meaningful result than random trial splitting.

## CLI

Synthetic benchmark:

```bash
neural-eval demo --subjects 40 --trials 30 --features 64 --seed 11 --out results/demo
```

Evaluate exported model embeddings:

```bash
neural-eval evaluate embeddings.csv \
  --label diagnosis \
  --group subject_id \
  --feature-prefix emb_ \
  --splits 5 \
  --out results/model
```

## Supported research workflows

The package is intended for embeddings from models such as:

- EEG transformers
- masked neural-signal encoders
- multimodal EEG + MRI models
- MRI foundation models
- self-supervised neural encoders
- clinical multimodal foundation models

It does **not** claim that a specific foundation model is clinically valid. It provides the machinery needed to test stronger generalization claims.

## Evaluation principles

1. Split by biological unit, never by arbitrary row.
2. Fit all preprocessing inside the training fold.
3. Compare simple and nonlinear probes.
4. Audit representation geometry.
5. Report uncertainty.
6. Run a subject-level null.
7. Treat leakage checks as assertions, not optional plots.

## License

MIT
