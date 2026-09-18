from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .baseline import evaluate_grouped_probe
from .nulls import subject_label_permutation_test
from .synthetic import make_neural_embeddings


def _write(result, out: Path, null=None):
    out.mkdir(parents=True, exist_ok=True)
    result.folds.to_csv(out / "folds.csv", index=False)
    payload = dict(result.summary)
    if null is not None:
        payload["permutation_test"] = {
            k: v for k, v in null.items() if k != "null_scores"
        }
        pd.DataFrame({"null_roc_auc": null["null_scores"]}).to_csv(
            out / "null_distribution.csv", index=False
        )
    (out / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser(prog="neural-eval")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo")
    demo.add_argument("--subjects", type=int, default=40)
    demo.add_argument("--trials", type=int, default=30)
    demo.add_argument("--features", type=int, default=64)
    demo.add_argument("--splits", type=int, default=5)
    demo.add_argument("--seed", type=int, default=11)
    demo.add_argument("--permutations", type=int, default=25)
    demo.add_argument("--out", type=Path, default=Path("results/demo"))

    ev = sub.add_parser("evaluate")
    ev.add_argument("csv", type=Path)
    ev.add_argument("--label", required=True)
    ev.add_argument("--group", required=True)
    ev.add_argument("--feature-prefix", default="emb_")
    ev.add_argument("--splits", type=int, default=5)
    ev.add_argument("--seed", type=int, default=11)
    ev.add_argument("--out", type=Path, default=Path("results/eval"))
    return parser


def main():
    args = build_parser().parse_args()

    if args.command == "demo":
        X, y, groups = make_neural_embeddings(
            n_subjects=args.subjects,
            trials_per_subject=args.trials,
            n_features=args.features,
            seed=args.seed,
        )
        result = evaluate_grouped_probe(
            X, y, groups, n_splits=args.splits, seed=args.seed
        )
        null = subject_label_permutation_test(
            X,
            y,
            groups,
            observed_score=result.summary["roc_auc_mean"],
            n_permutations=args.permutations,
            n_splits=args.splits,
            seed=args.seed,
        )
        _write(result, args.out, null)
        return

    df = pd.read_csv(args.csv)
    features = [c for c in df.columns if c.startswith(args.feature_prefix)]
    if not features:
        raise ValueError("no feature columns matched --feature-prefix")

    result = evaluate_grouped_probe(
        df[features].to_numpy(),
        df[args.label].to_numpy(),
        df[args.group].to_numpy(),
        n_splits=args.splits,
        seed=args.seed,
    )
    _write(result, args.out)


if __name__ == "__main__":
    main()
