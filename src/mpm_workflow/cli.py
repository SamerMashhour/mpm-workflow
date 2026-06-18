"""Command-line interface for the MPM workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .core import MPMConfig, evaluate_candidates, evaluate_high_priority, fit_mpm, load_model, predict_mpm, save_model
from .synthetic import SyntheticMPMConfig, write_synthetic_mpm


def _train(args: argparse.Namespace) -> int:
    data = pd.read_csv(args.data)
    config = MPMConfig(random_state=args.random_state, n_estimators=args.n_estimators, numeric_transform="legacy_clipped_log" if args.legacy_clipped_log else "quantile")
    metrics = evaluate_candidates(data, config=config, model_names=tuple(args.candidate_models))
    Path(args.metrics_output).parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.metrics_output, index=False)
    save_model(fit_mpm(data, config=config, model_name=args.model), args.model_output)
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    return 0


def _predict(args: argparse.Namespace) -> int:
    data = pd.read_csv(args.data)
    model = load_model(args.model)
    output = predict_mpm(model, data, low_quantile=args.low_quantile, high_quantile=args.high_quantile)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(json.dumps(evaluate_high_priority(data, output, config=model.config), indent=2))
    return 0


def _generate(args: argparse.Namespace) -> int:
    config = SyntheticMPMConfig(
        scenario=args.scenario,
        n_cells=args.n_cells,
        positive_count=args.positive_count,
        negative_training_count=args.negative_training_count,
        random_state=args.random_state,
    )
    output = write_synthetic_mpm(args.output, config)
    print(f"Saved {output} ({config.n_cells:,} cells).")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tabular mineral prospectivity mapping workflow.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="Evaluate candidates and fit a model.")
    train.add_argument("--data", required=True, type=Path)
    train.add_argument("--model-output", default=Path("artifacts/mpm_model.joblib"), type=Path)
    train.add_argument("--metrics-output", default=Path("artifacts/model_metrics.csv"), type=Path)
    train.add_argument("--model", choices=["random_forest", "svm_rbf", "adaboost", "mlp"], default="random_forest")
    train.add_argument("--candidate-models", nargs="+", choices=["random_forest", "svm_rbf", "adaboost", "mlp"], default=["svm_rbf", "random_forest", "adaboost", "mlp"])
    train.add_argument("--n-estimators", type=int, default=300)
    train.add_argument("--random-state", type=int, default=42)
    train.add_argument("--legacy-clipped-log", action="store_true")
    train.set_defaults(func=_train)
    predict = subparsers.add_parser("predict", help="Predict every input cell.")
    predict.add_argument("--data", required=True, type=Path)
    predict.add_argument("--model", required=True, type=Path)
    predict.add_argument("--output", default=Path("artifacts/mpm_predictions.csv"), type=Path)
    predict.add_argument("--low-quantile", type=float, default=0.80)
    predict.add_argument("--high-quantile", type=float, default=0.90)
    predict.set_defaults(func=_predict)
    synthetic = subparsers.add_parser("generate-synthetic", help="Write a shareable synthetic MPM-style input.")
    synthetic.add_argument("--output", default=Path("data/generated_belt_cover.csv"), type=Path)
    synthetic.add_argument("--scenario", choices=["belt_cover", "structural_corridors"], default="belt_cover")
    synthetic.add_argument("--n-cells", type=int, default=30_000)
    synthetic.add_argument("--positive-count", type=int, default=760)
    synthetic.add_argument("--negative-training-count", type=int, default=760)
    synthetic.add_argument("--random-state", type=int, default=20260618)
    synthetic.set_defaults(func=_generate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
