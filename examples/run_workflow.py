"""Minimal script for an MPM-style CSV held locally by the user."""

from pathlib import Path

import pandas as pd

from mpm_workflow import MPMConfig, evaluate_candidates, evaluate_high_priority, fit_mpm, predict_mpm, save_model

DATA_PATH = Path("data/data_mpm.csv")
ARTIFACTS = Path("artifacts")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Place your local input CSV at {DATA_PATH} or edit DATA_PATH.")
    ARTIFACTS.mkdir(exist_ok=True)
    cells = pd.read_csv(DATA_PATH)
    config = MPMConfig(random_state=42, n_estimators=300)
    metrics = evaluate_candidates(cells, config=config)
    model = fit_mpm(cells, config=config, model_name="random_forest")
    predictions = predict_mpm(model, cells)
    metrics.to_csv(ARTIFACTS / "candidate_metrics.csv", index=False)
    predictions.to_csv(ARTIFACTS / "mpm_predictions.csv", index=False)
    save_model(model, ARTIFACTS / "mpm_random_forest.joblib")
    print(metrics.round(3).to_string(index=False))
    print(evaluate_high_priority(cells, predictions, config=config))


if __name__ == "__main__":
    main()
