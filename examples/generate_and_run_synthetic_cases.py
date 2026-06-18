"""Generate, train, score and map both synthetic MPM scenarios."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from mpm_workflow import MPMConfig, SyntheticMPMConfig, evaluate_candidates, evaluate_high_priority, fit_mpm, predict_mpm, write_synthetic_mpm

DATA = Path("data")
ARTIFACTS = Path("artifacts")
IMAGES = Path("docs/images")

CASES = {
    "belt_cover": SyntheticMPMConfig(scenario="belt_cover", random_state=20260701),
    "structural_corridors": SyntheticMPMConfig(
        scenario="structural_corridors",
        random_state=20260702,
        latitude_range=(43.5, 47.4),
        longitude_range=(-121.8, -112.3),
        label_noise=0.82,
    ),
}


def plot_targets(data: pd.DataFrame, output: Path, name: str) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 6.7))
    for target, label in (("medium", "Medium prospectivity target"), ("high", "High prospectivity target")):
        selected = data["MPM_BIN"].astype(str).eq(target)
        ax.scatter(data.loc[selected, "LONGITUDE"], data.loc[selected, "LATITUDE"], s=8, label=label, alpha=0.92, linewidths=0)
    occurrences = data["TRAINING_MINERAL_OCCURRENCE"].eq(1)
    deposits = data["TRAINING_DEPOSIT"].eq(1)
    ax.scatter(data.loc[occurrences, "LONGITUDE"], data.loc[occurrences, "LATITUDE"], s=25, marker="x", label="Synthetic occurrence")
    ax.scatter(data.loc[deposits, "LONGITUDE"], data.loc[deposits, "LATITUDE"], s=70, marker="*", label="Synthetic deposit")
    ax.set(title=f"Synthetic MPM case: {name.replace('_', ' ')}", xlabel="Synthetic longitude", ylabel="Synthetic latitude")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=190)
    plt.close(fig)


def main() -> None:
    DATA.mkdir(exist_ok=True)
    ARTIFACTS.mkdir(exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    for name, synthetic_config in CASES.items():
        csv_path = DATA / f"synthetic_mpm_{name}.csv"
        write_synthetic_mpm(csv_path, synthetic_config)
        cells = pd.read_csv(csv_path)
        config = MPMConfig(random_state=42, n_estimators=280)
        metrics = evaluate_candidates(cells, config=config, model_names=("random_forest",))
        model = fit_mpm(cells, config=config, model_name="random_forest")
        predictions = predict_mpm(model, cells)
        summary = evaluate_high_priority(cells, predictions, config=config)
        metrics.to_csv(ARTIFACTS / f"{name}_candidate_metrics.csv", index=False)
        predictions.to_csv(ARTIFACTS / f"{name}_predictions.csv", index=False)
        pd.Series(summary, name="value").to_csv(ARTIFACTS / f"{name}_high_priority_summary.csv")
        map_data = cells.merge(predictions[["H3_ADDRESS", "MPM_BIN", "MPM_PROB"]], on="H3_ADDRESS", validate="one_to_one")
        plot_targets(map_data, IMAGES / f"synthetic_{name}_targets.png", name)
        print(f"\n{name}\n{metrics.round(3).to_string(index=False)}\n{pd.Series(summary).round(3)}")


if __name__ == "__main__":
    main()
