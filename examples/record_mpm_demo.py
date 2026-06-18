# %% [markdown]
# # MPM Workflow: recording demo
#
# Run this file cell-by-cell in VS Code using the Python/Jupyter extension.
# It is designed for a short screen recording:
# install -> generate the repository input -> map occurrences -> train -> ranked targets.
#
# One-time setup from the repository root:
#
#     pip install -e ".[viz]"
#
# The input below is generated with the same structural-corridors configuration
# used in examples/generate_and_run_synthetic_cases.py. It writes the canonical
# local input file data/synthetic_mpm_structural_corridors.csv.

# %%
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import pandas as pd

from mpm_workflow import (
    MPMConfig,
    SyntheticMPMConfig,
    evaluate_candidates,
    evaluate_high_priority,
    fit_mpm,
    predict_mpm,
    write_synthetic_mpm,
)

DATA = Path("data")
ARTIFACTS = Path("artifacts")
DATA.mkdir(exist_ok=True)
ARTIFACTS.mkdir(exist_ok=True)

# This matches the official structural_corridors case in the repository.
DEMO = SyntheticMPMConfig(
    scenario="structural_corridors",
    random_state=20260702,
    latitude_range=(43.5, 47.4),
    longitude_range=(-121.8, -112.3),
    label_noise=0.82,
)
INPUT_PATH = DATA / "synthetic_mpm_structural_corridors.csv"

# %% [markdown]
# ## 1. Generate the repository synthetic input

# %%
start = perf_counter()
write_synthetic_mpm(INPUT_PATH, DEMO)
cells = pd.read_csv(INPUT_PATH)

print(f"Generated and loaded {len(cells):,} synthetic cells in {perf_counter() - start:.1f} s")
print(f"Columns available to the workflow: {cells.shape[1]}")
print(f"Canonical input written to: {INPUT_PATH}")

# %% [markdown]
# ## 2. View known synthetic occurrences before modelling

# %%
occurrences = cells["TRAINING_MINERAL_OCCURRENCE"].eq(1)
deposits = cells["TRAINING_DEPOSIT"].eq(1)

fig, ax = plt.subplots(figsize=(13.2, 6.8))
ax.scatter(
    cells["LONGITUDE"],
    cells["LATITUDE"],
    s=2.3,
    c="#d9d9d9",
    alpha=0.75,
    linewidths=0,
    label="Synthetic grid cells",
)
ax.scatter(
    cells.loc[occurrences, "LONGITUDE"],
    cells.loc[occurrences, "LATITUDE"],
    s=28,
    marker="x",
    c="#2ca02c",
    linewidths=1.15,
    label="Synthetic known mineral occurrence",
)
ax.scatter(
    cells.loc[deposits, "LONGITUDE"],
    cells.loc[deposits, "LATITUDE"],
    s=88,
    marker="*",
    c="#d62728",
    edgecolor="#d62728",
    linewidths=0.45,
    label="Synthetic known deposit",
)
ax.set(
    title="Input: synthetic structural-corridor MPM scenario",
    xlabel="Synthetic longitude",
    ylabel="Synthetic latitude",
)
ax.legend(loc="upper left", frameon=True)
ax.grid(alpha=0.22)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Train a Random Forest and predict prospectivity for every cell

# %%
# The reduced number of trees keeps the recording practical. It does not change
# the input scenario or the workflow steps demonstrated here.
config = MPMConfig(random_state=42, n_estimators=180, quantile_n_quantiles=500)

start = perf_counter()
metrics = evaluate_candidates(
    cells,
    config=config,
    model_names=("random_forest",),
)
model = fit_mpm(cells, config=config, model_name="random_forest")
predictions = predict_mpm(model, cells)
summary = evaluate_high_priority(cells, predictions, config=config)

print(f"Model trained and scored in {perf_counter() - start:.1f} s")
print(metrics[["model", "f1", "roc_auc"]].round(3).to_string(index=False))
print("\nHigh-priority target summary")
print(pd.Series(summary).round(3).to_string())

# %% [markdown]
# ## 4. Show ranked medium- and high-priority targets

# %%
results = cells.merge(
    predictions[["H3_ADDRESS", "MPM_PROB", "MPM_BIN"]],
    on="H3_ADDRESS",
    validate="one_to_one",
)
medium_targets = results["MPM_BIN"].eq("medium")
high_targets = results["MPM_BIN"].eq("high")
known_occurrences = results["TRAINING_MINERAL_OCCURRENCE"].eq(1)
known_deposits = results["TRAINING_DEPOSIT"].eq(1)

fig, ax = plt.subplots(figsize=(13.2, 6.8))
ax.scatter(
    results.loc[medium_targets, "LONGITUDE"],
    results.loc[medium_targets, "LATITUDE"],
    s=6,
    c="#3778bf",
    alpha=0.92,
    linewidths=0,
    label="Medium prospectivity target",
)
ax.scatter(
    results.loc[high_targets, "LONGITUDE"],
    results.loc[high_targets, "LATITUDE"],
    s=6,
    c="#f28e1c",
    alpha=0.95,
    linewidths=0,
    label="High prospectivity target",
)
ax.scatter(
    results.loc[known_occurrences, "LONGITUDE"],
    results.loc[known_occurrences, "LATITUDE"],
    s=30,
    marker="x",
    c="#2ca02c",
    linewidths=1.25,
    label="Synthetic known mineral occurrence",
)
ax.scatter(
    results.loc[known_deposits, "LONGITUDE"],
    results.loc[known_deposits, "LATITUDE"],
    s=95,
    marker="*",
    c="#d62728",
    edgecolor="#d62728",
    linewidths=0.5,
    label="Synthetic known deposit",
)
ax.set(
    title="Synthetic MPM case: structural corridors\nMedium = 80th–90th percentile; High = top 10%",
    xlabel="Synthetic longitude",
    ylabel="Synthetic latitude",
)
ax.legend(loc="upper left", frameon=True)
ax.grid(alpha=0.25)
fig.tight_layout()

output_path = ARTIFACTS / "structural_corridors_ranked_target_map.png"
fig.savefig(output_path, dpi=220, bbox_inches="tight")
plt.show()

print(f"Saved classified target map: {output_path}")
print("Synthetic demonstration only. Random holdout is a benchmark, not spatial validation.")
