# %% [markdown]
# # MPM Workflow: a short recording demo
#
# Run this file cell-by-cell in VS Code using the Python/Jupyter extension.
# It is designed for a 20-30 second screen recording: input map -> model run -> target map.
#
# One-time setup from the repository root:
#
#     pip install -e ".[viz]"

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
    make_synthetic_mpm,
    predict_mpm,
)

ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)

# A compact case keeps the demo quick while retaining a clear spatial pattern.
DEMO = SyntheticMPMConfig(
    scenario="structural_corridors",
    n_cells=6_000,
    positive_count=180,
    negative_training_count=180,
    random_state=20260702,
    label_noise=0.82,
    latitude_range=(43.5, 47.4),
    longitude_range=(-121.8, -112.3),
)

# %% [markdown]
# ## 1. Generate a synthetic geoscience data cube

# %%
start = perf_counter()
cells = make_synthetic_mpm(DEMO)
print(f"Generated {len(cells):,} synthetic cells in {perf_counter() - start:.1f} s")
print(f"Columns available to the workflow: {cells.shape[1]}")

# %% [markdown]
# ## 2. View known synthetic occurrences before modelling

# %%
occurrences = cells["TRAINING_MINERAL_OCCURRENCE"].eq(1)
deposits = cells["TRAINING_DEPOSIT"].eq(1)

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.scatter(
    cells["LONGITUDE"],
    cells["LATITUDE"],
    s=3,
    c="#d9d9d9",
    linewidths=0,
    label="Synthetic grid cells",
)
ax.scatter(
    cells.loc[occurrences, "LONGITUDE"],
    cells.loc[occurrences, "LATITUDE"],
    s=32,
    marker="x",
    c="#d95f02",
    linewidths=1.2,
    label="Known synthetic occurrence",
)
ax.scatter(
    cells.loc[deposits, "LONGITUDE"],
    cells.loc[deposits, "LATITUDE"],
    s=90,
    marker="*",
    c="#5e3c99",
    edgecolor="white",
    linewidths=0.5,
    label="Synthetic deposit",
)
ax.set(
    title="Input: synthetic structural-corridor MPM scenario",
    xlabel="Synthetic longitude",
    ylabel="Synthetic latitude",
)
ax.legend(loc="upper left", frameon=True)
ax.grid(alpha=0.2)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Train a Random Forest and predict prospectivity for every cell

# %%
config = MPMConfig(random_state=42, n_estimators=140, quantile_n_quantiles=250)

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
# ## 4. Show full-grid prospectivity and high-priority targets

# %%
results = cells.merge(
    predictions[["H3_ADDRESS", "MPM_PROB", "MPM_BIN"]],
    on="H3_ADDRESS",
    validate="one_to_one",
)
high_targets = results["MPM_BIN"].eq("high")
result_deposits = results["TRAINING_DEPOSIT"].eq(1)

fig, ax = plt.subplots(figsize=(10, 5.5))
points = ax.scatter(
    results["LONGITUDE"],
    results["LATITUDE"],
    c=results["MPM_PROB"],
    s=5,
    cmap="viridis",
    linewidths=0,
    rasterized=True,
)
ax.scatter(
    results.loc[high_targets, "LONGITUDE"],
    results.loc[high_targets, "LATITUDE"],
    s=11,
    facecolors="none",
    edgecolors="#ffcc33",
    linewidths=0.45,
    label="High-priority target",
)
ax.scatter(
    results.loc[result_deposits, "LONGITUDE"],
    results.loc[result_deposits, "LATITUDE"],
    s=88,
    marker="*",
    c="white",
    edgecolor="#202020",
    linewidths=0.55,
    label="Synthetic deposit",
)
colourbar = fig.colorbar(points, ax=ax, pad=0.02)
colourbar.set_label("Random Forest prospectivity score")
ax.set(
    title="Output: full-grid prospectivity and ranked targets",
    xlabel="Synthetic longitude",
    ylabel="Synthetic latitude",
)
ax.legend(loc="upper left", frameon=True)
ax.grid(alpha=0.2)
fig.tight_layout()

output_path = ARTIFACTS / "recording_demo_prospectivity_map.png"
fig.savefig(output_path, dpi=200, bbox_inches="tight")
plt.show()

print(f"Saved final map: {output_path}")
print("Synthetic demonstration only. Random holdout is a benchmark, not spatial validation.")
