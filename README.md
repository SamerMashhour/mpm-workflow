# MPM Workflow

A reusable Python package for a tabular mineral prospectivity mapping (MPM) workflow. It refactors the practical components of an introductory Orange/Jupyter workflow into testable Python code: positive-versus-background sampling, preprocessing, candidate-model evaluation, Random Forest fitting, full-grid prediction, ranked target classes, and diagnostic plots.

## What it does

- Reads a row-per-cell geospatial data cube.
- Excludes IDs, coordinates, labels, and every `TRAINING_*` field from predictors to prevent training-label leakage.
- Balances known positive cells against a random sample of background cells.
- Handles categorical geology and numerical covariates, including missing data.
- Compares Random Forest, RBF-SVM, AdaBoost, and MLP candidates using a stratified random holdout.
- Trains a selected classifier, scores all cells, and creates low/medium/high ranked target bins.
- Reports high-priority target coverage, apparent labelled-cell precision, and spatial selectivity.
- Includes a generator for two openly shareable synthetic MPM input scenarios.

## Numerical preprocessing

The original notebook clipped every negative numerical value to zero before applying `log1p`. That would remove the sign and much of the information in variables such as Bouguer gravity and magnetic residual fields. The default package transformation uses median imputation, quantile normalization, and scaling **without clipping signed data**.

Use `numeric_transform="legacy_clipped_log"` only when you need a deliberate, notebook-compatible reproduction. It is not recommended for signed geophysical variables.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Quick start with the synthetic examples

The repository includes a deterministic generator for two 30,000-cell, 33-column input scenarios, created from scratch and containing no workshop or third-party input data:

- `belt_cover`: broad arcuate belts, volcanic and intrusive complexes, and cover basins.
- `structural_corridors`: intersecting structural corridors and compact clusters around relay zones.

Generate them locally:

```bash
mpm-workflow generate-synthetic \
  --scenario belt_cover \
  --output data/generated_belt_cover.csv

mpm-workflow generate-synthetic \
  --scenario structural_corridors \
  --output data/generated_structural_corridors.csv
```

Then run the complete synthetic workflow:

```bash
python examples/generate_and_run_synthetic_cases.py
```

See [`docs/SYNTHETIC_DATA.md`](docs/SYNTHETIC_DATA.md) for limits and generation method.

## Command-line workflow

For a real, lawfully obtained data cube:

```bash
mpm-workflow train \
  --data data/data_mpm.csv \
  --model-output artifacts/mpm_model.joblib \
  --metrics-output artifacts/model_metrics.csv \
  --model random_forest

mpm-workflow predict \
  --data data/data_mpm.csv \
  --model artifacts/mpm_model.joblib \
  --output artifacts/mpm_predictions.csv \
  --low-quantile 0.80 \
  --high-quantile 0.90
```

## Python API

```python
import pandas as pd
from mpm_workflow import MPMConfig, evaluate_candidates, fit_mpm, predict_mpm

cells = pd.read_csv("data/generated_belt_cover.csv")
config = MPMConfig(random_state=42, n_estimators=300)

metrics = evaluate_candidates(cells, config)
model = fit_mpm(cells, config, model_name="random_forest")
predictions = predict_mpm(model, cells, low_quantile=0.80, high_quantile=0.90)
```

## Validation and interpretation

`evaluate_candidates` uses a random stratified holdout as a quick model-comparison benchmark. It is not spatially independent validation.

Before interpreting outputs from real data, apply spatial-block cross-validation, assess label quality, examine the geological plausibility of covariates, and evaluate predicted targets against independent geological evidence. High-priority target coverage and apparent precision are workflow diagnostics, not estimates of discovery probability.

## Repository structure

```text
src/mpm_workflow/     Package code
examples/             Reproducible scripts, including synthetic-data generation
data/                 Generated synthetic inputs and locally held data; ignored by Git
artifacts/            Locally generated models, predictions, metrics, and figures; ignored by Git
docs/                 Synthetic-data method and usage notes
tests/                Automated package tests
```

Generated CSV files, models, predictions, and local artifacts are excluded from Git by default.

## How to cite

If you use MPM Workflow in research, teaching, or a publication, please cite the software version used. Citation metadata are available through GitHub's **Cite this repository** button and in [`CITATION.cff`](CITATION.cff).

Example citation:

> Maghdour-Mashhour, S. R. (2026). *MPM Workflow* (Version 0.1.0) [Computer software]. https://github.com/SamerMashhour/mpm-workflow

## License

The package code and the generated synthetic examples are released under the MIT License. Do not assume the same license applies to independently acquired data or teaching materials.
