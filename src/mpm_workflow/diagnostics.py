"""Diagnostic summaries and plots for MPM inputs and predictions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA

from .core import MPMModel


def numeric_data_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Return count, missingness and basic statistics for numeric columns."""
    numeric = data.select_dtypes(include=np.number)
    summary = numeric.agg(["count", "mean", "std", "min", "median", "max"]).T
    summary["missing"] = numeric.isna().sum()
    summary["missing_fraction"] = numeric.isna().mean()
    return summary.sort_index()


def pca_projection(model: MPMModel, data: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    """Project a fitted model's transformed features into principal components."""
    transformed = model.pipeline.named_steps["preprocess"].transform(data[model.feature_columns])
    scores = PCA(n_components=n_components, random_state=model.config.random_state).fit_transform(transformed)
    result = data.loc[:, list(model.config.meta_cols)].copy()
    for index in range(n_components):
        result[f"PC{index + 1}"] = scores[:, index]
    return result


def transformed_feature_importance(model: MPMModel) -> pd.DataFrame:
    """Return classifier importances in transformed feature space when available."""
    classifier = model.pipeline.named_steps["classifier"]
    if not hasattr(classifier, "feature_importances_"):
        raise TypeError("The selected classifier does not provide feature_importances_.")
    names = model.pipeline.named_steps["preprocess"].get_feature_names_out()
    return pd.DataFrame({"feature": names, "importance": classifier.feature_importances_}).sort_values("importance", ascending=False).reset_index(drop=True)


def raw_permutation_importance(model: MPMModel, data: pd.DataFrame, n_repeats: int = 10) -> pd.DataFrame:
    """Estimate raw-feature importance with permutation on labelled rows."""
    labelled = data.dropna(subset=[model.config.target_col])
    result = permutation_importance(
        model.pipeline,
        labelled[model.feature_columns],
        labelled[model.config.target_col].astype(int),
        n_repeats=n_repeats,
        random_state=model.config.random_state,
        scoring="f1",
    )
    return pd.DataFrame({"feature": model.feature_columns, "importance_mean": result.importances_mean, "importance_std": result.importances_std}).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def plot_prospectivity_map(predictions: pd.DataFrame, output: str | Path, title: str = "MPM prospectivity bins") -> Path:
    """Save a simple longitude-latitude prospectivity scatter plot."""
    import matplotlib.pyplot as plt

    required = {"LONGITUDE", "LATITUDE", "MPM_BIN"}
    if not required.issubset(predictions.columns):
        raise ValueError(f"Predictions must contain {sorted(required)}.")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for label in ("low", "medium", "high"):
        selected = predictions["MPM_BIN"].astype(str).eq(label)
        ax.scatter(predictions.loc[selected, "LONGITUDE"], predictions.loc[selected, "LATITUDE"], s=5, label=label, linewidths=0)
    ax.set(title=title, xlabel="Longitude", ylabel="Latitude")
    ax.legend(title="Target class")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path
