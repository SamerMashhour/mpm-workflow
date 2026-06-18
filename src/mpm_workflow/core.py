"""Core functions for a tabular mineral prospectivity mapping workflow.

The implementation is a reusable refactor of the workshop notebook workflow:
1. validate a H3-cell tabular dataset;
2. select positive examples and an equally sized random background sample;
3. fit a preprocessing + classifier pipeline;
4. predict prospectivity over all cells; and
5. convert scores to budget-aware target classes.

The default numerical transformation deliberately preserves signed geophysical
variables. The notebook's legacy behaviour that clips all negative values to
zero remains available only through ``numeric_transform='legacy_clipped_log'``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, QuantileTransformer, StandardScaler
from sklearn.svm import SVC

ModelName = Literal["random_forest", "svm_rbf", "adaboost", "mlp"]
NumericTransform = Literal["quantile", "legacy_clipped_log"]

DEFAULT_META_COLUMNS = ("H3_ADDRESS", "H3_RESOLUTION", "LATITUDE", "LONGITUDE")
DEFAULT_CATEGORICAL_COLUMNS = (
    "GEOLOGY_LITHOLOGY_MAJORITY",
    "GEOLOGY_LITHOLOGY_MINORITY",
)


@dataclass(frozen=True)
class MPMConfig:
    """Configuration for the tabular mineral prospectivity workflow."""

    target_col: str = "TRAINING_CLASS"
    meta_cols: tuple[str, ...] = DEFAULT_META_COLUMNS
    categorical_cols: tuple[str, ...] = DEFAULT_CATEGORICAL_COLUMNS
    positive_value: int = 1
    random_state: int = 42
    numeric_transform: NumericTransform = "quantile"
    quantile_n_quantiles: int = 1000
    n_estimators: int = 300


@dataclass
class MPMModel:
    """A fitted MPM pipeline plus the exact raw columns it expects."""

    pipeline: Pipeline
    config: MPMConfig
    feature_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    model_name: str

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """Return positive-class prospectivity scores for every supplied cell."""
        _validate_required_columns(data, self.feature_columns)
        probabilities = self.pipeline.predict_proba(data[self.feature_columns])
        return probabilities[:, 1]


def _validate_required_columns(data: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


def _legacy_clip_log1p(values: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
    """Notebook-compatible transformation that discards all negative values.

    This exists only to reproduce the original notebook. It should not be the
    normal option for signed gravity or magnetic variables.
    """
    if isinstance(values, pd.DataFrame):
        clipped = values.copy()
        clipped[clipped < 0] = 0
        return np.log1p(clipped)
    array = np.asarray(values, dtype=float).copy()
    array[array < 0] = 0
    return np.log1p(array)


def infer_feature_columns(
    data: pd.DataFrame,
    config: MPMConfig,
) -> tuple[list[str], list[str], list[str]]:
    """Infer model features while excluding labels, IDs and all TRAINING_* leakage fields."""
    _validate_required_columns(data, [config.target_col, *config.meta_cols])

    training_cols = [column for column in data.columns if column.startswith("TRAINING_")]
    excluded = set(config.meta_cols).union(training_cols).union({config.target_col})
    feature_columns = [column for column in data.columns if column not in excluded]

    configured_categories = [
        column for column in config.categorical_cols if column in feature_columns
    ]
    inferred_categories = [
        column
        for column in feature_columns
        if pd.api.types.is_object_dtype(data[column])
        or isinstance(data[column].dtype, pd.CategoricalDtype)
    ]
    categorical_columns = list(dict.fromkeys(configured_categories + inferred_categories))
    numeric_columns = [column for column in feature_columns if column not in categorical_columns]

    if not numeric_columns and not categorical_columns:
        raise ValueError("No model features remain after leakage and metadata columns are removed.")

    return feature_columns, numeric_columns, categorical_columns


def build_balanced_training_set(data: pd.DataFrame, config: MPMConfig) -> pd.DataFrame:
    """Return all positives plus an equal-sized random sample of non-positive cells.

    The notebook treats every cell whose target is not ``positive_value`` as an
    eligible background candidate. This mirrors that behaviour explicitly. It
    is a positive-versus-background workflow, not proof that every sampled
    background cell is mineralised-free.
    """
    _validate_required_columns(data, [config.target_col])
    positives = data.loc[data[config.target_col] == config.positive_value].copy()
    background_pool = data.loc[data[config.target_col] != config.positive_value].copy()

    if positives.empty:
        raise ValueError(f"No positive rows found where {config.target_col} == {config.positive_value}.")
    if len(background_pool) < len(positives):
        raise ValueError(
            "The background pool is smaller than the positive set; cannot construct a balanced sample."
        )

    background = background_pool.sample(n=len(positives), random_state=config.random_state)
    balanced = pd.concat([positives, background], axis=0)
    return balanced.sample(frac=1.0, random_state=config.random_state).reset_index(drop=True)


def make_preprocessor(
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    config: MPMConfig,
) -> ColumnTransformer:
    """Create a feature-preserving preprocessing stage for numerical and categorical inputs."""
    numeric_steps: list[tuple[str, object]] = [
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
    ]
    if config.numeric_transform == "legacy_clipped_log":
        numeric_steps.append(
            (
                "legacy_clip_log",
                FunctionTransformer(_legacy_clip_log1p, validate=False, feature_names_out="one-to-one"),
            )
        )
    elif config.numeric_transform != "quantile":
        raise ValueError(f"Unsupported numeric_transform: {config.numeric_transform}")

    numeric_steps.extend(
        [
            (
                "quantile_normalize",
                QuantileTransformer(
                    n_quantiles=config.quantile_n_quantiles,
                    output_distribution="normal",
                    random_state=config.random_state,
                ),
            ),
            ("scale", StandardScaler()),
        ]
    )
    numeric_pipeline = Pipeline(numeric_steps)

    transformers: list[tuple[str, object, Sequence[str]]] = []
    if numeric_columns:
        transformers.append(("numeric", numeric_pipeline, list(numeric_columns)))
    if categorical_columns:
        categorical_pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, list(categorical_columns)))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _build_classifier(model_name: ModelName, config: MPMConfig):
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=config.n_estimators,
            random_state=config.random_state,
            n_jobs=-1,
        )
    if model_name == "svm_rbf":
        return SVC(kernel="rbf", probability=True, C=1.0, gamma="scale", random_state=config.random_state)
    if model_name == "adaboost":
        return AdaBoostClassifier(n_estimators=config.n_estimators, random_state=config.random_state)
    if model_name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(64,),
            max_iter=400,
            n_iter_no_change=25,
            random_state=config.random_state,
            early_stopping=True,
        )
    raise ValueError(f"Unsupported model_name: {model_name}")


def build_pipeline(
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    config: MPMConfig,
    model_name: ModelName,
) -> Pipeline:
    """Build an unfitted preprocessing plus classifier pipeline."""
    return Pipeline(
        [
            ("preprocess", make_preprocessor(numeric_columns, categorical_columns, config)),
            ("classifier", _build_classifier(model_name, config)),
        ]
    )


def _metrics_from_predictions(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def evaluate_candidates(
    data: pd.DataFrame,
    config: MPMConfig = MPMConfig(),
    model_names: Sequence[ModelName] = ("svm_rbf", "random_forest", "adaboost", "mlp"),
    test_size: float = 0.30,
) -> pd.DataFrame:
    """Compare candidate models on the notebook-style balanced train/test split.

    This is a fast benchmark, not a spatially independent validation design.
    Spatial cross-validation should be added for scientific model assessment.
    """
    balanced = build_balanced_training_set(data, config)
    feature_columns, numeric_columns, categorical_columns = infer_feature_columns(balanced, config)
    X = balanced[feature_columns]
    y = balanced[config.target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=config.random_state,
    )

    rows: list[dict[str, float | str]] = []
    for model_name in model_names:
        pipeline = build_pipeline(numeric_columns, categorical_columns, config, model_name)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        rows.append({"model": model_name, **_metrics_from_predictions(y_test, predictions, probabilities)})

    return pd.DataFrame(rows).sort_values(["f1", "roc_auc"], ascending=False).reset_index(drop=True)


def fit_mpm(
    data: pd.DataFrame,
    config: MPMConfig = MPMConfig(),
    model_name: ModelName = "random_forest",
) -> MPMModel:
    """Fit a prospectivity model on a balanced positive/background training set."""
    balanced = build_balanced_training_set(data, config)
    feature_columns, numeric_columns, categorical_columns = infer_feature_columns(balanced, config)
    pipeline = build_pipeline(numeric_columns, categorical_columns, config, model_name)
    pipeline.fit(balanced[feature_columns], balanced[config.target_col].astype(int))
    return MPMModel(
        pipeline=pipeline,
        config=config,
        feature_columns=feature_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        model_name=model_name,
    )


def _validate_quantiles(low_quantile: float, high_quantile: float) -> None:
    if not 0 <= low_quantile < high_quantile <= 1:
        raise ValueError("Quantiles must satisfy 0 <= low_quantile < high_quantile <= 1.")


def add_target_bins(
    predictions: pd.DataFrame,
    score_col: str = "MPM_PROB",
    bin_col: str = "MPM_BIN",
    low_quantile: float = 0.80,
    high_quantile: float = 0.90,
) -> pd.DataFrame:
    """Assign low/medium/high target classes by probability quantiles."""
    _validate_quantiles(low_quantile, high_quantile)
    if score_col not in predictions:
        raise ValueError(f"{score_col!r} is not present in prediction table.")
    result = predictions.copy()

    # Quantile cut-offs can be identical when an estimator returns tied scores
    # (for example a small random forest returning many 0.0/1.0 values). Rank
    # scores deterministically first, which preserves score ordering while still
    # producing the requested approximate low/medium/high coverage.
    percentile_rank = result[score_col].rank(method="first", pct=True)
    result[bin_col] = pd.Categorical(
        np.select(
            [percentile_rank <= low_quantile, percentile_rank <= high_quantile],
            ["low", "medium"],
            default="high",
        ),
        categories=["low", "medium", "high"],
        ordered=True,
    )
    return result


def predict_mpm(
    model: MPMModel,
    data: pd.DataFrame,
    low_quantile: float = 0.80,
    high_quantile: float = 0.90,
    probability_col: str = "MPM_PROB",
) -> pd.DataFrame:
    """Predict all cells and attach H3 metadata plus quantile-based target bins."""
    _validate_required_columns(data, [*model.config.meta_cols, *model.feature_columns])
    result = data.loc[:, list(model.config.meta_cols)].copy()
    probabilities = model.predict_proba(data)
    result[probability_col] = probabilities
    result["MPM_CLASS_0_1"] = (probabilities >= 0.5).astype(int)
    return add_target_bins(
        result,
        score_col=probability_col,
        bin_col="MPM_BIN",
        low_quantile=low_quantile,
        high_quantile=high_quantile,
    )


def evaluate_high_priority(
    data: pd.DataFrame,
    predictions: pd.DataFrame,
    config: MPMConfig = MPMConfig(),
    id_col: str = "H3_ADDRESS",
    bin_col: str = "MPM_BIN",
    high_label: str = "high",
) -> dict[str, float]:
    """Summarise high-priority coverage and apparent labelled-cell precision.

    ``apparent_precision`` assumes target=0 cells are negatives. In a
    positive-unlabelled exploration dataset this is a conservative proxy, not
    a literal estimate of the probability that a cell hosts a deposit.
    """
    _validate_required_columns(data, [id_col, config.target_col])
    _validate_required_columns(predictions, [id_col, bin_col])
    joined = data[[id_col, config.target_col]].merge(
        predictions[[id_col, bin_col]], on=id_col, how="inner", validate="one_to_one"
    )
    high = joined[bin_col].astype(str) == high_label
    positives = joined[config.target_col] == config.positive_value
    if positives.sum() == 0:
        raise ValueError("Cannot calculate high-priority recall: no positive labels found.")
    high_fraction = float(high.mean())
    true_positive_high = int((high & positives).sum())
    recall_high = true_positive_high / int(positives.sum())
    apparent_precision = float(true_positive_high / max(int(high.sum()), 1))
    return {
        "high_fraction": high_fraction,
        "spatial_selectivity": 1.0 - high_fraction,
        "recall_high": float(recall_high),
        "apparent_precision": apparent_precision,
    }


def save_model(model: MPMModel, path: str | Path) -> Path:
    """Serialize a fitted MPM model with joblib."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output)
    return output


def load_model(path: str | Path) -> MPMModel:
    """Load a previously serialized MPM model."""
    loaded = joblib.load(Path(path))
    if not isinstance(loaded, MPMModel):
        raise TypeError("The supplied joblib file does not contain an MPMModel.")
    return loaded
