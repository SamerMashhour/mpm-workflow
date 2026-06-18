import pandas as pd

from mpm_workflow import MPMConfig, SyntheticMPMConfig, evaluate_candidates, fit_mpm, make_synthetic_mpm, predict_mpm


def test_synthetic_schema_and_model_run():
    cells = make_synthetic_mpm(
        SyntheticMPMConfig(
            n_cells=400,
            positive_count=30,
            negative_training_count=30,
            random_state=7,
            scenario="belt_cover",
        )
    )
    assert cells.shape == (400, 33)
    config = MPMConfig(random_state=7, n_estimators=20)
    metrics = evaluate_candidates(cells, config=config, model_names=("random_forest",))
    assert set(["f1", "roc_auc", "model"]).issubset(metrics.columns)
    model = fit_mpm(cells, config=config)
    predictions = predict_mpm(model, cells)
    assert len(predictions) == len(cells)
    assert set(predictions["MPM_BIN"].astype(str)) == {"low", "medium", "high"}
