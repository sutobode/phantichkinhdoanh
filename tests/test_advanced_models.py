"""Targeted Phase 3 leakage, selection, calibration and robustness tests."""
from copy import deepcopy
from importlib import import_module
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from utils import load_config

cleaning = import_module("02_cleaning_features")
baseline = import_module("03_baseline_models")
advanced = import_module("04_advanced_models")


@pytest.fixture
def config():
    return deepcopy(load_config())


@pytest.fixture
def raw(config):
    rows = []

    def purchase(customer, date, price):
        rows.append({
            "Invoice": str(len(rows)), "StockCode": str(customer % 5),
            "Description": "synthetic", "Quantity": customer % 4 + 1,
            "InvoiceDate": pd.Timestamp(date), "Price": float(price),
            "Customer ID": float(customer), "Country": "United Kingdom",
        })

    for customer in range(1, 25):
        purchase(customer, "2009-12-01", 2 + customer / 10)
    for j, origin in enumerate(advanced.rolling_origin_schedule(config)["decision_date"]):
        for customer in range(1, 25):
            if (customer + j) % 3:
                purchase(customer, origin + pd.Timedelta(days=customer % 40 + 2), 3 + j)
    purchase(1, "2011-12-09 12:00:00", 2)
    return pd.DataFrame(rows)


def test_advanced_final_predictions_do_not_fit_final_labels(raw, config):
    snapshots = cleaning.build_temporal_snapshots(raw, config)
    _, predictions, _ = advanced.evaluate_model_comparison(snapshots, config)
    changed = snapshots.copy()
    final = changed["split"].eq("final_test")
    changed.loc[final, "repeat_purchase"] = 1 - changed.loc[final, "repeat_purchase"]
    changed.loc[final, "future_gross_sales"] += 1e6
    _, other, _ = advanced.evaluate_model_comparison(changed, config)
    for family in ("baseline", "advanced"):
        np.testing.assert_allclose(
            predictions[("final_test", family)][
                ["predicted_repeat_purchase_proba", "predicted_future_value"]
            ],
            other[("final_test", family)][
                ["predicted_repeat_purchase_proba", "predicted_future_value"]
            ],
        )


def test_model_selection_uses_validation_not_final_test(config):
    rows = [
        {"split": split, "model": model, "pr_auc": pr, "brier_score": brier,
         "mae": mae, "p90_absolute_error": p90}
        for split, model, pr, brier, mae, p90 in (
            ("validation", "baseline", .70, .20, 100, 200),
            ("validation", "advanced", .72, .20, 90, 190),
            ("final_test", "baseline", .99, .01, 1, 2),
            ("final_test", "advanced", .01, .99, 999, 999),
        )
    ]
    selected = advanced.select_models(pd.DataFrame(rows), config)
    changed = pd.DataFrame(rows)
    changed.loc[changed["split"].eq("final_test"), ["pr_auc", "brier_score", "mae", "p90_absolute_error"]] = 0
    pd.testing.assert_frame_equal(selected, advanced.select_models(changed, config))
    assert set(selected["selected_model"]) == {"gradient_boosting_classifier", "gradient_boosting_regressor"}


def test_calibration_excludes_evaluation_labels(raw, config):
    snapshots = cleaning.build_temporal_snapshots(raw, config)
    train, validation, final = cleaning.temporal_partitions(snapshots, config)
    proba, _, _ = advanced.fit_sigmoid_calibrator(
        baseline.baseline_models(config)[0], train, validation, final
    )
    changed = final.copy()
    changed["repeat_purchase"] = 1 - changed["repeat_purchase"]
    other, _, _ = advanced.fit_sigmoid_calibrator(
        baseline.baseline_models(config)[0], train, validation, changed
    )
    np.testing.assert_allclose(proba, other)
    with pytest.raises(ValueError, match="Calibration labels"):
        advanced.fit_sigmoid_calibrator(
            baseline.baseline_models(config)[0], train, final, validation
        )


def test_rolling_schedule_labels_and_features_are_temporal(raw, config):
    schedule = advanced.rolling_origin_schedule(config)
    for origin in schedule["decision_date"].iloc[1:]:
        prior = schedule[
            (schedule["decision_date"] < origin)
            & (schedule["outcome_end_exclusive"] < origin)
        ]
        assert not prior.empty
    snapshots = advanced.build_rolling_snapshots(raw, config)
    origin = pd.Timestamp("2011-06-01")
    training = advanced.eligible_rolling_training(snapshots, origin)
    assert pd.to_datetime(training["outcome_end_exclusive"]).lt(origin).all()
    original = snapshots[pd.to_datetime(snapshots["decision_date"]).eq(origin)].reset_index(drop=True)
    future = raw.iloc[[0]].copy()
    future["Invoice"] = "future-only"
    future["InvoiceDate"] = origin + pd.Timedelta(days=3)
    future["Price"] = 1e6
    changed = advanced.build_rolling_snapshots(pd.concat([raw, future], ignore_index=True), config)
    changed = changed[pd.to_datetime(changed["decision_date"]).eq(origin)].reset_index(drop=True)
    pd.testing.assert_frame_equal(original[baseline.FEATURE_COLS], changed[baseline.FEATURE_COLS])


def test_rolling_output_schema_and_sampling_are_stable(raw, config):
    snapshots = advanced.build_rolling_snapshots(raw, config)
    metrics = advanced.evaluate_rolling_origins(snapshots, config)
    assert list(metrics.columns) == advanced.ROLLING_COLUMNS
    assert metrics["decision_date"].nunique() == 5
    assert set(metrics["task"]) == {"classification", "regression"}
    sample_a = advanced.deterministic_sample(snapshots, 19, config["seed"])
    sample_b = advanced.deterministic_sample(snapshots.sample(frac=1), 19, config["seed"])
    assert sample_a["Customer ID"].tolist() == sample_b["Customer ID"].tolist()
