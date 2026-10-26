"""Temporal label availability and raw→snapshot→baseline→policy regression tests.

Synthetic transactions validate mechanics only; they are not capstone results.
"""
from copy import deepcopy
from importlib import import_module
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from utils import load_config, validate_final_test_predictions

cleaning = import_module("02_cleaning_features")
baseline = import_module("03_baseline_models")
decision = import_module("05_decision_scenario")


@pytest.fixture
def config():
    cfg = deepcopy(load_config())
    cfg["validation"]["n_bootstrap"] = 5
    return cfg


@pytest.fixture
def raw(config):
    rows = []

    def purchase(customer, date, price):
        rows.append({"Invoice": str(len(rows)), "StockCode": str(customer % 4),
                     "Description": "synthetic", "Quantity": customer + 1,
                     "InvoiceDate": pd.Timestamp(date), "Price": float(price),
                     "Customer ID": float(customer), "Country": "United Kingdom"})

    for customer in range(1, 17):
        purchase(customer, "2009-12-01", 2)
    for j, origin in enumerate(cleaning.snapshot_schedule(config)["decision_date"]):
        for customer in range(1, 17):
            if (customer + j) % 2 == 0:
                purchase(customer, origin + pd.Timedelta(days=customer + 10), 3 + j)
    purchase(1, "2011-12-09 12:00:00", 2)  # coverage, outside final outcome
    return pd.DataFrame(rows)


def test_temporal_schedule_has_earlier_origins_and_fully_observed_labels(config):
    schedule = cleaning.snapshot_schedule(config)
    for role in ("validation", "final_test"):
        origin = schedule.loc[schedule["split"] == role, "decision_date"].iloc[0]
        previous = schedule[schedule["decision_date"] < origin]
        assert len(previous) >= 2
        assert (previous["decision_date"] < origin).all()
        assert (previous["outcome_end_exclusive"] < origin).all()
    assert schedule["outcome_end_exclusive"].iloc[-1] == pd.Timestamp("2011-12-09")
    assert ((schedule["outcome_end_exclusive"] - schedule["decision_date"]).dt.days == 90).all()


def test_rejects_training_labels_unavailable_at_validation(config):
    config["validation"]["train_decision_dates"][-1] = "2011-04-01"
    with pytest.raises(ValueError, match="outcomes must finish"):
        cleaning.snapshot_schedule(config)


def test_rejects_incomplete_final_outcome(raw, config):
    raw.loc[raw.index[-1], "InvoiceDate"] = pd.Timestamp("2011-12-08 23:59:59")
    with pytest.raises(ValueError, match="does not cover"):
        cleaning.build_temporal_snapshots(raw, config)


def test_snapshot_features_ignore_future_purchases(raw, config):
    original = cleaning.build_temporal_snapshots(raw, config)
    extra = raw.iloc[[0]].copy()
    extra["Invoice"] = "future-only"
    extra["InvoiceDate"] = pd.Timestamp("2011-09-11")
    extra["Price"] = 1e6
    changed = cleaning.build_temporal_snapshots(pd.concat([raw, extra], ignore_index=True), config)
    pd.testing.assert_frame_equal(original[baseline.FEATURE_COLS], changed[baseline.FEATURE_COLS])
    assert original["future_gross_sales"].sum() != changed["future_gross_sales"].sum()


def test_exact_duplicate_audit_counts_without_removal(raw):
    repeated = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    report = cleaning.audit_duplicates(repeated)
    assert report["exact_duplicate_rows"].iloc[0] == 1
    assert report["duplicate_policy"].iloc[0] == "retain"
    assert len(repeated) == len(raw) + 1


@pytest.mark.parametrize("source", ["baseline", "advanced"])
def test_final_predictions_never_fit_final_outcomes(raw, config, source):
    snapshots = cleaning.build_temporal_snapshots(raw, config)
    models = (baseline.baseline_models(config) if source == "baseline" else
              (GradientBoostingClassifier(random_state=42), GradientBoostingRegressor(random_state=42)))
    scores, _, clf, _ = baseline.evaluate_temporal_models(snapshots, config, *models, source=source)
    changed = snapshots.copy()
    is_final = changed["split"].eq("final_test")
    changed.loc[is_final, "future_gross_sales"] += 1e6
    changed.loc[is_final, "repeat_purchase"] = 1 - changed.loc[is_final, "repeat_purchase"]
    other, _, _, _ = baseline.evaluate_temporal_models(changed, config, *models, source=source)
    pd.testing.assert_frame_equal(scores[["predicted_future_value", "predicted_repeat_purchase_proba"]],
                                  other[["predicted_future_value", "predicted_repeat_purchase_proba"]])
    validate_final_test_predictions(scores, config)
    assert scores["split"].eq("final_test").all()
    assert len(scores) == 16
    fit_rows = snapshots[~snapshots["split"].eq("final_test")]
    if source == "baseline":
        np.testing.assert_allclose(clf.named_steps["standardscaler"].mean_, fit_rows[baseline.FEATURE_COLS].mean())
    assert set(scores["Customer ID"]) == set(fit_rows["Customer ID"])  # earlier repeated customers are valid


def test_baseline_only_pipeline_reaches_policy_without_gbm(raw, config, tmp_path, monkeypatch):
    snapshots = cleaning.build_temporal_snapshots(raw, config)
    snapshots.to_csv(tmp_path / "customer_snapshots.csv", index=False)
    for module in (baseline, decision):
        monkeypatch.setattr(module, "load_config", lambda: config)
        monkeypatch.setattr(module, "processed_path", lambda name, cfg: tmp_path / name)
        monkeypatch.setattr(module, "outputs_path", lambda name: tmp_path / name)
    monkeypatch.setattr(baseline, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(baseline, "REGISTRY_PATH", tmp_path / "temporal_registry.json")
    monkeypatch.setattr(decision, "plot_tornado", lambda table: None)
    baseline.main()
    decision.main()
    scores = pd.read_csv(tmp_path / "customer_test_scores.csv")
    table = pd.read_csv(tmp_path / "scenario_table.csv")
    validate_final_test_predictions(scores, config)
    assert scores["prediction_source"].eq("baseline").all()
    assert not (tmp_path / "customer_test_predictions_advanced.csv").exists()
    assert table["scenario"].nunique() == 3
    assert set(table["policy"]) == {"non_targeted", "rfm", "proposed"}
    assert (scores["baseline_future_revenue"] >= 0).all()
    # All original assumptions are unprofitable with unchanged unit COGS.
    assert table["customers_targeted"].eq(0).all()
    assert table["simulated_net_incremental_margin"].eq(0).all()
