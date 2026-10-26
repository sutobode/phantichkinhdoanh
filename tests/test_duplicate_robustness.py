"""Focused tests for the separated exact-duplicate sensitivity path."""
from copy import deepcopy
from importlib import import_module
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from utils import load_config

cleaning = import_module("02_cleaning_features")
robustness = import_module("07_duplicate_robustness")


@pytest.fixture
def config():
    cfg = deepcopy(load_config())
    cfg["validation"]["n_bootstrap"] = 3
    cfg["decision"]["random_policy_simulations"] = 10
    return cfg


@pytest.fixture
def raw(config):
    rows = []

    def purchase(customer, date, price):
        rows.append({
            "Invoice": str(len(rows)), "StockCode": str(customer % 4),
            "Description": "synthetic", "Quantity": customer % 3 + 1,
            "InvoiceDate": pd.Timestamp(date), "Price": float(price),
            "Customer ID": float(customer), "Country": "United Kingdom",
        })

    for customer in range(1, 17):
        purchase(customer, "2009-12-01", 2 + customer / 10)
    for j, origin in enumerate(cleaning.snapshot_schedule(config)["decision_date"]):
        for customer in range(1, 17):
            if (customer + j) % 2 == 0:
                purchase(customer, origin + pd.Timedelta(days=customer + 2), 3 + j)
    purchase(1, "2011-12-09 12:00:00", 2)
    rows.append(dict(rows[0]))
    return pd.DataFrame(rows)


def test_primary_retains_and_robustness_drops_only_exact_duplicates(raw, config):
    original = raw.copy(deep=True)
    retained = robustness.apply_duplicate_policy(raw, robustness.RETAIN)
    dropped = robustness.apply_duplicate_policy(raw, robustness.DROP)
    assert config["cleaning"]["exact_duplicate_policy"] == robustness.RETAIN
    assert len(retained) == len(raw)
    assert len(dropped) == len(raw) - 1
    assert not dropped.duplicated().any()
    pd.testing.assert_frame_equal(raw, original)


def test_robustness_labels_policies_and_preserves_temporal_cutoffs(raw, config):
    main_snapshots = cleaning.build_temporal_snapshots(raw, config)
    summary, business = robustness.build_duplicate_robustness(raw, main_snapshots, config)
    assert set(summary["duplicate_policy"]) == {robustness.RETAIN, robustness.DROP}
    assert set(business["duplicate_policy"]) == {robustness.RETAIN, robustness.DROP}
    assert summary["analysis_type"].eq(robustness.ANALYSIS_TYPE).all()
    assert business["analysis_type"].eq(robustness.ANALYSIS_TYPE).all()
    assert summary["final_test_decision_date"].eq(config["validation"]["final_test_decision_date"]).all()
    assert summary["outcome_horizon_days"].eq(config["validation"]["outcome_horizon_days"]).all()
    assert summary.loc[summary["duplicate_policy"].eq(robustness.RETAIN), "exact_duplicate_rows_removed"].iloc[0] == 0
    assert summary.loc[summary["duplicate_policy"].eq(robustness.DROP), "exact_duplicate_rows_removed"].iloc[0] == 1
    assert len(business) == 2 * len(config["decision"]["discount_response_scenarios"])
    assert set(business["scenario"]) == {"conservative", "moderate", "aggressive"}


def test_robustness_writer_cannot_overwrite_primary_outputs(tmp_path, monkeypatch):
    sentinel = tmp_path / "scenario_table.csv"
    sentinel.write_text("primary-output-sentinel", encoding="utf-8")
    monkeypatch.setattr(robustness, "outputs_path", lambda name: tmp_path / name)
    summary = pd.DataFrame([{"duplicate_policy": robustness.RETAIN}])
    business = pd.DataFrame([{"duplicate_policy": robustness.RETAIN}])
    robustness.write_robustness_outputs(summary, business)
    assert sentinel.read_text(encoding="utf-8") == "primary-output-sentinel"
    assert {path.name for path in tmp_path.glob("duplicate_robustness_*.csv")} == robustness.ROBUSTNESS_OUTPUTS
    assert robustness.ROBUSTNESS_OUTPUTS.isdisjoint(robustness.PRIMARY_OUTPUTS)
