"""Unit test cho logic tài chính ở Tuần 4: incremental margin formula,
budget-constrained selection, recommendation assignment.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest
import numpy as np
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from importlib import import_module

decision = import_module("05_decision_scenario")
from utils import load_config


def test_effective_response_rate_applies_segment_multiplier():
    df = pd.DataFrame({"rfm_segment": ["Champions", "At-risk", "Lost"]})
    multipliers = {"Champions": 0.5, "At-risk": 1.3, "Lost": 0.4, "Loyal": 1.0}
    result = decision.effective_response_rate(df, base_response_rate=0.20, multipliers=multipliers)
    assert result.tolist() == pytest.approx([0.10, 0.26, 0.08])


def test_effective_response_rate_raises_on_unknown_segment():
    df = pd.DataFrame({"rfm_segment": ["Mystery"]})
    with pytest.raises(ValueError):
        decision.effective_response_rate(df, 0.2, {"Champions": 0.5})


def test_incremental_margin_formula_known_values():
    # v=1000, eff_response_rate=0.20, discount=0.10, margin_pct=0.40
    # Promoted revenue=1080; COGS=720; baseline margin=400 => -40 before contact.
    v = pd.Series([1000.0])
    eff_rr = pd.Series([0.20])
    result = decision.incremental_margin(v, eff_rr, discount_pct=0.10, gross_margin_pct=0.40)
    assert result.iloc[0] == pytest.approx(-40.0)


def test_incremental_margin_negative_when_discount_exceeds_response_benefit():
    # eff_response_rate=0 (không phản hồi gì thêm), discount=0.10 -> margin phải ÂM
    # vì chỉ mất margin do giảm giá mà không có thêm doanh thu.
    v = pd.Series([1000.0])
    eff_rr = pd.Series([0.0])
    result = decision.incremental_margin(v, eff_rr, discount_pct=0.10, gross_margin_pct=0.40)
    assert result.iloc[0] < 0
    assert result.iloc[0] == pytest.approx(-100.0)  # Full discount cost, unchanged unit COGS.


def test_budget_constrained_selection_respects_cap_and_positivity():
    margin = pd.Series([100, -50, 80, 60, -10, 40, 5])
    # budget=3 -> cho phép chọn tối đa 3 khách hàng (cost=1/khách)
    selected = decision.budget_constrained_selection(margin, budget_total=3.0, cost_per_customer=1.0)
    assert selected.sum() == 3
    # phải chọn 3 margin dương LỚN NHẤT: 100, 80, 60 — không được chọn margin âm
    selected_values = margin[selected].tolist()
    assert sorted(selected_values, reverse=True) == [100, 80, 60]
    assert (margin[~selected] >= margin[selected].min()).sum() <= selected.sum()  # sanity: không chọn nhầm giá trị thấp hơn


def test_budget_constrained_selection_never_selects_negative_margin():
    # Budget dư thừa (đủ chọn tất cả) nhưng vẫn KHÔNG được chọn margin âm
    margin = pd.Series([100, -50, 80, -10])
    selected = decision.budget_constrained_selection(margin, budget_total=10.0, cost_per_customer=1.0)
    assert selected.sum() == 2  # chỉ có 2 giá trị dương
    assert (margin[selected] > 0).all()


def test_assign_recommendation_logic():
    df = pd.DataFrame({
        "rfm_segment": ["Champions", "At-risk", "Lost", "Loyal"],
        "baseline_future_revenue": [500.0, 300.0, 0.0, 200.0],
    })
    # Champions: margin<=0 nhưng value>0 -> retain
    # At-risk: margin>0 -> nurture
    # Lost: margin<=0, value=0 -> no-offer
    # Loyal: margin<=0 nhưng value>0 -> retain
    margin_reference = pd.Series([-10.0, 50.0, -5.0, -2.0])
    rec = decision.assign_recommendation(df, margin_reference)
    assert rec.tolist() == ["retain", "nurture", "no-offer", "retain"]


@pytest.fixture
def config():
    cfg = deepcopy(load_config())
    cfg["decision"]["random_policy_simulations"] = 40
    return cfg


@pytest.fixture
def final_scores(config):
    return pd.DataFrame({
        "Customer ID": [1, 2, 3, 4, 5, 6],
        "decision_date": config["validation"]["final_test_decision_date"],
        "outcome_end_exclusive": "2011-12-09",
        "training_outcome_end_exclusive": "2011-08-30",
        "split": "final_test", "evaluation_design": "expanding_temporal_snapshots",
        "prediction_source": "baseline",
        "rfm_segment": ["Champions", "Loyal", "At-risk", "Lost", "At-risk", "Loyal"],
        "rfm_score": [12, 9, 7, 4, 6, 8], "monetary": [800, 700, 500, 200, 300, 600],
        "predicted_future_value": [-232.79, 100, 800, 0, 600, 300],
    })


@pytest.mark.parametrize("column,value", [
    ("split", "train"), ("split", "deployment_demo"),
    ("decision_date", "2011-06-01"), ("training_outcome_end_exclusive", "2011-12-09"),
])
def test_policy_rejects_non_final_or_leaked_predictions(final_scores, config, column, value):
    final_scores.loc[0, column] = value
    with pytest.raises(ValueError, match="final temporal-test"):
        decision.build_scenario_table(final_scores, config)


def test_policy_rejects_legacy_scores(final_scores, config):
    with pytest.raises(ValueError, match="Final temporal-test"):
        decision.prepare_scores(final_scores.drop(columns="decision_date"), config)


def test_negative_predictions_cannot_reverse_recommendation(final_scores, config):
    prepared = decision.prepare_scores(final_scores, config)
    assert prepared[decision.VALUE_COL].ge(0).all()
    assert prepared[decision.VALUE_COL].iloc[0] == 0
    growth = decision.effective_response_rate(prepared, 0.2, config["decision"]["segment_response_multiplier"])
    net = decision.incremental_margin(prepared[decision.VALUE_COL], growth, 0.1, 0.4, 0.5)
    assert net.iloc[0] == -0.5
    assert decision.assign_recommendation(prepared, net).iloc[0] == "no-offer"
    # Direct API/function callers are protected too.
    assert decision.incremental_margin(pd.Series([-100.0]), pd.Series([0.0]), 0.1, 0.4, 0.5).iloc[0] == -0.5


def test_contact_cost_deducted_from_fixed_unit_cost_margin():
    # 1080 promoted revenue - 720 COGS - 400 baseline margin - 0.50 contact.
    result = decision.incremental_margin(pd.Series([1000.0]), pd.Series([0.2]), 0.1, 0.4, 0.5)
    assert result.iloc[0] == pytest.approx(-40.5)


@pytest.mark.parametrize("budget,capacity", [(750.0, 6), (1.0, 2), (0.0, 0)])
def test_three_scenarios_baselines_use_capacity_when_proposed_abstains(final_scores, config, budget, capacity):
    config["decision"]["budget_total"] = budget
    table = decision.build_scenario_table(final_scores, config)
    assert set(table["scenario"]) == {"conservative", "moderate", "aggressive"}
    assert len(table) == 12
    pd.testing.assert_frame_equal(table, decision.build_scenario_table(final_scores.sample(frac=1, random_state=7), config))
    assert table["campaign_capacity"].eq(capacity).all()
    proposed = table[table.policy == "proposed"]
    baselines = table[table.policy.isin(["non_targeted", "rfm"])]
    no_promotion = table[table.policy == "no_promotion"]
    assert no_promotion[["customers_targeted", "campaign_contact_cost", "simulated_net_incremental_margin",
                         "simulated_margin_vs_no_promotion"]].eq(0).all().all()
    assert proposed["customers_targeted"].eq(0).all()
    assert proposed["campaign_contact_cost"].eq(0).all()
    assert proposed["simulated_net_incremental_margin"].eq(0).all()
    assert baselines["customers_targeted"].eq(capacity).all()
    assert baselines["campaign_contact_cost"].eq(capacity * 0.5).all()
    np.testing.assert_allclose(baselines["simulated_net_incremental_margin"],
                               baselines["simulated_gross_incremental_margin"] - capacity * 0.5)
    if capacity:
        assert baselines["simulated_net_incremental_margin"].lt(0).all()
        assert proposed["simulated_margin_difference_vs_baseline"].gt(0).all()
        random_net = table[table.policy == "non_targeted"].set_index("scenario")["simulated_net_incremental_margin"]
        np.testing.assert_allclose(proposed["simulated_margin_difference_vs_baseline"],
                                   -proposed["scenario"].map(random_net))
    assert decision.bootstrap_margin_ci(np.array([]), 10, 42) == (0, 0, 0)


def test_baselines_use_k_when_proposed_selects_fewer_positive_customers(final_scores, config):
    config["decision"]["budget_total"] = 2.0  # K=4, only two positive contributions
    prepared = decision.prepare_scores(final_scores, config)
    net = pd.Series([-10.0, 2.0, -3.0, 5.0, 0.0, -1.0])
    selected = decision.policy_selections(prepared, net, config)
    assert selected["non_targeted"].sum() == 4
    assert selected["rfm"].sum() == 4
    assert selected["proposed"].sum() == 2
    assert prepared.loc[selected["rfm"], "Customer ID"].tolist() == [1, 2, 3, 6]
    assert prepared.loc[selected["proposed"], "Customer ID"].tolist() == [2, 4]


def test_fair_comparison_nonzero_capacity_and_determinism(final_scores, config):
    # Deliberately profitable TEST assumption exercises the nonempty campaign path.
    config["decision"]["budget_total"] = 1.0
    for scenario in config["decision"]["discount_response_scenarios"]:
        scenario["response_rate"] = 2.0
    table = decision.build_scenario_table(final_scores, config)
    shuffled = final_scores.sample(frac=1, random_state=7)
    pd.testing.assert_frame_equal(table, decision.build_scenario_table(shuffled, config))
    for _, policies in table.groupby("scenario"):
        policies = policies[policies.policy != "no_promotion"]
        assert policies["customers_targeted"].eq(2).all()
        assert policies["campaign_capacity"].eq(2).all()
        assert policies["campaign_contact_cost"].eq(1.0).all()
        np.testing.assert_allclose(policies["simulated_gross_incremental_margin"] - 1.0,
                                   policies["simulated_net_incremental_margin"])
        assert policies.loc[policies.policy == "proposed", "simulated_net_incremental_margin"].iloc[0] >= policies["simulated_net_incremental_margin"].max() - 1e-8


def test_sensitivity_reversed_impact_bar():
    left, width = decision.tornado_bar_bounds([100, 10], [10, 100], 50)
    np.testing.assert_array_equal(left, [-40, -40])
    np.testing.assert_array_equal(width, [90, 90])


def test_monte_carlo_deterministic_capacity_and_interval(final_scores, config):
    values = np.array([-100., -20., -10., 0., 5., 30.])
    result = decision.monte_carlo_random_policy(values, 3, 1000, 42)
    assert result == decision.monte_carlo_random_policy(values, 3, 1000, 42)
    assert result["random_policy_sample_size"] == 3
    assert result["random_policy_simulations"] == 1000
    assert np.isfinite(list(result.values())).all()
    assert result["random_p025_net_margin"] <= result["random_mean_net_margin"] <= result["random_p975_net_margin"]
    # Exactly K distinct contacts on every draw, including the full-population case.
    constant = decision.monte_carlo_random_policy(np.ones(6), 3, 30, 42)
    assert constant["random_mean_net_margin"] == constant["random_p025_net_margin"] == constant["random_p975_net_margin"] == 3
    full = decision.monte_carlo_random_policy(values, 6, 30, 42)
    assert full["random_mean_net_margin"] == full["random_p025_net_margin"] == full["random_p975_net_margin"] == values.sum()
    config["decision"]["budget_total"] = 1.5
    table = decision.build_scenario_table(final_scores, config)
    row = table[(table.scenario == "moderate") & (table.policy == "non_targeted")].iloc[0]
    prepared = decision.prepare_scores(final_scores, config)
    growth = decision.effective_response_rate(prepared, .2, config["decision"]["segment_response_multiplier"])
    net = decision.incremental_margin(prepared[decision.VALUE_COL], growth, .1, .4, .5)
    expected = decision.monte_carlo_random_policy(net, 3, 40, 42)
    assert row["simulated_net_incremental_margin"] == expected["random_mean_net_margin"]
    assert row["interval_type"] == "random_policy_simulation_not_causal_CI"


def test_break_even_equations_and_invalid_domains():
    values = pd.Series([10., 100., 1000.])
    required = decision.break_even_effective_growth(values, .4, .1, .5)
    np.testing.assert_allclose(decision.incremental_margin(values, pd.Series(required), .1, .4, .5), 0, atol=1e-12)
    discount = decision.break_even_discount(values, .4, .5, .5)
    np.testing.assert_allclose(decision.incremental_margin(values, pd.Series([.5]*3), discount, .4, .5), 0, atol=1e-12)
    np.testing.assert_allclose(decision.break_even_base_response(required, 1.3) * 1.3, required)
    assert np.isnan(decision.break_even_effective_growth([0, -1, 100], .4, [.1, .1, .4], .5)).all()
    assert np.isnan(decision.break_even_effective_growth(100, .4, .5, .5))
    assert np.isnan(decision.break_even_base_response(required, 0)).all()
    assert np.isnan(decision.break_even_discount([0, 100], .4, [.5, -1], .5)).all()


def test_decision_surface_dimensions_labels_and_reference(final_scores, config):
    surface = decision.response_discount_surface(final_scores, config)
    grid = config["decision"]["sensitivity_grid"]
    assert len(surface) == len(grid["discount_pct"]) * len(grid["response_rate"])
    assert not surface.duplicated(["discount_pct", "response_rate"]).any()
    assert surface["analysis_type"].eq("scenario_based_non_causal").all()
    assert surface["evaluation_population"].eq("final_temporal_test").all()
    assert surface["simulated_net_incremental_margin"].equals(surface["simulated_margin_vs_no_promotion"])
    assert (surface["customers_targeted"] <= surface["campaign_capacity"]).all()
    reference = surface[(surface.discount_pct == .1) & (surface.response_rate == .2)].iloc[0]
    assert reference["customers_targeted"] == 0
    assert reference["simulated_net_incremental_margin"] == 0
    segment, summary = decision.build_break_even_tables(final_scores, config)
    assert len(segment) == 12 and len(summary) == 3
    assert summary["customers"].eq(len(final_scores)).all()
    assert summary["valid_break_even_customers"].eq(4).all()  # two nonpositive values


def test_manifest_hashes_stable_and_sensitive_to_input_changes(final_scores, config, tmp_path):
    from utils import build_run_manifest
    raw_file, config_file = tmp_path / "raw.xlsx", tmp_path / "params.yaml"
    raw_file.write_bytes(b"synthetic test input, not real workbook")
    config_file.write_bytes(b"seed: 42\n")
    first = build_run_manifest(config, raw_file, config_file, 10, final_scores)
    second = build_run_manifest(config, raw_file, config_file, 10, final_scores)
    for field in ("raw_data_sha256", "config_sha256"):
        assert first[field] == second[field]
        assert len(first[field]) == 64
    assert first["final_test_customers"] == 6 and first["raw_rows"] == 10
    assert first["python_version"].startswith("3.11.")
    assert first["git_commit"] and first["run_timestamp"]
    raw_file.write_bytes(b"changed input")
    changed = build_run_manifest(config, raw_file, config_file, 10, final_scores)
    assert changed["raw_data_sha256"] != first["raw_data_sha256"]
    assert changed["config_sha256"] == first["config_sha256"]
