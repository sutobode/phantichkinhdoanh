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
    return deepcopy(load_config())


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


def test_three_scenarios_and_equal_zero_contact_case(final_scores, config):
    table = decision.build_scenario_table(final_scores, config)
    assert set(table["scenario"]) == {"conservative", "moderate", "aggressive"}
    assert len(table) == 9
    assert table["customers_targeted"].eq(0).all()
    assert table["campaign_contact_cost"].eq(0).all()
    assert table["simulated_net_incremental_margin"].eq(0).all()
    assert decision.bootstrap_margin_ci(np.array([]), 10, 42) == (0, 0, 0)


def test_fair_comparison_nonzero_capacity_and_determinism(final_scores, config):
    # Deliberately profitable TEST assumption exercises the nonempty campaign path.
    config["decision"]["budget_total"] = 1.0
    for scenario in config["decision"]["discount_response_scenarios"]:
        scenario["response_rate"] = 2.0
    table = decision.build_scenario_table(final_scores, config)
    shuffled = final_scores.sample(frac=1, random_state=7)
    pd.testing.assert_frame_equal(table, decision.build_scenario_table(shuffled, config))
    for _, policies in table.groupby("scenario"):
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
