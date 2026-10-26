"""Unit test cho logic tài chính ở Tuần 4: incremental margin formula,
budget-constrained selection, recommendation assignment.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from importlib import import_module

decision = import_module("05_decision_scenario")


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
    # incremental_margin = 1000 * 0.40 * [(1.20)(0.90) - 1] = 400 * [1.08 - 1] = 400*0.08 = 32
    v = pd.Series([1000.0])
    eff_rr = pd.Series([0.20])
    result = decision.incremental_margin(v, eff_rr, discount_pct=0.10, gross_margin_pct=0.40)
    assert result.iloc[0] == pytest.approx(32.0)


def test_incremental_margin_negative_when_discount_exceeds_response_benefit():
    # eff_response_rate=0 (không phản hồi gì thêm), discount=0.10 -> margin phải ÂM
    # vì chỉ mất margin do giảm giá mà không có thêm doanh thu.
    v = pd.Series([1000.0])
    eff_rr = pd.Series([0.0])
    result = decision.incremental_margin(v, eff_rr, discount_pct=0.10, gross_margin_pct=0.40)
    assert result.iloc[0] < 0
    assert result.iloc[0] == pytest.approx(1000 * 0.40 * (1 * 0.90 - 1))  # = -40


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
        "predicted_future_value_gbm": [500.0, 300.0, 0.0, 200.0],
    })
    # Champions: margin<=0 nhưng value>0 -> retain
    # At-risk: margin>0 -> nurture
    # Lost: margin<=0, value=0 -> no-offer
    # Loyal: margin<=0 nhưng value>0 -> retain
    margin_reference = pd.Series([-10.0, 50.0, -5.0, -2.0])
    rec = decision.assign_recommendation(df, margin_reference)
    assert rec.tolist() == ["retain", "nurture", "no-offer", "retain"]
