"""Unit test cho các hàm thuần trong src/11_assumption_robustness.py (không cần raw data)."""
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

ar = import_module("11_assumption_robustness")
decision = import_module("05_decision_scenario")


def test_alpha_matches_decision_formula_per_pound():
    v = pd.Series([1.0])
    for d, b, m, g in [(0.10, 0.20, 1.3, 0.40), (0.05, 0.35, 0.5, 0.30), (0.20, 0.10, 1.0, 0.50)]:
        expected = decision.incremental_margin(v, pd.Series([b * m]), d, g).iloc[0]
        assert ar.alpha(d, b, m, g) == pytest.approx(expected)


def test_reference_scenario_alpha_negative_for_every_segment():
    for m in (0.5, 1.0, 1.3, 0.4):
        assert ar.alpha(0.10, 0.20, m, 0.40) < 0


def test_elasticity_break_even_zeroes_alpha():
    for d, g in [(0.05, 0.40), (0.10, 0.40), (0.20, 0.30)]:
        e = ar.elasticity_break_even(d, g)
        growth = ar.elasticity_growth(d, -e)
        assert ar.alpha(d, growth, 1.0, g) == pytest.approx(0.0, abs=1e-12)
    assert ar.elasticity_break_even(0.5, 0.4) == float("inf")


def test_analytic_random_expectation_matches_exhaustive_mean():
    from itertools import combinations
    net = np.array([3.0, -1.0, 2.0, -5.0, 0.5])
    k = 2
    exhaustive = np.mean([net[list(c)].sum() for c in combinations(range(len(net)), k)])
    assert ar.analytic_random_expectation(net, k) == pytest.approx(exhaustive)


def test_policy_abstains_when_all_alpha_negative_and_respects_capacity():
    margin = pd.Series([-1.0, -0.2, -3.0])
    assert not decision.budget_constrained_selection(margin, 750.0, 0.5).any()
    margin = pd.Series(np.linspace(-1, 5, 4000))
    selected = decision.budget_constrained_selection(margin, 750.0, 0.5)
    assert 0 <= selected.sum() <= 1500
    assert (margin[selected] > 0).all()
