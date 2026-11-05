"""Unit test cho các hàm thuần trong src/10_analysis_extensions.py (không cần raw data)."""
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

ext = import_module("10_analysis_extensions")
from utils import load_config  # noqa: E402


@pytest.fixture
def config():
    return load_config()


def test_sample_size_matches_closed_form():
    # 2 * (1.959964 + 0.841621)^2 * 1^2 / 0.5^2 = 62.79 -> 63
    assert ext.sample_size_per_arm(sigma=1.0, delta=0.5) == 63


def test_sample_size_rejects_nonpositive_delta():
    with pytest.raises(ValueError):
        ext.sample_size_per_arm(sigma=1.0, delta=0.0)


def test_break_even_multiplier_zeroes_alpha():
    d, b, g = 0.10, 0.20, 0.40
    m = ext.break_even_multiplier(d, b, g)
    assert (1 + b * m) * (g - d) - g == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(ext.break_even_multiplier(0.5, 0.2, 0.4))  # discount >= margin


def test_smearing_factor_is_one_for_zero_residuals():
    assert ext.smearing_factor(np.zeros(10)) == pytest.approx(1.0)


def test_is_product_keeps_numeric_stockcodes_only():
    raw = pd.DataFrame({"StockCode": ["85123A", "22423", "POST", "M", "AMAZONFEE", 21232]})
    assert ext.is_product(raw).tolist() == [True, True, False, False, False, True]


def test_monte_carlo_never_selects_when_every_alpha_negative(config):
    cfg = {"decision": dict(config["decision"])}
    # growth = 0 in every scenario -> alpha = (g - d) - g = -d < 0 for every draw
    cfg["decision"]["discount_response_scenarios"] = [
        {"name": "a", "discount_pct": 0.05, "response_rate": 0.0},
        {"name": "b", "discount_pct": 0.20, "response_rate": 0.0},
    ]
    v = np.array([100.0, 2000.0, 0.0, 50.0])
    sims = ext.monte_carlo_assumptions(v, np.array([0, 1, 2, 3]), np.array([0.5, 1.0, 1.3, 0.4]), cfg,
                                       n_draws=50, mode="independent", seed=1)
    assert (sims["n_selected"] == 0).all()
    assert (sims["proposed_net_margin"] == 0).all()
    summary = ext.summarize_monte_carlo(sims)
    assert summary["P(policy chọn 0 khách)"] == 1.0
    assert summary["P(proposed >= random)"] == 1.0


def test_monte_carlo_is_deterministic_for_seed(config):
    v = np.linspace(0, 1000, 40)
    seg = np.arange(40) % 4
    mult = np.array([0.5, 1.0, 1.3, 0.4])
    a = ext.monte_carlo_assumptions(v, seg, mult, config, n_draws=20, mode="linked", seed=7)
    b = ext.monte_carlo_assumptions(v, seg, mult, config, n_draws=20, mode="linked", seed=7)
    pd.testing.assert_frame_equal(a, b)
