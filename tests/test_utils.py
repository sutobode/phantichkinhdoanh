"""Unit test cho các helper dùng chung trong utils.py — đặc biệt các bug đã
phát hiện qua senior code review: giá trị dự đoán âm gây lật dấu margin, và
bootstrap_ci crash khi mọi resample đều lỗi.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils import bootstrap_ci, clip_nonneg


def test_clip_nonneg_clips_negative_to_zero():
    preds = np.array([-10.0, 0.0, 5.0, -0.01, 100.0])
    result = clip_nonneg(preds)
    assert result.tolist() == [0.0, 0.0, 5.0, 0.0, 100.0]


def test_clip_nonneg_prevents_sign_flip_in_incremental_margin():
    """Regression test cho bug thật: customer có predicted_future_value ÂM
    (model noise) x bracket ÂM (vd Champions, discount thấp) = margin DƯƠNG
    GIẢ, khiến assign_recommendation gắn nhầm "nurture" cho khách hàng model
    thực ra dự đoán giá trị âm/vô nghĩa (customer 17850 trong dữ liệu thật)."""
    from importlib import import_module

    decision = import_module("05_decision_scenario")
    import pandas as pd

    raw_negative_value = pd.Series([-232.79])
    eff_rr = pd.Series([0.10])
    # bracket = (1.10)(0.90) - 1 = -0.01 (âm) -> margin SAI (dương) nếu không clip
    margin_without_clip = decision.incremental_margin(raw_negative_value, eff_rr, 0.10, 0.40)
    assert margin_without_clip.iloc[0] > 0, "Xác nhận bug tồn tại nếu không clip trước"

    clipped_value = pd.Series(clip_nonneg(raw_negative_value.to_numpy()))
    margin_with_clip = decision.incremental_margin(clipped_value, eff_rr, 0.10, 0.40)
    assert margin_with_clip.iloc[0] == 0.0, "Sau khi clip, customer value=0 phải cho margin=0, không dương giả"


def test_bootstrap_ci_normal_case():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 200)
    y_score = rng.random(200)
    point, lo, hi = bootstrap_ci(y_true, y_score, lambda a, b: np.mean(a == (b > 0.5)), n_boot=50, seed=1)
    assert lo <= point <= hi


def test_bootstrap_ci_does_not_crash_when_all_resamples_fail():
    """Regression test: nếu metric_fn luôn raise ValueError (mô phỏng case
    PR-AUC/ROC-AUC trên resample chỉ có 1 class — dễ xảy ra ở cutoff nhỏ/mất
    cân bằng trong rolling-origin validation), bootstrap_ci phải trả về NaN
    CI thay vì crash trên np.percentile(mảng rỗng)."""
    y_true = np.array([1, 0, 1, 0])
    y_score = np.array([0.1, 0.2, 0.3, 0.4])

    def always_fail(a, b):
        raise ValueError("mô phỏng lỗi single-class trên resample")

    point, lo, hi = bootstrap_ci(y_true, y_score, always_fail, n_boot=10, seed=1)
    assert np.isnan(point) and np.isnan(lo) and np.isnan(hi)
