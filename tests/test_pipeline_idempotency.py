"""Regression test cho bug thật đã phát hiện: chạy 04_advanced_models.py sau
05_decision_scenario.py (hoặc chạy 04 hai lần liên tiếp) làm customer_scores.csv
bị merge trùng cột ("predicted_future_value_gbm_x", "..._y") thay vì ghi đè.

utils.merge_or_replace_columns() là fix cho bug này — test khóa lại hành vi
đúng để không ai vô tình quay về dùng merge() trần.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils import merge_or_replace_columns


def test_merge_or_replace_columns_no_duplicate_on_rerun():
    base = pd.DataFrame({
        "Customer ID": [1, 2, 3],
        "existing_col": ["a", "b", "c"],
        "predicted_future_value_gbm": [10.0, 20.0, 30.0],
    })
    new = pd.DataFrame({
        "Customer ID": [1, 2, 3],
        "predicted_future_value_gbm": [11.0, 21.0, 31.0],  # giá trị mới sau khi re-train
    })

    result = merge_or_replace_columns(base, new, on="Customer ID")

    assert "predicted_future_value_gbm_x" not in result.columns
    assert "predicted_future_value_gbm_y" not in result.columns
    assert list(result.columns) == ["Customer ID", "existing_col", "predicted_future_value_gbm"]
    assert result["predicted_future_value_gbm"].tolist() == [11.0, 21.0, 31.0]
    assert result["existing_col"].tolist() == ["a", "b", "c"]  # cột không liên quan giữ nguyên


def test_merge_or_replace_columns_idempotent_across_multiple_calls():
    """Mô phỏng chạy script 3 lần liên tiếp — kết quả lần 3 phải giống hệt lần 1."""
    base = pd.DataFrame({"Customer ID": [1, 2], "score": [0.5, 0.6]})
    new = pd.DataFrame({"Customer ID": [1, 2], "score": [0.7, 0.8]})

    once = merge_or_replace_columns(base, new, on="Customer ID")
    twice = merge_or_replace_columns(once, new, on="Customer ID")
    thrice = merge_or_replace_columns(twice, new, on="Customer ID")

    assert list(once.columns) == list(twice.columns) == list(thrice.columns)
    assert once["score"].tolist() == twice["score"].tolist() == thrice["score"].tolist()
