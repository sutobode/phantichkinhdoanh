"""Unit test cho feature engineering & no-leakage logic.

Dùng dữ liệu giả lập nhỏ (không phụ thuộc file raw thật) để test độc lập,
nhanh, chạy được trong CI.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from importlib import import_module

cleaning = import_module("02_cleaning_features")


@pytest.fixture
def sample_transactions() -> pd.DataFrame:
    return pd.DataFrame({
        "Invoice": ["1", "2", "3", "C4", "5", "6", "C7"],
        "StockCode": ["A", "B", "A", "A", "C", "A", "A"],
        "Description": ["x"] * 7,
        "Quantity": [2, 1, 3, -1, 5, 1, -2],
        "InvoiceDate": pd.to_datetime([
            "2010-01-05", "2010-02-10", "2010-03-01",
            "2010-03-05",     # cancellation TRONG observation window
            "2010-04-01",     # trong outcome window
            "2010-05-01",     # trong outcome window
            "2010-05-10",     # cancellation TRONG outcome window -> KHÔNG được ảnh hưởng return_rate (feature)
        ]),
        "Price": [10.0, 5.0, 10.0, 10.0, 0.0, 8.0, 8.0],
        "Customer ID": [1.0, 1.0, 2.0, 2.0, None, 1.0, 1.0],
        "Country": ["UK"] * 7,
    })


def _config():
    return {
        "cleaning": {
            "drop_cancelled_invoices": True,
            "drop_missing_customer_id": True,
            "drop_non_positive_quantity": True,
            "drop_non_positive_price": True,
        },
        "windows": {
            "observation_start": "2010-01-01",
            "observation_end": "2010-03-31",
            "outcome_end": "2010-06-30",
        },
    }


def test_filter_valid_customers_drops_nulls(sample_transactions):
    filtered = cleaning.filter_valid_customers(sample_transactions, _config())
    assert filtered["Customer ID"].isna().sum() == 0
    assert len(filtered) == 6  # chỉ loại dòng Customer ID null (invoice "5")


def test_split_windows_no_leakage(sample_transactions):
    filtered = cleaning.filter_valid_customers(sample_transactions, _config())
    obs_df, outcome_df = cleaning.split_windows(filtered, _config())

    obs_end = pd.Timestamp("2010-03-31")
    assert (obs_df["InvoiceDate"] <= obs_end).all(), "observation window leak: có dòng sau obs_end"
    assert (outcome_df["InvoiceDate"] > obs_end).all(), "outcome window leak: có dòng trước/bằng obs_end"
    assert set(obs_df["Invoice"]).isdisjoint(set(outcome_df["Invoice"])), (
        "Một giao dịch không được xuất hiện ở cả 2 window"
    )


def test_split_windows_includes_full_last_day_of_each_window():
    """Regression test: observation_end/outcome_end trong config là NGÀY (vd
    "2010-03-31"), parse ra thành midnight 00:00:00. Nếu split dùng "<=" trực
    tiếp trên mốc đó, mọi giao dịch xảy ra SAU 00:00:00 của chính ngày cuối sẽ
    bị loại nhầm (off-by-one tại biên) — đã phát hiện bug này trên dữ liệu
    thật (625 dòng ngày 2011-12-09 bị mất). Test này khóa lại hành vi đúng:
    observation_end/outcome_end phải được hiểu là TRỌN NGÀY.
    """
    df = pd.DataFrame({
        "Invoice": ["1", "2", "3"],
        "StockCode": ["A", "A", "A"],
        "Description": ["x"] * 3,
        "Quantity": [1, 1, 1],
        "InvoiceDate": pd.to_datetime([
            "2010-03-31 23:59:00",  # cuối ngày observation_end -> PHẢI thuộc obs window
            "2010-04-01 00:00:01",  # đầu ngày sau -> PHẢI thuộc outcome window
            "2010-06-30 23:59:00",  # cuối ngày outcome_end -> PHẢI thuộc outcome window
        ]),
        "Price": [10.0, 10.0, 10.0],
        "Customer ID": [1.0, 1.0, 1.0],
        "Country": ["UK"] * 3,
    })
    filtered = cleaning.filter_valid_customers(df, _config())
    obs_df, outcome_df = cleaning.split_windows(filtered, _config())

    assert "1" in obs_df["Invoice"].values, "Giao dịch cuối ngày observation_end bị loại nhầm"
    assert "2" in outcome_df["Invoice"].values
    assert "3" in outcome_df["Invoice"].values, "Giao dịch cuối ngày outcome_end bị loại nhầm"
    assert len(obs_df) == 1 and len(outcome_df) == 2


def test_return_rate_only_uses_observation_window(sample_transactions):
    """Customer 1 có 1 cancellation trong observation window (C4 thuộc customer 2,
    không phải 1) và 1 cancellation trong outcome window (C7). Return rate của
    customer 1 tính trong observation window phải KHÔNG bị ảnh hưởng bởi C7."""
    filtered = cleaning.filter_valid_customers(sample_transactions, _config())
    obs_df, outcome_df = cleaning.split_windows(filtered, _config())

    return_rate_obs = cleaning.compute_return_rate(obs_df)
    # Customer 1 trong obs window chỉ có invoice "1" và "2", không có cancellation
    assert return_rate_obs.loc[1] == 0.0

    # Customer 2 trong obs window có invoice "3" (bình thường) + "C4" (cancel) -> 1/2
    assert return_rate_obs.loc[2] == pytest.approx(0.5)

    # Nếu lỡ tính return_rate trên toàn bộ dataframe (không split), customer 1 sẽ
    # bị lộ thông tin cancellation C7 ở tương lai -> phải khác với kết quả đúng ở trên
    return_rate_leaked = cleaning.compute_return_rate(filtered)
    assert return_rate_leaked.loc[1] != return_rate_obs.loc[1], (
        "Test không phân biệt được leakage — kiểm tra lại fixture"
    )


def test_rfm_features_only_use_observation_window(sample_transactions):
    filtered = cleaning.filter_valid_customers(sample_transactions, _config())
    obs_raw, _ = cleaning.split_windows(filtered, _config())
    return_rate = cleaning.compute_return_rate(obs_raw)
    obs_clean = cleaning.drop_invalid_rows(obs_raw, _config(), "observation")

    as_of = pd.Timestamp("2010-03-31")
    features = cleaning.compute_rfm_features(obs_clean, return_rate, as_of)

    # Customer 1 có giao dịch trong outcome window (invoice 5, 6) — feature
    # của customer 1 không được phản ánh các giao dịch đó.
    cust1 = features[features["Customer ID"] == 1].iloc[0]
    assert cust1["frequency"] == 2, "frequency không được đếm giao dịch ngoài observation window"
    assert cust1["return_rate"] == 0.0


def test_labels_only_use_outcome_window(sample_transactions):
    filtered = cleaning.filter_valid_customers(sample_transactions, _config())
    obs_raw, outcome_raw = cleaning.split_windows(filtered, _config())
    obs_clean = cleaning.drop_invalid_rows(obs_raw, _config(), "observation")
    outcome_clean = cleaning.drop_invalid_rows(outcome_raw, _config(), "outcome")

    labels = cleaning.compute_labels(outcome_clean, obs_clean["Customer ID"].unique())

    cust1 = labels[labels["Customer ID"] == 1].iloc[0]
    # outcome window của customer 1: invoice 5 (Qty 1 * Price 8 = 8), invoice C7 bị loại
    assert cust1["repeat_purchase"] == 1
    assert cust1["future_value"] == pytest.approx(8.0)

    cust2 = labels[labels["Customer ID"] == 2].iloc[0]
    assert cust2["repeat_purchase"] == 0
    assert cust2["future_value"] == 0.0
