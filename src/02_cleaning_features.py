"""Week 2 — Cleaning, RFM, cohort view, feature engineering, labels.

Đọc raw Excel; expanding snapshots with 90-day repeat/gross-sales labels.

QUAN TRỌNG (chống leakage): thứ tự xử lý là **split theo thời gian TRƯỚC,
clean (loại cancellation/qty<=0/price<=0) SAU, riêng biệt theo từng window**.
Nếu clean toàn cục rồi mới split, feature "tỷ lệ return" sẽ không tính được
đúng theo window, và về nguyên tắc rất dễ vô tình trộn thông tin giữa observation
và outcome window khi thêm feature mới sau này. Mọi hàm compute_* chỉ nhận
input đã được giới hạn trong đúng 1 window.

Decision dates are configured and checked against actual raw coverage.

Output:
  - data/processed/customer_snapshots.csv (train/validation/final_test)
  - data/processed/customer_features.csv
  - data/processed/customer_labels.csv
  - outputs/cohort_retention.csv

Chạy: python src/02_cleaning_features.py
"""
from __future__ import annotations

import pandas as pd

from utils import get_logger, load_config, outputs_path, processed_path, raw_path

logger = get_logger("02_cleaning_features")

RAW_FILENAME = "online_retail_II.xlsx"


def load_raw(config: dict) -> pd.DataFrame:
    path = raw_path(RAW_FILENAME, config)
    sheets = pd.read_excel(path, sheet_name=None)
    df = pd.concat(sheets.values(), ignore_index=True)
    return df


def audit_duplicates(raw: pd.DataFrame) -> pd.DataFrame:
    """Count exact repeats beyond the first occurrence; retain all rows."""
    count = int(raw.duplicated().sum())
    logger.info("Exact duplicate rows beyond first: %d; policy=retain", count)
    return pd.DataFrame([{"raw_rows": len(raw), "exact_duplicate_rows": count,
                          "duplicate_policy": "retain"}])


def filter_valid_customers(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Lọc duy nhất điều kiện không phụ thuộc thời gian: phải có Customer ID
    (unit of analysis = customer). Giữ nguyên cancellations/qty<=0/price<=0 ở
    bước này để có thể split theo window trước khi quyết định loại gì."""
    df = df.copy()
    if config["cleaning"]["drop_missing_customer_id"]:
        n_missing = df["Customer ID"].isna().sum()
        logger.info("Loại %d dòng Customer ID null (toàn bộ dataset)", n_missing)
        df = df.dropna(subset=["Customer ID"])
    df["Customer ID"] = df["Customer ID"].astype(int)
    return df


def split_windows(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split theo InvoiceDate TRƯỚC khi loại cancellation/qty/price — mỗi
    window sau đó được clean độc lập, không window nào nhìn thấy dữ liệu của
    window kia."""
    w = config["windows"]
    if w["observation_start"] is None or w["observation_end"] is None or w["outcome_end"] is None:
        raise ValueError(
            "Chưa chốt observation/outcome window trong config/params.yaml. "
            "Chạy src/01_eda.py trước và điền giá trị."
        )
    # Các mốc ngày trong config (vd "2011-06-30") parse ra thành midnight
    # (00:00:00). Nếu so sánh "<= obs_end" trực tiếp, mọi giao dịch xảy ra
    # SAU 00:00:00 của chính ngày đó sẽ bị loại nhầm (off-by-one tại biên).
    # Cách đúng: coi mốc "end" là BAO GỒM TRỌN NGÀY đó, nên cộng thêm 1 ngày
    # và dùng "<" (exclusive) thay vì "<=" trên mốc gốc.
    obs_start = pd.Timestamp(w["observation_start"])
    obs_end_exclusive = pd.Timestamp(w["observation_end"]) + pd.Timedelta(days=1)
    outcome_end_exclusive = pd.Timestamp(w["outcome_end"]) + pd.Timedelta(days=1)

    obs_df = df[(df["InvoiceDate"] >= obs_start) & (df["InvoiceDate"] < obs_end_exclusive)]
    outcome_df = df[(df["InvoiceDate"] >= obs_end_exclusive) & (df["InvoiceDate"] < outcome_end_exclusive)]
    return obs_df, outcome_df


def compute_return_rate(window_df: pd.DataFrame) -> pd.Series:
    """Tỷ lệ invoice bị cancel trong CHÍNH window này (trước khi loại
    cancellation). window_df phải đã được split theo thời gian, chưa clean."""
    is_cancel = window_df["Invoice"].astype(str).str.startswith("C")
    n_cancelled = window_df[is_cancel].groupby("Customer ID")["Invoice"].nunique()
    n_total = window_df.groupby("Customer ID")["Invoice"].nunique()
    return (n_cancelled.reindex(n_total.index, fill_value=0) / n_total).rename("return_rate")


def drop_invalid_rows(window_df: pd.DataFrame, config: dict, window_name: str) -> pd.DataFrame:
    """Loại cancellation/qty<=0/price<=0 — áp dụng riêng cho 1 window đã split,
    log số dòng loại riêng theo từng window để source log minh bạch."""
    rules = config["cleaning"]
    n0 = len(window_df)
    df = window_df.copy()

    if rules["drop_cancelled_invoices"]:
        is_cancel = df["Invoice"].astype(str).str.startswith("C")
        logger.info("[%s] Loại %d dòng cancellation (Invoice bắt đầu 'C')", window_name, is_cancel.sum())
        df = df[~is_cancel]

    if rules["drop_non_positive_quantity"]:
        n_bad = (df["Quantity"] <= 0).sum()
        logger.info("[%s] Loại %d dòng Quantity <= 0", window_name, n_bad)
        df = df[df["Quantity"] > 0]

    if rules["drop_non_positive_price"]:
        n_bad = (df["Price"] <= 0).sum()
        logger.info("[%s] Loại %d dòng Price <= 0", window_name, n_bad)
        df = df[df["Price"] > 0]

    df["Revenue"] = df["Quantity"] * df["Price"]

    logger.info("[%s] Tổng: %d -> %d dòng sau cleaning (loại %d, %.2f%%)",
                window_name, n0, len(df), n0 - len(df), 100 * (n0 - len(df)) / max(n0, 1))
    return df


def compute_rfm_features(obs_clean: pd.DataFrame, return_rate: pd.Series, as_of: pd.Timestamp) -> pd.DataFrame:
    grouped = obs_clean.groupby("Customer ID").agg(
        recency_days=("InvoiceDate", lambda s: (as_of - s.max()).days),
        frequency=("Invoice", "nunique"),
        monetary=("Revenue", "sum"),
        avg_order_value=("Revenue", lambda s: s.sum() / obs_clean.loc[s.index, "Invoice"].nunique()),
        n_distinct_products=("StockCode", "nunique"),
        tenure_days=("InvoiceDate", lambda s: (as_of - s.min()).days),
        country=("Country", "first"),
    )
    grouped = grouped.join(return_rate, how="left")
    grouped["return_rate"] = grouped["return_rate"].fillna(0.0)
    return grouped.reset_index()


def compute_cohort_view(obs_clean: pd.DataFrame) -> pd.DataFrame:
    """Cohort theo tháng acquisition (tháng của giao dịch đầu tiên trong observation
    window) x số tháng kể từ acquisition -> số khách hàng còn mua (retention)."""
    first_purchase = obs_clean.groupby("Customer ID")["InvoiceDate"].min().rename("first_purchase")
    merged = obs_clean.merge(first_purchase, on="Customer ID")
    merged["cohort_month"] = merged["first_purchase"].dt.to_period("M")
    merged["order_month"] = merged["InvoiceDate"].dt.to_period("M")
    merged["months_since_acquisition"] = (
        (merged["order_month"] - merged["cohort_month"]).apply(lambda p: p.n)
    )

    cohort = (
        merged.groupby(["cohort_month", "months_since_acquisition"])["Customer ID"]
        .nunique()
        .reset_index(name="n_active_customers")
    )
    cohort_sizes = cohort[cohort["months_since_acquisition"] == 0].set_index("cohort_month")[
        "n_active_customers"
    ]
    cohort["cohort_size"] = cohort["cohort_month"].map(cohort_sizes)
    cohort["retention_rate"] = cohort["n_active_customers"] / cohort["cohort_size"]
    cohort["cohort_month"] = cohort["cohort_month"].astype(str)
    return cohort


def compute_labels(outcome_clean: pd.DataFrame, obs_customers: pd.Index) -> pd.DataFrame:
    """Nhãn chỉ tính cho khách hàng đã xuất hiện trong observation window
    (đúng unit of analysis đã định nghĩa). outcome_clean đã được split+clean
    riêng, hoàn toàn không overlap với dữ liệu dùng để tính feature."""
    future_value = outcome_clean.groupby("Customer ID")["Revenue"].sum()
    labels = pd.DataFrame(index=obs_customers)
    labels["future_gross_sales"] = future_value.reindex(obs_customers).fillna(0.0)
    labels["repeat_purchase"] = (labels["future_gross_sales"] > 0).astype(int)
    return labels.reset_index().rename(columns={"index": "Customer ID"})


def snapshot_schedule(config: dict) -> pd.DataFrame:
    """Half-open outcome intervals; reject label overlap with the next evaluation."""
    v = config["validation"]
    dates = v["train_decision_dates"] + [v["validation_decision_date"], v["final_test_decision_date"]]
    roles = ["train"] * len(v["train_decision_dates"]) + ["validation", "final_test"]
    decisions = pd.to_datetime(dates)
    if (not v["train_decision_dates"] or v["outcome_horizon_days"] <= 0
            or decisions.has_duplicates or not decisions.is_monotonic_increasing
            or decisions[0] <= pd.Timestamp(config["windows"]["observation_start"])):
        raise ValueError("Invalid temporal snapshot schedule")
    schedule = pd.DataFrame({"decision_date": decisions, "split": roles})
    schedule["outcome_end_exclusive"] = decisions + pd.Timedelta(days=v["outcome_horizon_days"])
    for role in ("validation", "final_test"):
        evaluation_date = schedule.loc[schedule["split"] == role, "decision_date"].iloc[0]
        earlier = schedule[schedule["decision_date"] < evaluation_date]
        if (earlier["outcome_end_exclusive"] >= evaluation_date).any():
            raise ValueError("Training outcomes must finish before the evaluation decision time")
    return schedule


def build_snapshot(raw_filtered: pd.DataFrame, config: dict, decision_date, split: str) -> pd.DataFrame:
    decision_date = pd.Timestamp(decision_date)
    end = decision_date + pd.Timedelta(days=config["validation"]["outcome_horizon_days"])
    window_config = {"windows": {
        "observation_start": config["windows"]["observation_start"],
        "observation_end": decision_date - pd.Timedelta(days=1),
        "outcome_end": end - pd.Timedelta(days=1),
    }}
    obs_raw, outcome_raw = split_windows(raw_filtered, window_config)
    obs_clean = drop_invalid_rows(obs_raw, config, f"{split}/observation")
    outcome_clean = drop_invalid_rows(outcome_raw, config, f"{split}/outcome")
    if obs_clean.empty:
        raise ValueError(f"Empty observation window at {decision_date}")
    features = compute_rfm_features(obs_clean, compute_return_rate(obs_raw), decision_date)
    labels = compute_labels(outcome_clean, features["Customer ID"])
    snapshot = features.merge(labels, on="Customer ID", validate="one_to_one")
    snapshot["decision_date"] = decision_date
    snapshot["outcome_end_exclusive"] = end
    snapshot["split"] = split
    return snapshot


def build_temporal_snapshots(raw: pd.DataFrame, config: dict) -> pd.DataFrame:
    schedule = snapshot_schedule(config)
    # Last recorded calendar day may be partial: require coverage up to the
    # exclusive midnight ending every outcome window, without inventing labels.
    if (raw.empty or raw["InvoiceDate"].min().normalize() > pd.Timestamp(config["windows"]["observation_start"])
            or schedule["outcome_end_exclusive"].max() > raw["InvoiceDate"].max().normalize()):
        raise ValueError("Raw data does not cover the complete configured observation/outcome windows")
    filtered = filter_valid_customers(raw, config)
    return pd.concat([
        build_snapshot(filtered, config, row.decision_date, row.split)
        for row in schedule.itertuples(index=False)
    ], ignore_index=True)


def temporal_partitions(snapshots: pd.DataFrame, config: dict):
    schedule = snapshot_schedule(config)
    df = snapshots.copy()
    for col in ("decision_date", "outcome_end_exclusive"):
        df[col] = pd.to_datetime(df[col])
    if df.duplicated(["Customer ID", "decision_date"]).any():
        raise ValueError("Duplicate customer snapshots")
    actual = df[["decision_date", "split", "outcome_end_exclusive"]].drop_duplicates().sort_values("decision_date")
    if not actual.reset_index(drop=True).equals(schedule):
        raise ValueError("Snapshot dates/labels do not match the configured temporal schedule")
    return tuple(df[df["split"] == role].copy() for role in ("train", "validation", "final_test"))


def main() -> None:
    config = load_config()
    raw = load_raw(config)

    audit_duplicates(raw).to_csv(outputs_path("data_quality.csv"), index=False)

    from data_validation import validate_raw
    raw = validate_raw(raw)

    snapshots = build_temporal_snapshots(raw, config)
    snapshots.to_csv(processed_path("customer_snapshots.csv", config), index=False)
    snapshot_schedule(config).to_csv(outputs_path("temporal_schedule.csv"), index=False)
    _, _, final = temporal_partitions(snapshots, config)
    label_cols = ["Customer ID", "future_gross_sales", "repeat_purchase"]
    features = final.drop(columns=["future_gross_sales", "repeat_purchase"])
    labels = final[label_cols]
    filtered = filter_valid_customers(raw, config)
    obs_raw = filtered[(filtered["InvoiceDate"] >= pd.Timestamp(config["windows"]["observation_start"]))
                       & (filtered["InvoiceDate"] < pd.Timestamp(config["validation"]["final_test_decision_date"]))]
    obs_clean = drop_invalid_rows(obs_raw, config, "final_test/cohort")
    cohort = compute_cohort_view(obs_clean)
    cohort_path = outputs_path("cohort_retention.csv")
    cohort.to_csv(cohort_path, index=False)
    logger.info("Đã ghi cohort retention: %s", cohort_path)

    features_path = processed_path("customer_features.csv", config)
    labels_path = processed_path("customer_labels.csv", config)
    features.to_csv(features_path, index=False)
    labels.to_csv(labels_path, index=False)
    logger.info("Đã ghi features: %s", features_path)
    logger.info("Đã ghi labels: %s", labels_path)

    repeat_rate = labels["repeat_purchase"].mean()
    logger.info("Tỷ lệ repeat_purchase trong outcome window: %.2f%%", 100 * repeat_rate)


if __name__ == "__main__":
    main()
