"""Week 1 — EDA sơ bộ trên Online Retail II.

Đọc raw Excel (2 sheet: "Year 2009-2010", "Year 2010-2011"), gộp lại,
in/ghi ra outputs/ các thống kê cần để chốt observation/outcome window
trong config/params.yaml.

Chạy: python src/01_eda.py
"""
from __future__ import annotations

import pandas as pd

from utils import PROJECT_ROOT, get_logger, load_config, raw_path

logger = get_logger("01_eda")

RAW_FILENAME = "online_retail_II.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"


def load_raw(config: dict) -> pd.DataFrame:
    path = raw_path(RAW_FILENAME, config)
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {path}. Giải nén data/raw/online_retail_ii.zip trước."
        )
    sheets = pd.read_excel(path, sheet_name=None)
    logger.info("Sheets tìm thấy: %s", list(sheets.keys()))
    df = pd.concat(sheets.values(), ignore_index=True)
    logger.info("Tổng số dòng sau khi gộp sheet: %d", len(df))
    return df


def basic_profile(df: pd.DataFrame) -> None:
    logger.info("Khoảng thời gian dữ liệu: %s -> %s",
                df["InvoiceDate"].min(), df["InvoiceDate"].max())
    logger.info("Số invoice duy nhất: %d", df["Invoice"].nunique())
    logger.info("Số khách hàng duy nhất (Customer ID không null): %d",
                df["Customer ID"].nunique())
    logger.info("Số dòng Customer ID bị null: %d (%.2f%%)",
                df["Customer ID"].isna().sum(),
                100 * df["Customer ID"].isna().mean())

    n_cancelled = df["Invoice"].astype(str).str.startswith("C").sum()
    logger.info("Số dòng là cancellation (Invoice bắt đầu 'C'): %d (%.2f%%)",
                n_cancelled, 100 * n_cancelled / len(df))

    n_neg_qty = (df["Quantity"] <= 0).sum()
    n_neg_price = (df["Price"] <= 0).sum()
    logger.info("Số dòng Quantity <= 0: %d", n_neg_qty)
    logger.info("Số dòng Price <= 0: %d", n_neg_price)

    logger.info("Top 10 quốc gia theo số dòng:\n%s",
                df["Country"].value_counts().head(10).to_string())


def monthly_transaction_profile(df: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        df.assign(month=df["InvoiceDate"].dt.to_period("M"))
        .groupby("month")
        .agg(n_invoices=("Invoice", "nunique"), n_customers=("Customer ID", "nunique"))
        .reset_index()
    )
    monthly["month"] = monthly["month"].astype(str)
    return monthly


def repeat_purchase_rate_by_horizon(df: pd.DataFrame) -> None:
    """Gợi ý chọn prediction horizon: với mỗi tháng acquisition,
    bao nhiêu % khách hàng mua lại trong 1/3/6 tháng tiếp theo."""
    clean = df.dropna(subset=["Customer ID"])
    clean = clean[~clean["Invoice"].astype(str).str.startswith("C")]
    first_purchase = clean.groupby("Customer ID")["InvoiceDate"].min().rename("first_purchase")
    merged = clean.merge(first_purchase, on="Customer ID")
    merged["days_since_first"] = (merged["InvoiceDate"] - merged["first_purchase"]).dt.days

    for horizon_days in (30, 90, 180):
        repeat_customers = merged.loc[
            merged["days_since_first"] > 0,
            "Customer ID",
        ][merged["days_since_first"] <= horizon_days].nunique()
        total_customers = first_purchase.shape[0]
        logger.info(
            "Tỷ lệ repeat-purchase trong %d ngày kể từ lần mua đầu: %.1f%% (%d/%d)",
            horizon_days, 100 * repeat_customers / total_customers,
            repeat_customers, total_customers,
        )


def main() -> None:
    config = load_config()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw(config)
    basic_profile(df)

    monthly = monthly_transaction_profile(df)
    monthly_path = OUTPUT_DIR / "monthly_transaction_profile.csv"
    monthly.to_csv(monthly_path, index=False)
    logger.info("Đã ghi profile theo tháng: %s", monthly_path)

    repeat_purchase_rate_by_horizon(df)

    logger.info(
        "BƯỚC TIẾP THEO: dựa vào khoảng thời gian và profile theo tháng ở trên, "
        "chốt observation_start/observation_end/outcome_end trong config/params.yaml."
    )


if __name__ == "__main__":
    main()
