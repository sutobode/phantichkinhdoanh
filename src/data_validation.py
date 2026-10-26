"""Data validation layer (pandera) — fail fast nếu raw data sai schema/type/
range, trước khi chạy cleaning/feature pipeline. Mô phỏng input contract
thật khi tích hợp vào hệ thống production.

Dùng: import và gọi validate_raw() ở đầu 01_eda.py / 02_cleaning_features.py,
hoặc chạy độc lập: python src/data_validation.py
"""
from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

from utils import get_logger

logger = get_logger("data_validation")

RAW_SCHEMA = DataFrameSchema(
    {
        "Invoice": Column(str, nullable=False, coerce=True),
        "StockCode": Column(str, nullable=False, coerce=True),
        "Description": Column(str, nullable=True, coerce=True),
        "Quantity": Column(int, nullable=False, coerce=True),
        "InvoiceDate": Column(pa.Timestamp, nullable=False),
        "Price": Column(float, nullable=False, coerce=True),
        "Customer ID": Column(float, nullable=True, coerce=True,
                               checks=Check.in_range(10000, 99999, ignore_na=True)),
        "Country": Column(str, nullable=False, coerce=True),
    },
    strict=False,  # cho phép cột thừa (dataset có thể thêm cột trong tương lai)
)


def validate_raw(df: pd.DataFrame) -> pd.DataFrame:
    try:
        validated = RAW_SCHEMA.validate(df, lazy=True)
        logger.info("Data validation PASSED: %d dòng, %d cột", len(validated), len(validated.columns))
        return validated
    except pa.errors.SchemaErrors as e:
        logger.error("Data validation FAILED:\n%s", e.failure_cases.to_string())
        raise


if __name__ == "__main__":
    from importlib import import_module

    from utils import load_config

    cleaning = import_module("02_cleaning_features")
    config = load_config()
    raw = cleaning.load_raw(config)
    validate_raw(raw)
