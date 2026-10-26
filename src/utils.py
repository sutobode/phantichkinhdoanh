"""Shared helpers: config loading, logging setup, path resolution.

Mọi script trong src/ import từ đây để tránh lặp code và đảm bảo
mọi nơi đọc tham số từ cùng một config/params.yaml (single source of truth).
"""
from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "params.yaml"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def get_logger(name: str) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(LOG_DIR / f"{name}.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    # Console Windows mặc định dùng codepage (cp1252/cp437), in tiếng Việt bị
    # lỗi (mojibake). reconfigure sang UTF-8 nếu stream hỗ trợ (Python 3.7+).
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    return logger


def processed_path(filename: str, config: dict) -> Path:
    path = PROJECT_ROOT / config["data"]["processed_dir"] / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def raw_path(filename: str, config: dict) -> Path:
    return PROJECT_ROOT / config["data"]["raw_dir"] / filename


def outputs_path(filename: str) -> Path:
    path = PROJECT_ROOT / "outputs" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def merge_or_replace_columns(base_df, new_df, on: str):
    """merge new_df vào base_df theo khóa `on`, nhưng trước tiên DROP khỏi
    base_df mọi cột (trừ `on`) đã tồn tại trong new_df.

    Dùng khi 1 script ghi đè/bổ sung cột vào 1 file CSV đã tồn tại (vd
    customer_scores.csv) mà script khác cũng ghi vào — nếu dùng merge() trần,
    chạy lại pipeline không đúng thứ tự "từ đầu" sẽ tạo cột trùng "_x"/"_y"
    (đã phát hiện bug thật khi chạy 04_advanced_models.py sau
    05_decision_scenario.py — xem tests/test_pipeline_idempotency.py)."""
    overlap = [c for c in new_df.columns if c != on and c in base_df.columns]
    base_df = base_df.drop(columns=overlap)
    return base_df.merge(new_df, on=on, how="left")
