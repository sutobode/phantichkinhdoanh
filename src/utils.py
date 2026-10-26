"""Shared helpers: config loading, logging setup, path resolution.

Mọi script trong src/ import từ đây để tránh lặp code và đảm bảo
mọi nơi đọc tham số từ cùng một config/params.yaml (single source of truth).
"""
from __future__ import annotations

import logging
import hashlib
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
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


def validate_final_test_predictions(df, config: dict) -> None:
    """Reject legacy/demo/mixed-snapshot scores before any policy evaluation."""
    import pandas as pd

    required = {"Customer ID", "decision_date", "outcome_end_exclusive", "split",
                "training_outcome_end_exclusive", "evaluation_design"}
    if df.empty or not required.issubset(df.columns):
        raise ValueError("Final temporal-test predictions required; rerun steps 02 and 03")
    decision = pd.Timestamp(config["validation"]["final_test_decision_date"])
    end = decision + pd.Timedelta(days=config["validation"]["outcome_horizon_days"])
    if (df["Customer ID"].duplicated().any()
            or not df["split"].eq("final_test").all()
            or not df["evaluation_design"].eq("expanding_temporal_snapshots").all()
            or not pd.to_datetime(df["decision_date"]).eq(decision).all()
            or not pd.to_datetime(df["outcome_end_exclusive"]).eq(end).all()
            or not pd.to_datetime(df["training_outcome_end_exclusive"]).lt(decision).all()):
        raise ValueError("Policy evaluation requires only out-of-sample final temporal-test predictions")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_run_manifest(config: dict, raw_file: Path, config_file: Path, raw_rows: int,
                       scores, requirements_file: Path | None = None) -> dict:
    """Small run evidence record; hashes are of actual file bytes, never constants."""
    validate_final_test_predictions(scores, config)
    manifest = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
        "git_worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True).strip()),
        "python_version": platform.python_version(), "platform": platform.platform(),
        "raw_data_sha256": sha256_file(raw_file), "config_sha256": sha256_file(config_file),
        "final_test_decision_date": config["validation"]["final_test_decision_date"],
        "outcome_horizon_days": config["validation"]["outcome_horizon_days"],
        "value_prediction_source": config["decision"]["value_prediction_source"],
        "raw_rows": int(raw_rows), "final_test_customers": int(scores["Customer ID"].nunique()),
    }
    if requirements_file is not None:
        manifest["requirements_sha256"] = sha256_file(requirements_file)
    return manifest
