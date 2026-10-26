"""Shared helpers: config loading, logging setup, path resolution.

Mọi script trong src/ import từ đây để tránh lặp code và đảm bảo
mọi nơi đọc tham số từ cùng một config/params.yaml (single source of truth).
"""
from __future__ import annotations

import logging
import random
import subprocess
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


def bootstrap_ci(y_true: np.ndarray, y_score: np.ndarray, metric_fn, n_boot: int, seed: int) -> tuple[float, float, float]:
    """Bootstrap 95% CI cho 1 metric. Dùng chung cho 03_baseline_models.py và
    04_advanced_models.py (trước đây copy-paste 2 nơi — gộp lại tránh fix bug
    1 chỗ quên chỗ kia).

    Nếu TẤT CẢ resample đều lỗi (vd metric phân loại như PR-AUC/ROC-AUC trên
    1 resample chỉ toàn 1 class), trả về CI = [NaN, NaN] kèm cảnh báo thay vì
    để np.percentile crash trên mảng rỗng — resample ít dữ liệu/mất cân bằng
    (vd cutoff đầu tiên trong rolling-origin validation) có thể gặp trường hợp
    này. point estimate cũng được bọc try/except tương tự, phòng trường hợp
    cực đoan metric_fn lỗi ngay cả trên dữ liệu đầy đủ (không chỉ resample)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            stats.append(metric_fn(y_true[idx], y_score[idx]))
        except ValueError:
            continue
    try:
        point = metric_fn(y_true, y_score)
    except ValueError:
        point = float("nan")
    if not stats:
        logging.getLogger(__name__).warning(
            "bootstrap_ci: tất cả %d resample đều lỗi (có thể do mất cân bằng "
            "class nghiêm trọng) — trả về CI = [NaN, NaN]", n_boot,
        )
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return point, lo, hi


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT
        ).decode().strip()
    except Exception:
        return "unknown"


def clip_nonneg(predictions: np.ndarray) -> np.ndarray:
    """Clip dự đoán về >= 0. Dùng cho mọi model dự báo `future_value`
    (Linear Regression, GBM) — ground truth future_value LUÔN >= 0 (tổng
    revenue không âm), nên dự đoán âm chắc chắn là lỗi model, không phải tín
    hiệu hợp lệ. Nếu không clip, 1 khách hàng có predicted_future_value < 0
    có thể làm incremental_margin_i = v*margin_pct*bracket bị LẬT DẤU thành
    dương giả khi bracket (1+r)(1-d)-1 cũng âm (vd segment Champions ở
    discount thấp) — đã phát hiện bug này thật trên dữ liệu (customer 17850,
    xem tests/test_decision_scenario.py)."""
    return np.clip(predictions, 0, None)


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
