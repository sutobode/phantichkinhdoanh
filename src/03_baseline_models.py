"""Week 2 — Baseline: RFM segmentation + naive baseline + interpretable models.

Đọc customer_features.csv + customer_labels.csv (output của 02_cleaning_features.py),
chia temporal holdout, train:
  - naive baseline (tỷ lệ chung / trung bình lịch sử)
  - RFM quantile segmentation
  - logistic regression cho repeat_purchase
  - linear regression cho future_value

Ghi metric (kèm bootstrap CI) vào models/registry.json và
data/processed/customer_scores.csv.

Chạy: python src/03_baseline_models.py
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from utils import PROJECT_ROOT, get_logger, load_config, processed_path, set_seed

logger = get_logger("03_baseline_models")

MODELS_DIR = PROJECT_ROOT / "models"
REGISTRY_PATH = MODELS_DIR / "registry.json"


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT
        ).decode().strip()
    except Exception:
        return "unknown"


def load_dataset(config: dict) -> pd.DataFrame:
    features = pd.read_csv(processed_path("customer_features.csv", config))
    labels = pd.read_csv(processed_path("customer_labels.csv", config))
    df = features.merge(labels, on="Customer ID", how="inner")
    return df


def rfm_segment(df: pd.DataFrame) -> pd.Series:
    """Quantile-based RFM segmentation (quartiles, dùng pd.qcut với rank để
    tránh lỗi khi nhiều giá trị trùng nhau)."""
    r_score = pd.qcut(df["recency_days"].rank(method="first"), 4, labels=[4, 3, 2, 1]).astype(int)
    f_score = pd.qcut(df["frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    m_score = pd.qcut(df["monetary"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    rfm_sum = r_score + f_score + m_score

    def label(score: int) -> str:
        if score >= 10:
            return "Champions"
        if score >= 8:
            return "Loyal"
        if score >= 6:
            return "At-risk"
        return "Lost"

    return rfm_sum.apply(label)


FEATURE_COLS = [
    "recency_days", "frequency", "monetary",
    "avg_order_value", "n_distinct_products", "tenure_days", "return_rate",
]


def bootstrap_ci(y_true: np.ndarray, y_score: np.ndarray, metric_fn, n_boot: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            stats.append(metric_fn(y_true[idx], y_score[idx]))
        except ValueError:
            continue
    point = metric_fn(y_true, y_score)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return point, lo, hi


def register_run(entry: dict) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    registry = []
    if REGISTRY_PATH.exists():
        registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    registry.append(entry)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    logger.info("Đã ghi registry entry: %s", entry["run_id"])


def main() -> None:
    config = load_config()
    set_seed(config["seed"])

    df = load_dataset(config)
    df["rfm_segment"] = rfm_segment(df)
    logger.info("Phân phối RFM segment:\n%s", df["rfm_segment"].value_counts().to_string())

    # Lưu ý về validation design: tính temporal đã được đảm bảo ở tầng nhãn —
    # mọi feature chỉ dùng dữ liệu <= observation_end, mọi label chỉ dùng dữ liệu
    # trong outcome window (sau observation_end). Việc chia train/test dưới đây
    # chỉ là random split GIỮA CÁC KHÁCH HÀNG (cross-sectional), không phải random
    # split theo thời gian trong lịch sử giao dịch — nên không vi phạm rule
    # "no random split khi có leakage thời gian". Rolling-origin validation ở
    # Tuần 3 (04_advanced_models.py) sẽ lặp lại toàn bộ pipeline ở nhiều
    # observation_end khác nhau để kiểm tra model ổn định qua thời gian.
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=config["seed"], stratify=df["repeat_purchase"]
    )
    logger.info("Train/test split (random theo khách hàng): train=%d, test=%d",
                len(train_df), len(test_df))

    X_train, X_test = train_df[FEATURE_COLS], test_df[FEATURE_COLS]

    # --- Naive baseline ---
    naive_proba = np.full(len(test_df), train_df["repeat_purchase"].mean())
    naive_pr_auc = average_precision_score(test_df["repeat_purchase"], naive_proba)
    naive_mae = mean_absolute_error(test_df["future_value"], np.full(len(test_df), train_df["future_value"].mean()))
    logger.info("Naive baseline — PR-AUC: %.3f | MAE: %.2f", naive_pr_auc, naive_mae)

    # --- Logistic regression: repeat_purchase ---
    clf = LogisticRegression(max_iter=1000, random_state=config["seed"])
    clf.fit(X_train, train_df["repeat_purchase"])
    proba = clf.predict_proba(X_test)[:, 1]

    pr_auc, pr_lo, pr_hi = bootstrap_ci(
        test_df["repeat_purchase"].to_numpy(), proba, average_precision_score,
        config["validation"]["n_bootstrap"], config["seed"],
    )
    roc_auc, roc_lo, roc_hi = bootstrap_ci(
        test_df["repeat_purchase"].to_numpy(), proba, roc_auc_score,
        config["validation"]["n_bootstrap"], config["seed"],
    )
    logger.info("Logistic regression — PR-AUC: %.3f [%.3f, %.3f] | ROC-AUC: %.3f [%.3f, %.3f]",
                pr_auc, pr_lo, pr_hi, roc_auc, roc_lo, roc_hi)

    # --- Linear regression: future_value ---
    reg = LinearRegression()
    reg.fit(X_train, train_df["future_value"])
    pred_value = reg.predict(X_test)
    mae, mae_lo, mae_hi = bootstrap_ci(
        test_df["future_value"].to_numpy(), pred_value, mean_absolute_error,
        config["validation"]["n_bootstrap"], config["seed"],
    )
    logger.info("Linear regression — MAE: %.2f [%.2f, %.2f]", mae, mae_lo, mae_hi)

    # --- Registry log ---
    run_id = f"baseline_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    register_run({
        "run_id": run_id,
        "git_commit": git_commit_hash(),
        "models": ["naive", "logistic_regression", "linear_regression"],
        "metrics": {
            "naive_pr_auc": naive_pr_auc,
            "naive_mae": naive_mae,
            "logreg_pr_auc": [pr_auc, pr_lo, pr_hi],
            "logreg_roc_auc": [roc_auc, roc_lo, roc_hi],
            "linreg_mae": [mae, mae_lo, mae_hi],
        },
    })

    # --- customer_scores.csv (deliverable) ---
    full_proba = clf.predict_proba(df[FEATURE_COLS])[:, 1]
    full_value = reg.predict(df[FEATURE_COLS])
    scores = df[["Customer ID", "country", "rfm_segment"] + FEATURE_COLS].copy()
    scores["predicted_repeat_purchase_proba"] = full_proba
    scores["predicted_future_value"] = full_value
    scores_path = processed_path("customer_scores.csv", config)
    scores.to_csv(scores_path, index=False)
    logger.info("Đã ghi customer_scores.csv: %s", scores_path)


if __name__ == "__main__":
    main()
