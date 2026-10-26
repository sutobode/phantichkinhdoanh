"""Week 2 — Baseline: RFM segmentation + naive baseline + interpretable models.

Đọc customer_snapshots.csv; train on earlier fully observed outcomes, validate
on a later origin, then refit on train+validation before final temporal test:
  - naive baseline (tỷ lệ chung / trung bình lịch sử)
  - RFM quantile segmentation
  - logistic regression cho repeat_purchase
  - linear regression cho future_gross_sales

Ghi metric vào models/temporal_registry.json và final-test-only predictions
vào data/processed/customer_test_predictions.csv. No random customer split.

Chạy: python src/03_baseline_models.py
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    roc_auc_score,
    root_mean_squared_error,
)
from utils import PROJECT_ROOT, get_logger, load_config, outputs_path, processed_path, set_seed

logger = get_logger("03_baseline_models")

MODELS_DIR = PROJECT_ROOT / "models"
REGISTRY_PATH = MODELS_DIR / "temporal_registry.json"
cleaning = import_module("02_cleaning_features")


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT
        ).decode().strip()
    except Exception:
        return "unknown"


def load_dataset(config: dict) -> pd.DataFrame:
    return pd.read_csv(processed_path("customer_snapshots.csv", config))


def rfm_score(df: pd.DataFrame) -> pd.Series:
    """Quantile-based RFM segmentation (quartiles, dùng pd.qcut với rank để
    tránh lỗi khi nhiều giá trị trùng nhau)."""
    r_score = pd.qcut(df["recency_days"].rank(method="first"), 4, labels=[4, 3, 2, 1]).astype(int)
    f_score = pd.qcut(df["frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    m_score = pd.qcut(df["monetary"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    return r_score + f_score + m_score


def rfm_segment(df: pd.DataFrame) -> pd.Series:

    def label(score: int) -> str:
        if score >= 10:
            return "Champions"
        if score >= 8:
            return "Loyal"
        if score >= 6:
            return "At-risk"
        return "Lost"

    return rfm_score(df).apply(label)


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
            value = metric_fn(y_true[idx], y_score[idx])
            if np.isfinite(value):
                stats.append(value)
        except ValueError:
            continue
    point = metric_fn(y_true, y_score)
    lo, hi = np.percentile(stats, [2.5, 97.5]) if stats else (float("nan"), float("nan"))
    return point, lo, hi


def register_run(entry: dict) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    registry = []
    if REGISTRY_PATH.exists():
        registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    registry.append(entry)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    logger.info("Đã ghi registry entry: %s", entry["run_id"])


def baseline_models(config: dict):
    # Scaling is fitted inside each training fold, never on later snapshots.
    return (make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=config["seed"])),
            LinearRegression())


def evaluate_temporal_models(snapshots, config, classifier, regressor, source="baseline"):
    train, validation, test = cleaning.temporal_partitions(snapshots, config)
    results = {}
    for role, fit_df, eval_df in (
        ("validation", train, validation),
        ("final_test", pd.concat([train, validation], ignore_index=True), test),
    ):
        # This guard is also enforced by temporal_partitions; keep it adjacent
        # to fitting so that future edits cannot accidentally fit a test label.
        if fit_df["outcome_end_exclusive"].max() >= eval_df["decision_date"].min():
            raise ValueError("Training labels are not yet observable")
        clf, reg = clone(classifier), clone(regressor)
        clf.fit(fit_df[FEATURE_COLS], fit_df["repeat_purchase"])
        reg.fit(fit_df[FEATURE_COLS], fit_df["future_gross_sales"])
        proba = clf.predict_proba(eval_df[FEATURE_COLS])[:, 1]
        raw_value = reg.predict(eval_df[FEATURE_COLS])
        value = np.clip(raw_value, 0, None)
        metrics = {}
        for name, target, prediction, metric in (
            ("pr_auc", "repeat_purchase", proba, average_precision_score),
            ("roc_auc", "repeat_purchase", proba, roc_auc_score),
            ("mae", "future_gross_sales", value, mean_absolute_error),
            ("rmse", "future_gross_sales", value, root_mean_squared_error),
        ):
            metrics[name] = bootstrap_ci(eval_df[target].to_numpy(), prediction, metric,
                                         config["validation"]["n_bootstrap"], config["seed"])
        naive_proba = np.full(len(eval_df), fit_df["repeat_purchase"].mean())
        naive_value = np.full(len(eval_df), fit_df["future_gross_sales"].mean())
        metrics.update(naive_pr_auc=average_precision_score(eval_df["repeat_purchase"], naive_proba),
                       naive_mae=mean_absolute_error(eval_df["future_gross_sales"], naive_value),
                       naive_rmse=root_mean_squared_error(eval_df["future_gross_sales"], naive_value),
                       n_train=len(fit_df), n_evaluation=len(eval_df))
        results[role] = metrics
        logger.info("%s %s metrics: %s", source, role, metrics)
        scores = eval_df.copy()
        scores["rfm_score"] = rfm_score(scores)
        scores["rfm_segment"] = rfm_segment(scores)
        scores["predicted_repeat_purchase_proba"] = proba
        scores["predicted_future_value_raw"] = raw_value
        scores["predicted_future_value"] = value
        scores["prediction_source"] = source
        scores["evaluation_design"] = "expanding_temporal_snapshots"
        scores["training_outcome_end_exclusive"] = fit_df["outcome_end_exclusive"].max()
    # Only final temporal-test predictions leave this function.
    return scores, results, clf, reg


def main() -> None:
    config = load_config()
    set_seed(config["seed"])
    scores, metrics, _, _ = evaluate_temporal_models(load_dataset(config), config, *baseline_models(config))
    register_run({
        "run_id": f"baseline_temporal_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "git_commit": git_commit_hash(), "validation": config["validation"],
        "models": ["naive", "logistic_regression", "linear_regression"], "metrics": metrics,
    })
    scores.to_csv(processed_path("customer_test_predictions.csv", config), index=False)
    fraction, predicted = calibration_curve(scores["repeat_purchase"], scores["predicted_repeat_purchase_proba"], n_bins=10)
    pd.DataFrame({"mean_predicted_logreg": predicted, "fraction_positive_logreg": fraction}).to_csv(
        outputs_path("temporal_calibration_logreg.csv"), index=False)


if __name__ == "__main__":
    main()
