"""Week 3 — Rolling-origin validation + Gradient Boosting + SHAP + calibration.

Mục đích:
  1. Rolling-origin validation: lặp lại toàn bộ pipeline (split -> clean ->
     feature -> label -> train -> eval) ở NHIỀU observation_end khác nhau để
     kiểm tra model có ổn định qua thời gian không — không chỉ đánh giá 1 lần
     chia duy nhất.
  2. Gradient Boosting (stretch) + so sánh với Logistic/Linear baseline.
  3. Calibration curve: logistic vs GBM — model nào đáng tin hơn khi dùng
     probability để ra quyết định targeting.
  4. SHAP feature attribution cho GBM.
  5. Error analysis theo RFM segment.

QUAN TRỌNG: mỗi cutoff trong rolling-origin validation build lại dataset từ
đầu bằng CHÍNH các hàm trong 02_cleaning_features.py (filter -> split ->
return_rate -> drop_invalid -> rfm -> labels), với observation_end khác nhau
nhưng CÙNG observation_start. Điều này đảm bảo mỗi cutoff vẫn tuân thủ đúng
nguyên tắc "chỉ dùng dữ liệu <= observation_end tính feature", không có
shortcut nào tái sử dụng feature đã tính sẵn từ cutoff khác (tránh leakage
chéo giữa các cutoff).

Chạy: python src/04_advanced_models.py
"""
from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from utils import PROJECT_ROOT, get_logger, load_config, merge_or_replace_columns, outputs_path, processed_path, set_seed

logger = get_logger("04_advanced_models")
cleaning = import_module("02_cleaning_features")

MODELS_DIR = PROJECT_ROOT / "models"
REGISTRY_PATH = MODELS_DIR / "registry.json"

FEATURE_COLS = [
    "recency_days", "frequency", "monetary",
    "avg_order_value", "n_distinct_products", "tenure_days", "return_rate",
]


def register_run(entry: dict) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    registry = []
    if REGISTRY_PATH.exists():
        registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    registry.append(entry)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    logger.info("Đã ghi registry entry: %s", entry["run_id"])


def build_dataset(raw_filtered: pd.DataFrame, obs_start: str, obs_end: str, outcome_end: str) -> pd.DataFrame:
    """Build lại toàn bộ feature+label từ đầu cho 1 cặp (obs_end, outcome_end)
    bất kỳ, dùng lại đúng các hàm đã test trong 02_cleaning_features.py."""
    window_cfg = {"windows": {
        "observation_start": obs_start, "observation_end": obs_end, "outcome_end": outcome_end,
    }}
    obs_raw, outcome_raw = cleaning.split_windows(raw_filtered, window_cfg)
    if len(obs_raw) == 0 or len(outcome_raw) == 0:
        raise ValueError(f"Cutoff {obs_end} cho ra window rỗng — kiểm tra lại config.")

    return_rate = cleaning.compute_return_rate(obs_raw)
    clean_cfg = {"cleaning": {
        "drop_cancelled_invoices": True, "drop_non_positive_quantity": True, "drop_non_positive_price": True,
    }}
    obs_clean = cleaning.drop_invalid_rows(obs_raw, clean_cfg, "observation")
    outcome_clean = cleaning.drop_invalid_rows(outcome_raw, clean_cfg, "outcome")

    as_of = pd.Timestamp(obs_end) + pd.Timedelta(days=1)
    features = cleaning.compute_rfm_features(obs_clean, return_rate, as_of)
    labels = cleaning.compute_labels(outcome_clean, features["Customer ID"])
    return features.merge(labels, on="Customer ID", how="inner")


def bootstrap_ci(y_true: np.ndarray, y_score: np.ndarray, metric_fn, n_boot: int, seed: int):
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


def rolling_origin_validation(raw_filtered: pd.DataFrame, config: dict) -> pd.DataFrame:
    obs_start = config["windows"]["observation_start"]
    horizon = config["validation"]["rolling_origin_outcome_horizon_days"]
    results = []

    for cutoff in config["validation"]["rolling_origin_cutoffs"]:
        outcome_end = (pd.Timestamp(cutoff) + pd.Timedelta(days=horizon)).strftime("%Y-%m-%d")
        df = build_dataset(raw_filtered, obs_start, cutoff, outcome_end)
        logger.info("[rolling cutoff=%s] n_customers=%d, outcome_end=%s", cutoff, len(df), outcome_end)

        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=config["seed"], stratify=df["repeat_purchase"]
        )
        X_train, X_test = train_df[FEATURE_COLS], test_df[FEATURE_COLS]

        clf = LogisticRegression(max_iter=1000, random_state=config["seed"])
        clf.fit(X_train, train_df["repeat_purchase"])
        proba_logreg = clf.predict_proba(X_test)[:, 1]

        gbm = GradientBoostingClassifier(random_state=config["seed"])
        gbm.fit(X_train, train_df["repeat_purchase"])
        proba_gbm = gbm.predict_proba(X_test)[:, 1]

        pr_auc_logreg = average_precision_score(test_df["repeat_purchase"], proba_logreg)
        pr_auc_gbm = average_precision_score(test_df["repeat_purchase"], proba_gbm)

        results.append({
            "cutoff": cutoff, "outcome_end": outcome_end, "n_customers": len(df),
            "pr_auc_logreg": pr_auc_logreg, "pr_auc_gbm": pr_auc_gbm,
        })
        logger.info("[rolling cutoff=%s] PR-AUC logreg=%.3f | PR-AUC gbm=%.3f",
                    cutoff, pr_auc_logreg, pr_auc_gbm)

    result_df = pd.DataFrame(results)
    logger.info(
        "Rolling-origin stability — logreg PR-AUC: mean=%.3f std=%.3f | gbm PR-AUC: mean=%.3f std=%.3f",
        result_df["pr_auc_logreg"].mean(), result_df["pr_auc_logreg"].std(),
        result_df["pr_auc_gbm"].mean(), result_df["pr_auc_gbm"].std(),
    )
    return result_df


def main() -> None:
    config = load_config()
    set_seed(config["seed"])

    # --- Rolling-origin validation (nhiều cutoff, build dataset riêng mỗi lần) ---
    raw = cleaning.load_raw(config)
    raw_filtered = cleaning.filter_valid_customers(raw, config)
    rolling_results = rolling_origin_validation(raw_filtered, config)
    rolling_path = outputs_path("rolling_origin_validation.csv")
    rolling_results.to_csv(rolling_path, index=False)
    logger.info("Đã ghi rolling-origin results: %s", rolling_path)

    # --- Main dataset (cutoff chính thức trong config) — build lại từ đầu để
    # double-check khớp với customer_features.csv đã ghi ở bước 02 ---
    df = build_dataset(
        raw_filtered,
        config["windows"]["observation_start"],
        config["windows"]["observation_end"],
        config["windows"]["outcome_end"],
    )
    saved_features = pd.read_csv(processed_path("customer_features.csv", config))
    assert len(df) == len(saved_features), (
        "Số khách hàng build lại khác với customer_features.csv đã lưu ở bước 02 "
        "— có thể config đã đổi mà chưa chạy lại 02_cleaning_features.py"
    )
    logger.info("Double-check: build lại dataset chính khớp với bước 02 (%d khách hàng)", len(df))

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=config["seed"], stratify=df["repeat_purchase"]
    )
    X_train, X_test = train_df[FEATURE_COLS], test_df[FEATURE_COLS]

    # --- Logistic (để so sánh calibration) ---
    logreg = LogisticRegression(max_iter=1000, random_state=config["seed"])
    logreg.fit(X_train, train_df["repeat_purchase"])
    proba_logreg = logreg.predict_proba(X_test)[:, 1]

    # --- Gradient Boosting classifier + regressor ---
    gbm_clf = GradientBoostingClassifier(random_state=config["seed"])
    gbm_clf.fit(X_train, train_df["repeat_purchase"])
    proba_gbm = gbm_clf.predict_proba(X_test)[:, 1]

    gbm_reg = GradientBoostingRegressor(random_state=config["seed"])
    gbm_reg.fit(X_train, train_df["future_value"])
    pred_gbm_value = gbm_reg.predict(X_test)

    n_boot = config["validation"]["n_bootstrap"]
    pr_auc_gbm, pr_lo, pr_hi = bootstrap_ci(
        test_df["repeat_purchase"].to_numpy(), proba_gbm, average_precision_score, n_boot, config["seed"]
    )
    roc_auc_gbm, roc_lo, roc_hi = bootstrap_ci(
        test_df["repeat_purchase"].to_numpy(), proba_gbm, roc_auc_score, n_boot, config["seed"]
    )
    mae_gbm, mae_lo, mae_hi = bootstrap_ci(
        test_df["future_value"].to_numpy(), pred_gbm_value, mean_absolute_error, n_boot, config["seed"]
    )
    logger.info("GBM classifier — PR-AUC: %.3f [%.3f, %.3f] | ROC-AUC: %.3f [%.3f, %.3f]",
                pr_auc_gbm, pr_lo, pr_hi, roc_auc_gbm, roc_lo, roc_hi)
    logger.info("GBM regressor — MAE: %.2f [%.2f, %.2f]", mae_gbm, mae_lo, mae_hi)

    # --- Calibration: logistic vs GBM ---
    frac_pos_log, mean_pred_log = calibration_curve(test_df["repeat_purchase"], proba_logreg, n_bins=10)
    frac_pos_gbm, mean_pred_gbm = calibration_curve(test_df["repeat_purchase"], proba_gbm, n_bins=10)
    calib_df = pd.DataFrame({
        "mean_predicted_logreg": mean_pred_log, "fraction_positive_logreg": frac_pos_log,
    })
    calib_gbm_df = pd.DataFrame({
        "mean_predicted_gbm": mean_pred_gbm, "fraction_positive_gbm": frac_pos_gbm,
    })
    calib_path = outputs_path("calibration_logreg.csv")
    calib_gbm_path = outputs_path("calibration_gbm.csv")
    calib_df.to_csv(calib_path, index=False)
    calib_gbm_df.to_csv(calib_gbm_path, index=False)
    logger.info("Đã ghi calibration curve data: %s, %s", calib_path, calib_gbm_path)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
        ax.plot(mean_pred_log, frac_pos_log, "o-", label="Logistic Regression")
        ax.plot(mean_pred_gbm, frac_pos_gbm, "o-", label="Gradient Boosting")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title("Calibration curve — repeat_purchase")
        ax.legend()
        fig_path = PROJECT_ROOT / "outputs" / "figures" / "calibration_curve.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info("Đã ghi calibration curve chart: %s", fig_path)
    except Exception as e:
        logger.warning("Không vẽ được calibration chart: %s", e)

    # --- SHAP feature attribution cho GBM classifier ---
    try:
        import shap
        explainer = shap.TreeExplainer(gbm_clf)
        shap_values = explainer.shap_values(X_test)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({"feature": FEATURE_COLS, "mean_abs_shap": mean_abs_shap})
        shap_df = shap_df.sort_values("mean_abs_shap", ascending=False)
        shap_path = outputs_path("shap_feature_importance.csv")
        shap_df.to_csv(shap_path, index=False)
        logger.info("SHAP feature importance (GBM classifier):\n%s", shap_df.to_string(index=False))
    except Exception as e:
        logger.warning("Không tính được SHAP: %s", e)

    # --- Error analysis theo RFM segment ---
    test_df = test_df.copy()
    test_df["pred_proba"] = proba_gbm
    test_df["abs_error_class"] = np.abs(test_df["repeat_purchase"] - test_df["pred_proba"])

    baseline = import_module("03_baseline_models")
    test_df["rfm_segment"] = baseline.rfm_segment(test_df)
    error_by_segment = test_df.groupby("rfm_segment")["abs_error_class"].mean().sort_values(ascending=False)
    logger.info("Error analysis (mean abs error của GBM) theo RFM segment:\n%s", error_by_segment.to_string())
    error_by_segment.to_csv(outputs_path("error_by_segment.csv"))

    # --- Ethics/bias check: model có lỗi bất thường cao hơn ở nhóm quốc gia
    # nào không? (dùng ground truth thật từ test set, không chỉ liệt kê rủi ro
    # chung chung). Gộp quốc gia có <20 khách hàng trong test set vào "Other
    # (small country, n<20)" để tránh kết luận sai do mẫu quá nhỏ. ---
    country_counts = test_df["country"].value_counts()
    small_countries = country_counts[country_counts < 20].index
    test_df["country_group"] = test_df["country"].where(~test_df["country"].isin(small_countries),
                                                          "Other (small country, n<20)")
    bias_by_country = test_df.groupby("country_group").agg(
        n_customers=("abs_error_class", "size"),
        mean_abs_error=("abs_error_class", "mean"),
        mean_predicted_proba=("pred_proba", "mean"),
        actual_repeat_purchase_rate=("repeat_purchase", "mean"),
    ).sort_values("mean_abs_error", ascending=False)
    overall_mean_error = test_df["abs_error_class"].mean()
    bias_by_country["error_vs_overall_pct"] = 100 * (bias_by_country["mean_abs_error"] / overall_mean_error - 1)
    logger.info("Ethics/bias check — error theo country_group (overall mean error=%.3f):\n%s",
                overall_mean_error, bias_by_country.to_string())
    bias_path = outputs_path("ethics_bias_check_by_country.csv")
    bias_by_country.to_csv(bias_path)
    logger.info("Đã ghi ethics/bias check: %s", bias_path)

    worst = bias_by_country["error_vs_overall_pct"].drop("Other (small country, n<20)", errors="ignore")
    if (worst.abs() > 30).any():
        logger.warning(
            "⚠️ CẢNH BÁO BIAS: có country_group lệch >30%% so với mean_abs_error chung "
            "— xem docs/model_card.md mục Ethics/Bias trước khi dùng model cho quốc gia đó."
        )

    # --- Save models + registry ---
    joblib.dump(gbm_clf, MODELS_DIR / "gbm_classifier_v1.joblib")
    joblib.dump(gbm_reg, MODELS_DIR / "gbm_regressor_v1.joblib")
    register_run({
        "run_id": "advanced_gbm_v1",
        "models": ["gradient_boosting_classifier", "gradient_boosting_regressor"],
        "metrics": {
            "gbm_pr_auc": [pr_auc_gbm, pr_lo, pr_hi],
            "gbm_roc_auc": [roc_auc_gbm, roc_lo, roc_hi],
            "gbm_mae": [mae_gbm, mae_lo, mae_hi],
            "rolling_origin_pr_auc_logreg_mean_std": [
                float(rolling_results["pr_auc_logreg"].mean()), float(rolling_results["pr_auc_logreg"].std())
            ],
            "rolling_origin_pr_auc_gbm_mean_std": [
                float(rolling_results["pr_auc_gbm"].mean()), float(rolling_results["pr_auc_gbm"].std())
            ],
        },
    })

    # --- Cập nhật customer_scores.csv với cột GBM để so sánh (idempotent —
    # xem utils.merge_or_replace_columns) ---
    scores_path = processed_path("customer_scores.csv", config)
    scores = pd.read_csv(scores_path)

    full_proba_gbm = gbm_clf.predict_proba(df[FEATURE_COLS])[:, 1]
    full_value_gbm = gbm_reg.predict(df[FEATURE_COLS])
    gbm_scores = pd.DataFrame({
        "Customer ID": df["Customer ID"],
        "predicted_repeat_purchase_proba_gbm": full_proba_gbm,
        "predicted_future_value_gbm": full_value_gbm,
    })
    scores = merge_or_replace_columns(scores, gbm_scores, on="Customer ID")
    scores.to_csv(scores_path, index=False)
    logger.info("Đã cập nhật customer_scores.csv với cột GBM: %s", scores_path)


if __name__ == "__main__":
    main()
