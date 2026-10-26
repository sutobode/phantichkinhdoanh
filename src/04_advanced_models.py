"""Optional GBM comparison on the SAME expanding temporal design as the baseline.

Keep calibration, SHAP and segment/country diagnostics. Final test outcomes
are never used for fitting or choosing the policy prediction source.
Run: python src/04_advanced_models.py (after step 02).
"""
from __future__ import annotations

from datetime import datetime, timezone
from importlib import import_module

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from utils import PROJECT_ROOT, get_logger, load_config, outputs_path, processed_path, set_seed

logger = get_logger("04_advanced_models")
baseline = import_module("03_baseline_models")
FEATURE_COLS = baseline.FEATURE_COLS
MODELS_DIR = PROJECT_ROOT / "models"


def main() -> None:
    config = load_config()
    set_seed(config["seed"])
    snapshots = baseline.load_dataset(config)
    scores, metrics, gbm_clf, gbm_reg = baseline.evaluate_temporal_models(
        snapshots, config, GradientBoostingClassifier(random_state=config["seed"]),
        GradientBoostingRegressor(random_state=config["seed"]), source="advanced",
    )
    log_scores, log_metrics, _, _ = baseline.evaluate_temporal_models(
        snapshots, config, *baseline.baseline_models(config),
    )
    pd.DataFrame([
        {"split": role, "model": name, "pr_auc": result[role]["pr_auc"][0],
         "roc_auc": result[role]["roc_auc"][0], "mae": result[role]["mae"][0],
         "rmse": result[role]["rmse"][0]}
        for role in ("validation", "final_test")
        for name, result in (("baseline", log_metrics), ("advanced", metrics))
    ]).to_csv(outputs_path("temporal_model_comparison.csv"), index=False)

    X_test = scores[FEATURE_COLS]
    proba_gbm = scores["predicted_repeat_purchase_proba"].to_numpy()
    proba_logreg = log_scores["predicted_repeat_purchase_proba"].to_numpy()
    frac_pos_log, mean_pred_log = calibration_curve(scores["repeat_purchase"], proba_logreg, n_bins=10)
    frac_pos_gbm, mean_pred_gbm = calibration_curve(scores["repeat_purchase"], proba_gbm, n_bins=10)
    pd.DataFrame({"mean_predicted_logreg": mean_pred_log, "fraction_positive_logreg": frac_pos_log}).to_csv(
        outputs_path("temporal_calibration_logreg.csv"), index=False)
    pd.DataFrame({"mean_predicted_gbm": mean_pred_gbm, "fraction_positive_gbm": frac_pos_gbm}).to_csv(
        outputs_path("temporal_calibration_gbm.csv"), index=False)
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
        ax.set_title("Final temporal test — repeat purchase")
        ax.legend()
        fig.savefig(outputs_path("figures/temporal_calibration_curve.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)
    except ImportError as exc:
        logger.warning("Calibration plot unavailable: %s", exc)

    try:
        import shap
        shap_values = shap.TreeExplainer(gbm_clf).shap_values(X_test)
        shap_df = pd.DataFrame({"feature": FEATURE_COLS, "mean_abs_shap": np.abs(shap_values).mean(axis=0)})
        shap_df.sort_values("mean_abs_shap", ascending=False).to_csv(
            outputs_path("temporal_shap_feature_importance.csv"), index=False)
    except ImportError as exc:
        logger.warning("SHAP unavailable: %s", exc)

    # Use the same final-snapshot RFM definitions as the policy, not new test-subset bins.
    diagnostics = scores.assign(abs_error_class=np.abs(scores["repeat_purchase"] - proba_gbm))
    diagnostics.groupby("rfm_segment")["abs_error_class"].mean().sort_values(ascending=False).to_csv(
        outputs_path("temporal_error_by_segment.csv"))
    country_counts = diagnostics["country"].value_counts()
    small = country_counts[country_counts < 20].index
    diagnostics["country_group"] = diagnostics["country"].where(
        ~diagnostics["country"].isin(small), "Other (small country, n<20)")
    bias = diagnostics.groupby("country_group").agg(
        n_customers=("abs_error_class", "size"), mean_abs_error=("abs_error_class", "mean"),
        mean_predicted_proba=("predicted_repeat_purchase_proba", "mean"),
        actual_repeat_purchase_rate=("repeat_purchase", "mean"),
    )
    bias["error_vs_overall_pct"] = 100 * (bias["mean_abs_error"] / diagnostics["abs_error_class"].mean() - 1)
    bias.to_csv(outputs_path("temporal_ethics_bias_check_by_country.csv"))
    if (bias["error_vs_overall_pct"].abs() > 30).any():
        logger.warning("Country-group probability error differs by >30%%; inspect the bias table")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(gbm_clf, MODELS_DIR / "gbm_classifier_v1.joblib")
    joblib.dump(gbm_reg, MODELS_DIR / "gbm_regressor_v1.joblib")
    baseline.register_run({
        "run_id": f"advanced_temporal_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "git_commit": baseline.git_commit_hash(), "validation": config["validation"],
        "models": ["gradient_boosting_classifier", "gradient_boosting_regressor"], "metrics": metrics,
    })
    scores.to_csv(processed_path("customer_test_predictions_advanced.csv", config), index=False)


if __name__ == "__main__":
    main()
