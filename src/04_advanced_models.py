"""Phase 3 advanced benchmark on the frozen temporal and business design.

Model and calibration choices use validation information only. The final test is
evaluation-only, and the secondary rolling origins do not replace the main split.
Run: python src/04_advanced_models.py (after steps 02 and 03).
"""
from __future__ import annotations

from datetime import datetime, timezone
from importlib import import_module
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    median_absolute_error,
    roc_auc_score,
    root_mean_squared_error,
)

from utils import PROJECT_ROOT, get_logger, load_config, outputs_path, processed_path, set_seed

logger = get_logger("04_advanced_models")
baseline = import_module("03_baseline_models")
cleaning = import_module("02_cleaning_features")
FEATURE_COLS = baseline.FEATURE_COLS
MODELS_DIR = PROJECT_ROOT / "models"

COMPARISON_COLUMNS = [
    "split", "model", "classifier", "regressor", "n_train", "n_evaluation",
    "pr_auc", "roc_auc", "brier_score", "mae", "rmse", "median_absolute_error",
    "p90_absolute_error", "fit_seconds", "inference_ms_per_1000",
]
ROLLING_COLUMNS = [
    "decision_date", "n_train", "n_evaluation", "model", "task", "metric", "value",
]


def advanced_models(config: dict):
    """One predeclared nonlinear candidate per task; no final-test tuning."""
    seed = config["seed"]
    return GradientBoostingClassifier(random_state=seed), GradientBoostingRegressor(random_state=seed)


def _check_fit_precedes_evaluation(fit_df: pd.DataFrame, eval_df: pd.DataFrame) -> None:
    if pd.to_datetime(fit_df["outcome_end_exclusive"]).max() >= pd.to_datetime(
        eval_df["decision_date"]
    ).min():
        raise ValueError("Training labels are not fully observed before evaluation")


def _metrics(y_class, proba, y_value, value) -> dict:
    absolute_error = np.abs(np.asarray(y_value) - np.asarray(value))
    return {
        "pr_auc": average_precision_score(y_class, proba),
        "roc_auc": roc_auc_score(y_class, proba),
        "brier_score": brier_score_loss(y_class, proba),
        "mae": mean_absolute_error(y_value, value),
        "rmse": root_mean_squared_error(y_value, value),
        "median_absolute_error": median_absolute_error(y_value, value),
        "p90_absolute_error": float(np.quantile(absolute_error, 0.90)),
    }


def evaluate_model_comparison(snapshots: pd.DataFrame, config: dict):
    """Evaluate baseline and advanced pairs; return only out-of-time predictions."""
    train, validation, test = cleaning.temporal_partitions(snapshots, config)
    candidates = {
        "baseline": (*baseline.baseline_models(config), "logistic_regression", "linear_regression"),
        "advanced": (*advanced_models(config), "gradient_boosting_classifier", "gradient_boosting_regressor"),
    }
    records, predictions, final_models = [], {}, {}
    for split, fit_df, eval_df in (
        ("validation", train, validation),
        ("final_test", pd.concat([train, validation], ignore_index=True), test),
    ):
        _check_fit_precedes_evaluation(fit_df, eval_df)
        for model, (classifier, regressor, classifier_name, regressor_name) in candidates.items():
            clf, reg = clone(classifier), clone(regressor)
            start = perf_counter()
            clf.fit(fit_df[FEATURE_COLS], fit_df["repeat_purchase"])
            reg.fit(fit_df[FEATURE_COLS], fit_df["future_gross_sales"])
            fit_seconds = perf_counter() - start
            start = perf_counter()
            proba = clf.predict_proba(eval_df[FEATURE_COLS])[:, 1]
            raw_value = reg.predict(eval_df[FEATURE_COLS])
            inference_seconds = perf_counter() - start
            value = np.clip(raw_value, 0, None)
            record = {
                "split": split, "model": model, "classifier": classifier_name,
                "regressor": regressor_name, "n_train": len(fit_df),
                "n_evaluation": len(eval_df),
                "fit_seconds": fit_seconds,
                "inference_ms_per_1000": 1000 * inference_seconds / len(eval_df) * 1000,
                **_metrics(eval_df["repeat_purchase"], proba,
                           eval_df["future_gross_sales"], value),
            }
            records.append(record)
            predictions[(split, model)] = pd.DataFrame({
                "Customer ID": eval_df["Customer ID"].to_numpy(),
                "repeat_purchase": eval_df["repeat_purchase"].to_numpy(),
                "future_gross_sales": eval_df["future_gross_sales"].to_numpy(),
                "predicted_repeat_purchase_proba": proba,
                "predicted_future_value_raw": raw_value,
                "predicted_future_value": value,
            })
            if split == "final_test":
                final_models[model] = (clf, reg)
    return pd.DataFrame(records, columns=COMPARISON_COLUMNS), predictions, final_models


def select_models(comparison: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Apply predeclared rules exclusively to validation rows."""
    settings = config["advanced_modeling"]
    validation = comparison[comparison["split"].eq("validation")].set_index("model")
    if not {"baseline", "advanced"}.issubset(validation.index):
        raise ValueError("Baseline and advanced validation rows are required")
    base, advanced = validation.loc["baseline"], validation.loc["advanced"]
    pr_gain = advanced["pr_auc"] - base["pr_auc"]
    brier_increase = advanced["brier_score"] - base["brier_score"]
    keep_advanced_classifier = (
        pr_gain >= settings["classification_min_pr_auc_gain"]
        and brier_increase <= settings["classification_max_brier_increase"]
    )
    relative_mae_gain = (base["mae"] - advanced["mae"]) / base["mae"]
    p90_increase = advanced["p90_absolute_error"] / base["p90_absolute_error"] - 1
    keep_advanced_regressor = (
        relative_mae_gain >= settings["regression_min_relative_mae_gain"]
        and p90_increase <= settings["regression_max_p90_error_increase"]
    )
    return pd.DataFrame([
        {
            "task": "classification",
            "selected_model": "gradient_boosting_classifier" if keep_advanced_classifier else "logistic_regression",
            "validation_gain": pr_gain,
            "primary_metric": "pr_auc_absolute_gain",
            "threshold": settings["classification_min_pr_auc_gain"],
            "guardrail_metric": "brier_score_increase",
            "guardrail_value": brier_increase,
            "guardrail_threshold": settings["classification_max_brier_increase"],
        },
        {
            "task": "regression",
            "selected_model": "gradient_boosting_regressor" if keep_advanced_regressor else "linear_regression",
            "validation_gain": relative_mae_gain,
            "primary_metric": "relative_mae_gain",
            "threshold": settings["regression_min_relative_mae_gain"],
            "guardrail_metric": "p90_absolute_error_relative_increase",
            "guardrail_value": p90_increase,
            "guardrail_threshold": settings["regression_max_p90_error_increase"],
        },
    ])


def fit_sigmoid_calibrator(base_classifier, fit_df: pd.DataFrame,
                           calibration_df: pd.DataFrame, evaluation_df: pd.DataFrame,
                           seed: int = 42):
    """Fit base then Platt-style sigmoid in strict temporal order."""
    fit_end = pd.to_datetime(fit_df["outcome_end_exclusive"]).max()
    calibration_date = pd.to_datetime(calibration_df["decision_date"]).min()
    calibration_end = pd.to_datetime(calibration_df["outcome_end_exclusive"]).max()
    evaluation_date = pd.to_datetime(evaluation_df["decision_date"]).min()
    if fit_end > calibration_date or calibration_end >= evaluation_date:
        raise ValueError("Calibration labels must be observed before evaluation")
    fitted_base = clone(base_classifier).fit(fit_df[FEATURE_COLS], fit_df["repeat_purchase"])
    calibration_score = fitted_base.decision_function(calibration_df[FEATURE_COLS]).reshape(-1, 1)
    sigmoid = LogisticRegression(random_state=seed).fit(
        calibration_score, calibration_df["repeat_purchase"]
    )
    evaluation_score = fitted_base.decision_function(evaluation_df[FEATURE_COLS]).reshape(-1, 1)
    return sigmoid.predict_proba(evaluation_score)[:, 1], fitted_base, sigmoid


def calibration_table(y_true, proba, model: str, split: str, n_bins: int) -> pd.DataFrame:
    """Ten equal-width bins; empty bins are omitted and ECE weights observed bins."""
    frame = pd.DataFrame({"actual": np.asarray(y_true), "probability": np.asarray(proba)})
    frame["bin"] = pd.cut(frame["probability"], np.linspace(0, 1, n_bins + 1),
                          labels=False, include_lowest=True)
    table = frame.groupby("bin", observed=True).agg(
        n=("actual", "size"), mean_predicted_probability=("probability", "mean"),
        observed_positive_rate=("actual", "mean"),
    ).reset_index()
    table.insert(0, "split", split)
    table.insert(0, "model", model)
    return table


def expected_calibration_error(table: pd.DataFrame) -> float:
    return float(np.average(
        np.abs(table["observed_positive_rate"] - table["mean_predicted_probability"]),
        weights=table["n"],
    ))


def evaluate_calibration(snapshots: pd.DataFrame, config: dict, comparison: pd.DataFrame,
                         predictions: dict):
    train, validation, test = cleaning.temporal_partitions(snapshots, config)
    train_dates = sorted(pd.to_datetime(train["decision_date"]).unique())
    calibration_fit = train[pd.to_datetime(train["decision_date"]).eq(train_dates[0])]
    calibration_train = train[pd.to_datetime(train["decision_date"]).eq(train_dates[1])]
    classifier = baseline.baseline_models(config)[0]
    calibrated_validation, _, _ = fit_sigmoid_calibrator(
        classifier, calibration_fit, calibration_train, validation, config["seed"]
    )
    calibrated_test, _, _ = fit_sigmoid_calibrator(
        classifier, train, validation, test, config["seed"]
    )
    calibrated = {"validation": calibrated_validation, "final_test": calibrated_test}
    settings = config["advanced_modeling"]
    baseline_validation = comparison[
        comparison["split"].eq("validation") & comparison["model"].eq("baseline")
    ].iloc[0]
    validation_brier = brier_score_loss(validation["repeat_purchase"], calibrated_validation)
    validation_pr = average_precision_score(validation["repeat_purchase"], calibrated_validation)
    adopted = (
        baseline_validation["brier_score"] - validation_brier
        >= settings["calibration_min_brier_gain"]
        and baseline_validation["pr_auc"] - validation_pr
        <= settings["calibration_max_pr_auc_loss"]
    )
    metrics, tables = [], []
    split_frames = {"validation": validation, "final_test": test}
    for split, frame in split_frames.items():
        candidates = {
            "logistic_regression": predictions[(split, "baseline")]["predicted_repeat_purchase_proba"],
            "gradient_boosting_classifier": predictions[(split, "advanced")]["predicted_repeat_purchase_proba"],
            "logistic_regression_sigmoid": calibrated[split],
        }
        for model, proba in candidates.items():
            table = calibration_table(frame["repeat_purchase"], proba, model, split,
                                      settings["calibration_bins"])
            tables.append(table)
            metrics.append({
                "model": model, "split": split,
                "brier_score": brier_score_loss(frame["repeat_purchase"], proba),
                "ece_equal_width_10_bins": expected_calibration_error(table),
                "pr_auc": average_precision_score(frame["repeat_purchase"], proba),
                "roc_auc": roc_auc_score(frame["repeat_purchase"], proba),
                "calibration_retained": bool(adopted and model == "logistic_regression_sigmoid"),
            })
    return pd.DataFrame(metrics), pd.concat(tables, ignore_index=True), adopted


def deterministic_sample(frame: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    ordered = frame.sort_values("Customer ID").reset_index(drop=True)
    if len(ordered) <= sample_size:
        return ordered
    return ordered.sample(n=sample_size, random_state=seed).sort_values("Customer ID").reset_index(drop=True)


def write_shap_attribution(model, final_frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Predictive attribution for a deterministic final-test sample; never causal."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    sample = deterministic_sample(
        final_frame, config["advanced_modeling"]["attribution_sample_size"], config["seed"]
    )
    X_sample = sample[FEATURE_COLS]
    values = shap.TreeExplainer(model).shap_values(X_sample)
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]
    importance = pd.DataFrame({
        "model": "gradient_boosting_classifier", "task": "classification",
        "feature": FEATURE_COLS, "mean_abs_attribution": np.abs(values).mean(axis=0),
        "method": "tree_shap", "sample_size": len(sample),
    }).sort_values("mean_abs_attribution", ascending=False)
    importance.to_csv(outputs_path("advanced_feature_importance.csv"), index=False)
    shap.summary_plot(values, X_sample, max_display=len(FEATURE_COLS), show=False)
    plt.title("Advanced classifier predictive attribution (non-causal)")
    plt.savefig(outputs_path("figures/shap_summary.png"), dpi=120, bbox_inches="tight")
    plt.close()
    return importance


def rolling_origin_schedule(config: dict) -> pd.DataFrame:
    decisions = pd.to_datetime(config["advanced_modeling"]["rolling_origin_dates"])
    if len(decisions) < 3 or decisions.has_duplicates or not decisions.is_monotonic_increasing:
        raise ValueError("Rolling-origin dates must be unique and increasing")
    schedule = pd.DataFrame({"decision_date": decisions})
    schedule["outcome_end_exclusive"] = decisions + pd.Timedelta(
        days=config["validation"]["outcome_horizon_days"]
    )
    for origin in decisions[1:]:
        prior = schedule[schedule["decision_date"] < origin]
        if prior.empty or not (prior["outcome_end_exclusive"] < origin).any():
            raise ValueError("Each rolling evaluation needs fully observed earlier labels")
    return schedule


def build_rolling_snapshots(raw: pd.DataFrame, config: dict) -> pd.DataFrame:
    schedule = rolling_origin_schedule(config)
    if schedule["outcome_end_exclusive"].max() > raw["InvoiceDate"].max().normalize():
        raise ValueError("Raw data does not cover all rolling-origin outcomes")
    filtered = cleaning.filter_valid_customers(raw, config)
    return pd.concat([
        cleaning.build_snapshot(filtered, config, row.decision_date, "rolling_origin")
        for row in schedule.itertuples(index=False)
    ], ignore_index=True)


def eligible_rolling_training(snapshots: pd.DataFrame, decision_date) -> pd.DataFrame:
    decision_date = pd.Timestamp(decision_date)
    training = snapshots[
        (pd.to_datetime(snapshots["decision_date"]) < decision_date)
        & (pd.to_datetime(snapshots["outcome_end_exclusive"]) < decision_date)
    ].copy()
    if training.empty:
        raise ValueError("No fully observed rolling-origin training labels")
    return training


def evaluate_rolling_origins(snapshots: pd.DataFrame, config: dict) -> pd.DataFrame:
    records = []
    model_pairs = {
        "baseline": baseline.baseline_models(config),
        "advanced": advanced_models(config),
    }
    dates = sorted(pd.to_datetime(snapshots["decision_date"]).unique())
    for decision_date in dates[1:]:
        evaluation = snapshots[pd.to_datetime(snapshots["decision_date"]).eq(decision_date)]
        training = eligible_rolling_training(snapshots, decision_date)
        for family, (classifier, regressor) in model_pairs.items():
            clf = clone(classifier).fit(training[FEATURE_COLS], training["repeat_purchase"])
            reg = clone(regressor).fit(training[FEATURE_COLS], training["future_gross_sales"])
            proba = clf.predict_proba(evaluation[FEATURE_COLS])[:, 1]
            value = np.clip(reg.predict(evaluation[FEATURE_COLS]), 0, None)
            values = {
                ("classification", "pr_auc"): average_precision_score(evaluation["repeat_purchase"], proba),
                ("classification", "roc_auc"): roc_auc_score(evaluation["repeat_purchase"], proba),
                ("regression", "mae"): mean_absolute_error(evaluation["future_gross_sales"], value),
                ("regression", "rmse"): root_mean_squared_error(evaluation["future_gross_sales"], value),
            }
            for (task, metric), result in values.items():
                records.append({
                    "decision_date": pd.Timestamp(decision_date).date().isoformat(),
                    "n_train": len(training), "n_evaluation": len(evaluation),
                    "model": family, "task": task, "metric": metric, "value": result,
                })
    return pd.DataFrame(records, columns=ROLLING_COLUMNS)


def plot_calibration(table: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    final = table[table["split"].eq("final_test")]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    for model, group in final.groupby("model"):
        ax.plot(group["mean_predicted_probability"], group["observed_positive_rate"], "o-", label=model)
    ax.set(xlabel="Mean predicted probability", ylabel="Observed positive rate",
           title="Final temporal-test calibration")
    ax.legend(fontsize=8)
    fig.savefig(outputs_path("figures/temporal_calibration_curve.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_rolling(metrics: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, task, metric, title in (
        (axes[0], "classification", "pr_auc", "Repeat purchase PR-AUC"),
        (axes[1], "regression", "mae", "Future gross sales MAE"),
    ):
        view = metrics[metrics["task"].eq(task) & metrics["metric"].eq(metric)]
        for model, group in view.groupby("model"):
            ax.plot(pd.to_datetime(group["decision_date"]), group["value"], "o-", label=model)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35)
        ax.legend()
    fig.suptitle("Secondary rolling-origin robustness")
    fig.tight_layout()
    fig.savefig(outputs_path("figures/rolling_origin_performance.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    config = load_config()
    set_seed(config["seed"])
    snapshots = baseline.load_dataset(config)
    comparison, predictions, final_models = evaluate_model_comparison(snapshots, config)
    selection = select_models(comparison, config)
    comparison.to_csv(outputs_path("temporal_model_comparison.csv"), index=False)
    selection.to_csv(outputs_path("model_selection.csv"), index=False)

    calibration_metrics, calibration_bins, calibration_adopted = evaluate_calibration(
        snapshots, config, comparison, predictions
    )
    calibration_metrics.to_csv(outputs_path("calibration_metrics.csv"), index=False)
    calibration_bins.to_csv(outputs_path("calibration_table.csv"), index=False)
    plot_calibration(calibration_bins)

    _, _, final = cleaning.temporal_partitions(snapshots, config)
    advanced_predictions = predictions[("final_test", "advanced")]
    scores = final.copy()
    for column in ("predicted_repeat_purchase_proba", "predicted_future_value_raw",
                   "predicted_future_value"):
        scores[column] = advanced_predictions[column].to_numpy()
    scores["rfm_score"] = baseline.rfm_score(scores)
    scores["rfm_segment"] = baseline.rfm_segment(scores)
    scores["prediction_source"] = "advanced"
    scores["evaluation_design"] = "expanding_temporal_snapshots"
    train, validation, _ = cleaning.temporal_partitions(snapshots, config)
    scores["training_outcome_end_exclusive"] = pd.concat(
        [train, validation], ignore_index=True
    )["outcome_end_exclusive"].max()
    scores.to_csv(processed_path("customer_test_predictions_advanced.csv", config), index=False)

    importance = write_shap_attribution(final_models["advanced"][0], scores, config)
    raw = cleaning.load_raw(config)
    rolling_metrics = evaluate_rolling_origins(build_rolling_snapshots(raw, config), config)
    rolling_metrics.to_csv(outputs_path("rolling_origin_metrics.csv"), index=False)
    plot_rolling(rolling_metrics)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_models["advanced"][0], MODELS_DIR / "gbm_classifier_v1.joblib")
    joblib.dump(final_models["advanced"][1], MODELS_DIR / "gbm_regressor_v1.joblib")
    baseline.register_run({
        "run_id": f"advanced_temporal_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "git_commit": baseline.git_commit_hash(), "validation": config["validation"],
        "models": ["gradient_boosting_classifier", "gradient_boosting_regressor"],
        "metrics": comparison.to_dict(orient="records"),
        "selection": selection.to_dict(orient="records"),
        "calibration_retained": bool(calibration_adopted),
    })
    logger.info("Validation-only selection:\n%s", selection.to_string(index=False))
    logger.info("Top predictive attributions:\n%s", importance.head().to_string(index=False))
    logger.info("Rolling-origin evaluations: %d origins", rolling_metrics["decision_date"].nunique())


if __name__ == "__main__":
    main()
