"""Phase 4 secondary sensitivity to exact-duplicate handling.

The primary pipeline continues to retain exact duplicate transaction rows. This
script evaluates a separately labeled drop-exact-duplicates path in memory and
writes only namespaced robustness artifacts.

Run: python src/07_duplicate_robustness.py (after steps 02 and 03).
"""
from __future__ import annotations

from importlib import import_module

import pandas as pd

from utils import get_logger, load_config, outputs_path, set_seed

logger = get_logger("07_duplicate_robustness")
cleaning = import_module("02_cleaning_features")
baseline = import_module("03_baseline_models")
decision = import_module("05_decision_scenario")

RETAIN = "retain"
DROP = "drop_exact_duplicates"
ANALYSIS_TYPE = "duplicate_policy_sensitivity_non_causal"
ROBUSTNESS_OUTPUTS = {
    "duplicate_robustness_summary.csv",
    "duplicate_robustness_business_metrics.csv",
}
PRIMARY_OUTPUTS = {
    "data_quality.csv", "temporal_schedule.csv", "scenario_table.csv",
    "break_even_by_segment.csv", "break_even_summary.csv",
    "response_discount_surface.csv",
}


def apply_duplicate_policy(raw: pd.DataFrame, policy: str) -> pd.DataFrame:
    """Return a separate frame; never mutate or silently replace primary data."""
    if policy == RETAIN:
        return raw.copy()
    if policy == DROP:
        return raw.drop_duplicates(keep="first").reset_index(drop=True)
    raise ValueError(f"Unsupported duplicate policy: {policy}")


def recommendation_from_proposed(proposed: pd.DataFrame) -> str:
    if proposed["customers_targeted"].eq(0).all() and proposed[
        "simulated_net_incremental_margin"
    ].eq(0).all():
        return "do_not_launch_configured_campaigns"
    return "campaign_available_under_at_least_one_scenario"


def evaluate_duplicate_policy(raw: pd.DataFrame, main_snapshots: pd.DataFrame,
                              config: dict, policy: str):
    handled = apply_duplicate_policy(raw, policy)
    snapshots = main_snapshots.copy() if policy == RETAIN else cleaning.build_temporal_snapshots(
        handled, config
    )
    scores, metrics, _, _ = baseline.evaluate_temporal_models(
        snapshots, config, *baseline.baseline_models(config), source="baseline"
    )
    scenario_table = decision.build_scenario_table(scores, config)
    proposed = scenario_table[scenario_table["policy"].eq("proposed")]
    recommendation = recommendation_from_proposed(proposed)
    final_metrics = metrics["final_test"]
    summary = {
        "analysis_type": ANALYSIS_TYPE,
        "duplicate_policy": policy,
        "is_primary_analysis": policy == RETAIN,
        "raw_rows": len(raw),
        "rows_after_duplicate_handling": len(handled),
        "exact_duplicate_rows_removed": len(raw) - len(handled),
        "final_test_decision_date": config["validation"]["final_test_decision_date"],
        "outcome_horizon_days": config["validation"]["outcome_horizon_days"],
        "final_test_customers": len(scores),
        "repeat_purchase_rate": scores["repeat_purchase"].mean(),
        "logistic_pr_auc": final_metrics["pr_auc"][0],
        "logistic_roc_auc": final_metrics["roc_auc"][0],
        "linear_mae": final_metrics["mae"][0],
        "linear_rmse": final_metrics["rmse"][0],
        "campaign_recommendation": recommendation,
    }
    business = scenario_table[scenario_table["policy"].isin(["non_targeted", "rfm"])].pivot(
        index="scenario", columns="policy", values="simulated_net_incremental_margin"
    ).reset_index().rename(columns={
        "non_targeted": "non_targeted_monte_carlo_mean_net_margin",
        "rfm": "rfm_net_margin",
    }).merge(
        proposed[["scenario", "customers_targeted", "simulated_net_incremental_margin"]].rename(
            columns={
                "customers_targeted": "proposed_contacts",
                "simulated_net_incremental_margin": "proposed_net_margin",
            }
        ),
        on="scenario",
    )
    business.insert(0, "duplicate_policy", policy)
    business.insert(0, "analysis_type", ANALYSIS_TYPE)
    business["campaign_recommendation"] = recommendation
    order = {item["name"]: index for index, item in enumerate(
        config["decision"]["discount_response_scenarios"]
    )}
    business = business.sort_values(
        "scenario", key=lambda values: values.map(order)
    ).reset_index(drop=True)
    return summary, business


def build_duplicate_robustness(raw: pd.DataFrame, main_snapshots: pd.DataFrame,
                               config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    if config["cleaning"].get("exact_duplicate_policy") != RETAIN:
        raise ValueError("Primary exact-duplicate policy must remain retain")
    if config["robustness"].get("duplicate_policy") != DROP:
        raise ValueError("Robustness duplicate policy must be drop_exact_duplicates")
    summaries, businesses = [], []
    for policy in (RETAIN, DROP):
        summary, business = evaluate_duplicate_policy(raw, main_snapshots, config, policy)
        summaries.append(summary)
        businesses.append(business)
    summary = pd.DataFrame(summaries)
    retain_recommendation = summary.loc[
        summary["duplicate_policy"].eq(RETAIN), "campaign_recommendation"
    ].iloc[0]
    summary["qualitatively_matches_retain"] = summary["campaign_recommendation"].eq(
        retain_recommendation
    )
    business = pd.concat(businesses, ignore_index=True)
    business["recommendation_changed_vs_retain"] = ~business["campaign_recommendation"].eq(
        retain_recommendation
    )
    return summary, business


def write_robustness_outputs(summary: pd.DataFrame, business: pd.DataFrame) -> None:
    if ROBUSTNESS_OUTPUTS & PRIMARY_OUTPUTS:
        raise RuntimeError("Robustness outputs must not overlap primary outputs")
    summary.to_csv(outputs_path("duplicate_robustness_summary.csv"), index=False)
    business.to_csv(outputs_path("duplicate_robustness_business_metrics.csv"), index=False)


def main() -> None:
    config = load_config()
    set_seed(config["seed"])
    raw = cleaning.load_raw(config)
    main_snapshots = baseline.load_dataset(config)
    summary, business = build_duplicate_robustness(raw, main_snapshots, config)
    write_robustness_outputs(summary, business)
    logger.info("Duplicate-policy robustness summary:\n%s", summary.to_string(index=False))
    logger.info("Primary outputs were not overwritten; wrote only: %s", sorted(ROBUSTNESS_OUTPUTS))


if __name__ == "__main__":
    main()
