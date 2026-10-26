"""Week 4 — Decision Logic & Business Impact.

Thiết kế promotion-targeting rule, mô phỏng expected incremental margin dưới
nhiều giả định discount/response-rate, budget-constrained optimization, và
customer-level recommendation layer (retain/nurture/no-offer).

⚠️ QUAN TRỌNG (Scope Guardrail của đề): dataset KHÔNG có randomized
treatment/control, nên MỌI con số "incremental margin" dưới đây là
SCENARIO-BASED ESTIMATE dựa trên giả định discount_pct/response_rate khai báo
tường minh trong config/params.yaml — KHÔNG phải causal uplift đo được từ dữ
liệu thật. Không được diễn giải các số liệu này như "revenue lift đã được xác
nhận" khi trình bày cho decision-maker.

Mô hình scenario (xem comment trong config/params.yaml để biết rationale):
  effective_response_rate_i = response_rate_scenario * segment_multiplier[segment_i]
  gross_incremental_margin_i = v_i * [(1 + effective_response_rate_i) * (g - d) - g]
  net_incremental_margin_i = gross_incremental_margin_i - contact_cost
  v_i = clipped predicted gross future sales under historical behavior.
  response_rate = assumed proportional pre-discount sales-volume growth, NOT
  response probability. Underlying unit product costs remain unchanged.
  All reported policies use FINAL TEMPORAL TEST predictions and equal contact N.

Output:
  - outputs/scenario_table.csv
  - outputs/tornado_sensitivity.csv (+ chart)
  - data/processed/customer_test_scores.csv (evaluation, not deployment/demo)

Chạy: python src/05_decision_scenario.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils import PROJECT_ROOT, get_logger, load_config, outputs_path, processed_path, set_seed, validate_final_test_predictions

logger = get_logger("05_decision_scenario")

VALUE_COL = "baseline_future_revenue"


def load_scores(config: dict) -> pd.DataFrame:
    source = config["decision"]["value_prediction_source"]
    filenames = {"baseline": "customer_test_predictions.csv", "advanced": "customer_test_predictions_advanced.csv"}
    if source not in filenames:
        raise ValueError("value_prediction_source must be baseline or advanced")
    return prepare_scores(pd.read_csv(processed_path(filenames[source], config)), config)


def prepare_scores(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    validate_final_test_predictions(df, config)
    df = df.sort_values("Customer ID", kind="stable").reset_index(drop=True).copy()
    if not df["prediction_source"].eq(config["decision"]["value_prediction_source"]).all():
        raise ValueError("Prediction source does not match the preselected policy model")
    values = df["predicted_future_value"]
    if not np.isfinite(values).all():
        raise ValueError("Non-finite future revenue prediction")
    df[VALUE_COL] = values.clip(lower=0)
    return df


def effective_response_rate(df: pd.DataFrame, base_response_rate: float, multipliers: dict) -> pd.Series:
    mult = df["rfm_segment"].map(multipliers)
    if mult.isna().any():
        missing = df.loc[mult.isna(), "rfm_segment"].unique()
        raise ValueError(f"segment_response_multiplier thiếu segment: {missing}")
    return base_response_rate * mult


def incremental_margin(v: pd.Series, eff_response_rate: pd.Series, discount_pct: float,
                       gross_margin_pct: float, contact_cost_per_customer: float = 0.0) -> pd.Series:
    """Fixed unit COGS: promoted revenue - promoted COGS - baseline margin - contact.

    A zero contact cost explicitly requests margin BEFORE contact expense.
    Defensive clipping also protects API callers from negative raw predictions.
    """
    if not np.isfinite(v).all():
        raise ValueError("Non-finite future revenue prediction")
    return v.clip(lower=0) * ((1 + eff_response_rate) * (gross_margin_pct - discount_pct) - gross_margin_pct) - contact_cost_per_customer


def budget_constrained_selection(margin: pd.Series, budget_total: float, cost_per_customer: float) -> pd.Series:
    """Chọn top-N theo incremental_margin giảm dần, N = floor(budget/cost),
    NHƯNG chỉ chọn khách hàng có margin > 0 (0/1 knapsack với cost đồng nhất:
    không chọn item có giá trị âm dù còn ngân sách — chọn top-N theo margin
    giảm dần trong tập margin>0 là lời giải TỐI ƯU, không phải xấp xỉ, vì mọi
    item có cùng cost)."""
    if cost_per_customer <= 0 or budget_total < 0:
        raise ValueError("Contact cost must be positive and campaign budget nonnegative")
    max_n = int(budget_total // cost_per_customer)
    eligible = margin[margin > 0].sort_values(ascending=False, kind="stable")
    selected_index = eligible.index[:max_n]
    is_selected = pd.Series(False, index=margin.index)
    is_selected.loc[selected_index] = True
    return is_selected


def build_scenario_table(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    df = prepare_scores(df, config)
    dec = config["decision"]
    rows = []
    for scenario in dec["discount_response_scenarios"]:
        eff_rr = effective_response_rate(df, scenario["response_rate"], dec["segment_response_multiplier"])
        gross = incremental_margin(df[VALUE_COL], eff_rr, scenario["discount_pct"], dec["gross_margin_pct"])
        net = gross - dec["contact_cost_per_customer"]
        policies = policy_selections(df, net, config)
        random_total = float(net[policies["non_targeted"]].sum())
        for policy, selected in policies.items():
            n = int(selected.sum())
            total_net = float(net[selected].sum())
            rows.append({
                "scenario": scenario["name"], "policy": policy,
                "decision_date": config["validation"]["final_test_decision_date"],
                "prediction_source": dec["value_prediction_source"],
                "evaluation_population": "final_temporal_test", "n_evaluation_customers": len(df),
                "discount_pct": scenario["discount_pct"], "response_rate_reference": scenario["response_rate"],
                "gross_margin_pct": dec["gross_margin_pct"], "campaign_budget": dec["budget_total"],
                "campaign_capacity": int(dec["budget_total"] // dec["contact_cost_per_customer"]),
                "contact_cost_per_customer": dec["contact_cost_per_customer"],
                "customers_targeted": n, "campaign_contact_cost": n * dec["contact_cost_per_customer"],
                "simulated_gross_incremental_margin": float(gross[selected].sum()),
                "simulated_net_incremental_margin": total_net,
                "net_margin_per_targeted_customer": total_net / n if n else 0.0,
                "simulated_margin_difference_vs_baseline": total_net - random_total,
            })
    return pd.DataFrame(rows)


def policy_selections(df: pd.DataFrame, net_margin: pd.Series, config: dict) -> dict:
    dec = config["decision"]
    proposed = budget_constrained_selection(net_margin, dec["budget_total"], dec["contact_cost_per_customer"])
    n = int(proposed.sum())
    # Equal actual N as well as equal maximum capacity, including the no-contact case.
    random_ids = df.sample(n=n, random_state=config["seed"]).index
    rfm_ids = df.sort_values(["rfm_score", "monetary", "Customer ID"],
                            ascending=[False, False, True], kind="stable").index[:n]
    return {"non_targeted": pd.Series(df.index.isin(random_ids), index=df.index),
            "rfm": pd.Series(df.index.isin(rfm_ids), index=df.index), "proposed": proposed}


def bootstrap_margin_ci(margin_selected: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(margin_selected)
    if n == 0:
        return 0.0, 0.0, 0.0
    totals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        totals.append(margin_selected[idx].sum())
    point = margin_selected.sum()
    lo, hi = np.percentile(totals, [2.5, 97.5])
    return point, lo, hi


def assign_recommendation(df: pd.DataFrame, margin_reference: pd.Series) -> pd.Series:
    """Scenario-based actions, not identified causal response categories:
      - nurture: positive NET margin and positive predicted revenue.
      - retain: margin <= 0 (khuyến mãi không đáng) NHƯNG vẫn là khách hàng
        giá trị (Champions/Loyal) -> giữ chân bằng kênh KHÔNG
        tốn discount (chăm sóc, loyalty recognition), không lãng phí margin.
      - no-offer: còn lại (Lost/At-risk với margin âm và giá trị thấp) ->
        không đáng đầu tư nguồn lực.
    """
    recommendation = pd.Series("no-offer", index=df.index)
    recommendation[(margin_reference > 0) & (df[VALUE_COL] > 0)] = "nurture"
    retain_mask = (margin_reference <= 0) & df["rfm_segment"].isin(["Champions", "Loyal"]) & (df[VALUE_COL] > 0)
    recommendation[retain_mask] = "retain"
    return recommendation


def tornado_sensitivity(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Vary từng giả định quanh reference scenario, giữ nguyên các giả định
    khác, đo tác động lên budget-constrained simulated NET margin."""
    df = prepare_scores(df, config)
    dec = config["decision"]
    ref = next(s for s in dec["discount_response_scenarios"] if s["name"] == dec["reference_scenario"])
    base_discount = ref["discount_pct"]
    base_response = ref["response_rate"]
    base_margin_pct = dec["gross_margin_pct"]
    base_at_risk_mult = dec["segment_response_multiplier"]["At-risk"]

    all_discounts = [s["discount_pct"] for s in dec["discount_response_scenarios"]]
    all_responses = [s["response_rate"] for s in dec["discount_response_scenarios"]]

    def total_margin(discount_pct, response_rate, gross_margin_pct, at_risk_mult):
        mult = dict(dec["segment_response_multiplier"])
        mult["At-risk"] = at_risk_mult
        eff_rr = effective_response_rate(df, response_rate, mult)
        margin = incremental_margin(df[VALUE_COL], eff_rr, discount_pct, gross_margin_pct,
                                    dec["contact_cost_per_customer"])
        selected = budget_constrained_selection(margin, dec["budget_total"], dec["contact_cost_per_customer"])
        return margin[selected].sum()

    base_total = total_margin(base_discount, base_response, base_margin_pct, base_at_risk_mult)

    scenarios = [
        ("discount_pct", min(all_discounts), max(all_discounts),
         lambda v: total_margin(v, base_response, base_margin_pct, base_at_risk_mult)),
        ("response_rate", min(all_responses), max(all_responses),
         lambda v: total_margin(base_discount, v, base_margin_pct, base_at_risk_mult)),
        ("gross_margin_pct", base_margin_pct * 0.75, base_margin_pct * 1.25,
         lambda v: total_margin(base_discount, base_response, v, base_at_risk_mult)),
        ("at_risk_response_multiplier", base_at_risk_mult * 0.7, base_at_risk_mult * 1.3,
         lambda v: total_margin(base_discount, base_response, base_margin_pct, v)),
    ]

    rows = []
    for name, low_val, high_val, fn in scenarios:
        low_total = fn(low_val)
        high_total = fn(high_val)
        rows.append({
            "parameter": name, "base_total_margin": base_total,
            "low_value": low_val, "low_total_margin": low_total,
            "high_value": high_val, "high_total_margin": high_total,
            "swing": abs(high_total - low_total),
        })
    result = pd.DataFrame(rows).sort_values("swing", ascending=False)
    logger.info("Base simulated net margin (reference scenario, budget-constrained): %.0f", base_total)
    logger.info("Tornado sensitivity:\n%s", result.to_string(index=False))
    return result


def tornado_bar_bounds(low, high, base):
    """Nonnegative widths even when higher parameter values reduce margin."""
    low, high = np.asarray(low) - base, np.asarray(high) - base
    return np.minimum(low, high), np.abs(high - low)


def plot_tornado(tornado_df: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        base = tornado_df["base_total_margin"].iloc[0]
        y_pos = np.arange(len(tornado_df))
        left, width = tornado_bar_bounds(tornado_df["low_total_margin"], tornado_df["high_total_margin"], base)
        ax.barh(y_pos, width, left=left, color="#4C72B0")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(tornado_df["parameter"])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Change in simulated NET margin vs base case (£)")
        ax.set_title("Tornado sensitivity — promotion targeting policy")
        fig_path = PROJECT_ROOT / "outputs" / "figures" / "tornado_sensitivity.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info("Đã ghi tornado chart: %s", fig_path)
    except Exception as e:
        logger.warning("Không vẽ được tornado chart: %s", e)


def main() -> None:
    config = load_config()
    set_seed(config["seed"])
    dec = config["decision"]

    df = load_scores(config)
    logger.info("Final temporal-test customers: %d | Predicted gross sales: %.0f", len(df), df[VALUE_COL].sum())

    # --- Scenario table (bắt buộc theo đề: >=3 discount/response assumptions) ---
    scenario_table = build_scenario_table(df, config)
    scenario_path = outputs_path("scenario_table.csv")
    scenario_table.to_csv(scenario_path, index=False)
    logger.info("Đã ghi scenario table: %s", scenario_path)

    # --- Reference scenario: recommendation layer + budget-constrained selection ---
    ref = next(s for s in dec["discount_response_scenarios"] if s["name"] == dec["reference_scenario"])
    eff_rr_ref = effective_response_rate(df, ref["response_rate"], dec["segment_response_multiplier"])
    gross_ref = incremental_margin(df[VALUE_COL], eff_rr_ref, ref["discount_pct"], dec["gross_margin_pct"])
    margin_ref = gross_ref - dec["contact_cost_per_customer"]
    is_targeted_budget = budget_constrained_selection(margin_ref, dec["budget_total"], dec["contact_cost_per_customer"])

    df["effective_response_rate_reference"] = eff_rr_ref
    df["promotion_scenario"] = ref["name"]
    df["simulated_gross_incremental_margin"] = gross_ref
    df["simulated_net_incremental_margin"] = margin_ref
    df["expected_contact_cost_if_targeted"] = dec["contact_cost_per_customer"]
    df["final_recommendation"] = assign_recommendation(df, margin_ref)
    df["targeted_under_budget_reference_scenario"] = is_targeted_budget
    df["target_rank"] = margin_ref.where(margin_ref > 0).rank(method="first", ascending=False).astype("Int64")

    logger.info("Phân phối final_recommendation:\n%s", df["final_recommendation"].value_counts().to_string())
    logger.info("Phân phối recommendation theo segment:\n%s",
                pd.crosstab(df["rfm_segment"], df["final_recommendation"]).to_string())

    # Conditional bootstrap of selected predicted contributions, NOT causal uncertainty.
    positive_margin = margin_ref[is_targeted_budget].to_numpy()
    point, lo, hi = bootstrap_margin_ci(positive_margin, config["validation"]["n_bootstrap"], config["seed"])
    logger.info("Conditional bootstrap — simulated NET margin (reference, budget-constrained): "
                "%.0f [%.0f, %.0f]", point, lo, hi)

    # --- Tornado sensitivity ---
    tornado_df = tornado_sensitivity(df, config)
    tornado_path = outputs_path("tornado_sensitivity.csv")
    tornado_df.to_csv(tornado_path, index=False)
    plot_tornado(tornado_df)

    # Final-test evaluation deliverable; never consume legacy/demo customer_scores.csv.
    scores_path = processed_path("customer_test_scores.csv", config)
    df.to_csv(scores_path, index=False)
    logger.info("Final temporal-test decision scores: %s", scores_path)


if __name__ == "__main__":
    main()
