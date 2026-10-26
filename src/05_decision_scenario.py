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
  incremental_margin_i = v_i * gross_margin_pct * [(1 + effective_response_rate_i) * (1 - discount_pct) - 1]
  trong đó v_i = predicted_future_value (GBM, Tuần 3) — ước lượng unconditional
  expected revenue trong outcome window nếu KHÔNG có khuyến mãi.

Output:
  - outputs/scenario_table.csv
  - outputs/tornado_sensitivity.csv (+ chart)
  - data/processed/customer_scores.csv (cập nhật: incremental_margin, final_recommendation)

Chạy: python src/05_decision_scenario.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils import PROJECT_ROOT, get_logger, load_config, outputs_path, processed_path, set_seed

logger = get_logger("05_decision_scenario")

VALUE_COL = "predicted_future_value_gbm"


def load_scores(config: dict) -> pd.DataFrame:
    df = pd.read_csv(processed_path("customer_scores.csv", config))
    if VALUE_COL not in df.columns:
        raise ValueError(
            f"Thiếu cột {VALUE_COL} trong customer_scores.csv — chạy "
            "src/04_advanced_models.py trước (Tuần 3) để có GBM predictions."
        )
    return df


def effective_response_rate(df: pd.DataFrame, base_response_rate: float, multipliers: dict) -> pd.Series:
    mult = df["rfm_segment"].map(multipliers)
    if mult.isna().any():
        missing = df.loc[mult.isna(), "rfm_segment"].unique()
        raise ValueError(f"segment_response_multiplier thiếu segment: {missing}")
    return base_response_rate * mult


def incremental_margin(v: pd.Series, eff_response_rate: pd.Series, discount_pct: float, gross_margin_pct: float) -> pd.Series:
    return v * gross_margin_pct * ((1 + eff_response_rate) * (1 - discount_pct) - 1)


def budget_constrained_selection(margin: pd.Series, budget_total: float, cost_per_customer: float) -> pd.Series:
    """Chọn top-N theo incremental_margin giảm dần, N = floor(budget/cost),
    NHƯNG chỉ chọn khách hàng có margin > 0 (0/1 knapsack với cost đồng nhất:
    không chọn item có giá trị âm dù còn ngân sách — chọn top-N theo margin
    giảm dần trong tập margin>0 là lời giải TỐI ƯU, không phải xấp xỉ, vì mọi
    item có cùng cost)."""
    max_n = int(budget_total // cost_per_customer)
    eligible = margin[margin > 0].sort_values(ascending=False)
    selected_index = eligible.index[:max_n]
    is_selected = pd.Series(False, index=margin.index)
    is_selected.loc[selected_index] = True
    return is_selected


def build_scenario_table(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    dec = config["decision"]
    rows = []
    for scenario in dec["discount_response_scenarios"]:
        eff_rr = effective_response_rate(df, scenario["response_rate"], dec["segment_response_multiplier"])
        margin = incremental_margin(df[VALUE_COL], eff_rr, scenario["discount_pct"], dec["gross_margin_pct"])

        total_blanket = margin.sum()  # non-targeted policy: khuyến mãi cho TẤT CẢ khách hàng
        positive_mask = margin > 0
        total_targeted_positive = margin[positive_mask].sum()
        n_targeted_positive = int(positive_mask.sum())

        is_selected_budget = budget_constrained_selection(margin, dec["budget_total"], dec["contact_cost_per_customer"])
        total_budget_constrained = margin[is_selected_budget].sum()

        rows.append({
            "scenario": scenario["name"],
            "discount_pct": scenario["discount_pct"],
            "response_rate_reference": scenario["response_rate"],
            "gross_margin_pct": dec["gross_margin_pct"],
            "total_margin_non_targeted_blanket": total_blanket,
            "total_margin_targeted_positive_only": total_targeted_positive,
            "n_customers_targeted_positive_only": n_targeted_positive,
            "total_margin_budget_constrained": total_budget_constrained,
            "n_customers_budget_constrained": int(is_selected_budget.sum()),
            "uplift_vs_blanket": total_targeted_positive - total_blanket,
        })
        logger.info(
            "[%s] d=%.0f%% r_ref=%.0f%% -> blanket=%.0f | targeted(positive-only)=%.0f (n=%d) | "
            "budget-constrained=%.0f (n=%d)",
            scenario["name"], 100 * scenario["discount_pct"], 100 * scenario["response_rate"],
            total_blanket, total_targeted_positive, n_targeted_positive,
            total_budget_constrained, int(is_selected_budget.sum()),
        )
    return pd.DataFrame(rows)


def bootstrap_margin_ci(margin_selected: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(margin_selected)
    totals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        totals.append(margin_selected[idx].sum())
    point = margin_selected.sum()
    lo, hi = np.percentile(totals, [2.5, 97.5])
    return point, lo, hi


def assign_recommendation(df: pd.DataFrame, margin_reference: pd.Series) -> pd.Series:
    """retain/nurture/no-offer — gắn với lý thuyết uplift modeling:
      - nurture: incremental_margin > 0 dưới reference scenario (persuadable,
        khuyến mãi margin-accretive) -> gửi promo.
      - retain: margin <= 0 (khuyến mãi không đáng) NHƯNG vẫn là khách hàng
        giá trị (Champions/Loyal, "sure things") -> giữ chân bằng kênh KHÔNG
        tốn discount (chăm sóc, loyalty recognition), không lãng phí margin.
      - no-offer: còn lại (Lost/At-risk với margin âm và giá trị thấp) ->
        không đáng đầu tư nguồn lực.
    """
    recommendation = pd.Series("no-offer", index=df.index)
    recommendation[margin_reference > 0] = "nurture"
    retain_mask = (margin_reference <= 0) & df["rfm_segment"].isin(["Champions", "Loyal"]) & (df[VALUE_COL] > 0)
    recommendation[retain_mask] = "retain"
    return recommendation


def tornado_sensitivity(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Vary từng giả định quanh reference scenario, giữ nguyên các giả định
    khác, đo tác động lên total_margin_targeted_positive_only."""
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
        margin = incremental_margin(df[VALUE_COL], eff_rr, discount_pct, gross_margin_pct)
        return margin[margin > 0].sum()

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
    logger.info("Base total margin (reference scenario, targeted positive-only): %.0f", base_total)
    logger.info("Tornado sensitivity:\n%s", result.to_string(index=False))
    return result


def plot_tornado(tornado_df: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        base = tornado_df["base_total_margin"].iloc[0]
        y_pos = np.arange(len(tornado_df))
        low = tornado_df["low_total_margin"] - base
        high = tornado_df["high_total_margin"] - base
        for i, (lo, hi) in enumerate(zip(low, high)):
            ax.barh(i, hi - lo, left=min(lo, hi), color="#4C72B0")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(tornado_df["parameter"])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Thay đổi total expected margin so với base case")
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
    logger.info("Số khách hàng: %d | Tổng predicted_future_value_gbm: %.0f", len(df), df[VALUE_COL].sum())

    # --- Scenario table (bắt buộc theo đề: >=3 discount/response assumptions) ---
    scenario_table = build_scenario_table(df, config)
    scenario_path = outputs_path("scenario_table.csv")
    scenario_table.to_csv(scenario_path, index=False)
    logger.info("Đã ghi scenario table: %s", scenario_path)

    # --- Reference scenario: recommendation layer + budget-constrained selection ---
    ref = next(s for s in dec["discount_response_scenarios"] if s["name"] == dec["reference_scenario"])
    eff_rr_ref = effective_response_rate(df, ref["response_rate"], dec["segment_response_multiplier"])
    margin_ref = incremental_margin(df[VALUE_COL], eff_rr_ref, ref["discount_pct"], dec["gross_margin_pct"])
    is_targeted_budget = budget_constrained_selection(margin_ref, dec["budget_total"], dec["contact_cost_per_customer"])

    df["effective_response_rate_reference"] = eff_rr_ref
    df["incremental_margin_reference_scenario"] = margin_ref
    df["final_recommendation"] = assign_recommendation(df, margin_ref)
    df["targeted_under_budget_reference_scenario"] = is_targeted_budget

    logger.info("Phân phối final_recommendation:\n%s", df["final_recommendation"].value_counts().to_string())
    logger.info("Phân phối recommendation theo segment:\n%s",
                pd.crosstab(df["rfm_segment"], df["final_recommendation"]).to_string())

    # --- Bootstrap CI cho total margin của chính sách margin-positive-only ---
    positive_margin = margin_ref[margin_ref > 0].to_numpy()
    point, lo, hi = bootstrap_margin_ci(positive_margin, config["validation"]["n_bootstrap"], config["seed"])
    logger.info("Bootstrap CI — total incremental margin (reference scenario, targeted positive-only): "
                "%.0f [%.0f, %.0f]", point, lo, hi)

    # --- Tornado sensitivity ---
    tornado_df = tornado_sensitivity(df, config)
    tornado_path = outputs_path("tornado_sensitivity.csv")
    tornado_df.to_csv(tornado_path, index=False)
    plot_tornado(tornado_df)

    # --- Cập nhật customer_scores.csv (deliverable) ---
    scores_path = processed_path("customer_scores.csv", config)
    df.to_csv(scores_path, index=False)
    logger.info("Đã cập nhật customer_scores.csv với decision layer: %s", scores_path)


if __name__ == "__main__":
    main()
