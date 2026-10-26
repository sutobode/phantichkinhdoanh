"""Interactive dashboard — Customer Value & Promotion Targeting.

3-5 KPI quyết định + 1 actionable view (danh sách khách hàng ưu tiên target),
có filter theo country/segment/recommendation.

Chạy: streamlit run dashboard/app.py
(chạy từ project root, hoặc dùng path tuyệt đối bên dưới nên chạy ở đâu cũng được)
"""
from __future__ import annotations

import sys
import json
from importlib import import_module
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils import load_config, processed_path, outputs_path, validate_final_test_predictions  # noqa: E402
decision = import_module("05_decision_scenario")

st.set_page_config(page_title="Customer Value & Promotion Targeting", layout="wide")


@st.cache_data
def load_data():
    config = load_config(PROJECT_ROOT / "config" / "params.yaml")
    scores = pd.read_csv(processed_path("customer_test_scores.csv", config))
    validate_final_test_predictions(scores, config)
    scenario_table = pd.read_csv(outputs_path("scenario_table.csv"))
    if (not scores["prediction_source"].eq(config["decision"]["value_prediction_source"]).all()
            or "policy" not in scenario_table
            or not scenario_table["prediction_source"].eq(config["decision"]["value_prediction_source"]).all()
            or not scenario_table["decision_date"].eq(config["validation"]["final_test_decision_date"]).all()):
        raise ValueError("Regenerate final temporal-test policy outputs for the current configuration")
    cohort = pd.read_csv(outputs_path("cohort_retention.csv"))
    registry = json.loads((PROJECT_ROOT / "models" / "temporal_registry.json").read_text(encoding="utf-8"))
    model_name = "logistic_regression" if config["decision"]["value_prediction_source"] == "baseline" else "gradient_boosting_classifier"
    metrics = [run for run in registry if model_name in run["models"]][-1]["metrics"]["final_test"]
    return config, decision.prepare_scores(scores, config), scenario_table, cohort, metrics


try:
    config, scores, scenario_table, cohort, metrics = load_data()
except (FileNotFoundError, ValueError, KeyError) as exc:
    st.error(f"Final temporal-test results unavailable: {exc}. Run steps 02 → 03 → 05 first.")
    st.stop()

st.title("Customer Value & Promotion Targeting — Online Retailer")
st.caption(f"Evaluation only: final temporal test at {config['validation']['final_test_decision_date']}; "
           f"model source: {config['decision']['value_prediction_source']}. Not a current deployment list.")
st.caption(f"Full test: {len(scores):,} customers; repeat purchase {scores['repeat_purchase'].mean():.2%}; "
           f"PR-AUC {metrics['pr_auc'][0]:.6f}; ROC-AUC {metrics['roc_auc'][0]:.6f}; "
           f"MAE £{metrics['mae'][0]:,.2f}; RMSE £{metrics['rmse'][0]:,.2f}.")
st.caption(
    "Business question: khách hàng nào nên được ưu tiên retention/promotion để "
    "tăng expected future margin mà không over-discount? "
    "⚠️ incremental margin là SCENARIO-BASED ESTIMATE, không phải causal uplift "
    "đo được từ dữ liệu thật (dataset không có randomized treatment/control)."
)

# --- Sidebar filters ---
st.sidebar.header("Filter")
countries = sorted(scores["country"].dropna().unique().tolist())
selected_countries = st.sidebar.multiselect("Quốc gia", countries, default=countries)

segments = sorted(scores["rfm_segment"].dropna().unique().tolist())
selected_segments = st.sidebar.multiselect("RFM Segment", segments, default=segments)

recommendations = sorted(scores["final_recommendation"].dropna().unique().tolist())
selected_recs = st.sidebar.multiselect("Recommendation (saved reference scenario)", recommendations, default=recommendations)

filtered = scores[
    scores["country"].isin(selected_countries)
    & scores["rfm_segment"].isin(selected_segments)
    & scores["final_recommendation"].isin(selected_recs)
]

st.sidebar.markdown(f"**{len(filtered)} / {len(scores)}** khách hàng sau filter")

# --- Scenario Sandbox: only policy arithmetic on fixed final-test predictions ---
st.subheader("Scenario Sandbox")
st.info("Scenario assumptions are exploratory and non-causal. No Promotion = £0 simulated incremental margin.")
st.caption(f"Sandbox scope: {len(filtered):,} filtered final-test customers. Budget is reallocated within this subset. "
           "Global saved comparisons and cohort view below do not change with filters or sandbox inputs. No model retraining.")
ref = next(s for s in config["decision"]["discount_response_scenarios"] if s["name"] == config["decision"]["reference_scenario"])
a, b, c, d = st.columns(4)
discount = a.slider("Discount rate", 0.0, 0.40, float(ref["discount_pct"]), 0.01)
response = b.slider("Assumed base sales-volume growth", 0.0, 1.0, float(ref["response_rate"]), 0.01)
margin = c.slider("Gross margin rate", 0.05, 0.80, float(config["decision"]["gross_margin_pct"]), 0.01)
budget = d.number_input("Contact budget (£)", min_value=0.0, max_value=10000.0,
                        value=float(config["decision"]["budget_total"]), step=50.0)
sandbox, summary = decision.evaluate_proposed_scenario(filtered, config, discount, response, margin, budget)
sandbox["promotion_scenario"] = "exploratory_sandbox"
sandbox["assumed_discount_pct"] = discount
sandbox["assumed_base_sales_growth"] = response
sandbox["assumed_gross_margin_pct"] = margin
sandbox["contact_budget"] = budget

# --- Five decision KPIs ---
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Eligible customers", f"{summary['customers_eligible']:,}")
col2.metric("Customers targeted", f"{summary['customers_targeted']:,}")
col3.metric("Simulated NET margin", f"£{summary['simulated_net_incremental_margin']:,.2f}")
col4.metric("Difference vs No Promotion", f"£{summary['simulated_margin_vs_no_promotion']:,.2f}")
col5.metric("Campaign contact cost", f"£{summary['campaign_contact_cost']:,.2f}")
if summary["customers_targeted"] == 0:
    st.caption("No economically eligible campaign under these inputs: retain the No Promotion reference.")

st.divider()

# --- Scenario comparison chart ---
st.subheader("So sánh policy theo scenario (toàn bộ customer base, không áp filter)")
st.caption(f"Non-targeted = mean of {config['decision']['random_policy_simulations']:,} seeded K-customer campaigns; "
           "error bars are the 2.5th–97.5th random-policy simulation percentiles, NOT a causal confidence interval. "
           "No Promotion = £0; RFM = score/monetary ranking; proposed = positive net margin. "
           "Same final-test population, maximum capacity K, assumptions and per-contact cost. "
           "Random/RFM contact K customers; proposed contacts only positive-NET-margin customers, up to K, possibly zero.")
chart_data = scenario_table.copy()
chart_data["policy_label"] = chart_data["policy"].map({"no_promotion": "No Promotion", "non_targeted": "Non-targeted (MC mean)",
                                                       "rfm": "RFM", "proposed": "Proposed"})
chart_data["error_up"] = (chart_data["random_p975_net_margin"] - chart_data["simulated_net_incremental_margin"]).fillna(0).clip(lower=0)
chart_data["error_down"] = (chart_data["simulated_net_incremental_margin"] - chart_data["random_p025_net_margin"]).fillna(0).clip(lower=0)
fig_scenario = px.bar(
    chart_data, x="scenario", y="simulated_net_incremental_margin", color="policy_label", barmode="group",
    error_y="error_up", error_y_minus="error_down",
    labels={"simulated_net_incremental_margin": "Simulated NET margin (£)", "scenario": "Scenario"},
    title="Equal-capacity policy comparison — final temporal test",
)
st.plotly_chart(fig_scenario, width="stretch")
st.dataframe(scenario_table[["scenario", "policy", "customers_targeted", "campaign_contact_cost",
                            "simulated_net_incremental_margin", "random_p025_net_margin", "random_p975_net_margin",
                             "simulated_margin_vs_no_promotion"]], width="stretch")

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Phân phối RFM Segment")
    st.caption(f"Filtered predicted gross sales: £{filtered['baseline_future_revenue'].sum():,.0f}; "
               f"average repeat probability: {filtered['predicted_repeat_purchase_proba'].mean():.1%}.")
    seg_counts = filtered["rfm_segment"].value_counts().reset_index()
    seg_counts.columns = ["segment", "count"]
    fig_seg = px.bar(seg_counts, x="segment", y="count", color="segment")
    st.plotly_chart(fig_seg, width="stretch")

with col_b:
    st.subheader("Sandbox recommendation theo Segment")
    cross = pd.crosstab(sandbox["rfm_segment"], sandbox["final_recommendation"]).reset_index()
    cross_long = cross.melt(id_vars="rfm_segment", var_name="recommendation", value_name="count")
    fig_rec = px.bar(cross_long, x="rfm_segment", y="count", color="recommendation", barmode="stack")
    st.plotly_chart(fig_rec, width="stretch")

st.divider()

# --- Cohort retention heatmap ---
st.subheader("Cohort retention (theo tháng acquisition)")
st.caption("Full final-test observation population; sidebar filters do not apply.")
cohort_pivot = cohort.pivot(index="cohort_month", columns="months_since_acquisition", values="retention_rate")
fig_cohort = px.imshow(
    cohort_pivot, color_continuous_scale="Blues", aspect="auto",
    labels={"x": "Tháng kể từ acquisition", "y": "Cohort (tháng mua đầu)", "color": "Retention rate"},
)
st.plotly_chart(fig_cohort, width="stretch")

st.divider()

# --- Actionable view: danh sách khách hàng ưu tiên target ---
st.subheader("🎯 Actionable view — Sandbox targeting list")
st.caption(f"Exploratory, filtered-population selection. Contact budget: £{budget:.0f} "
           f"@ £{config['decision']['contact_cost_per_customer']:.2f}/khách hàng "
            f"= capacity {summary['campaign_capacity']} customers")

priority_list = sandbox[sandbox["targeted_under_budget_reference_scenario"]].sort_values(
    "simulated_net_incremental_margin", ascending=False
)
display_cols = [
    "Customer ID", "decision_date", "country", "rfm_segment", "final_recommendation",
    "baseline_future_revenue", "predicted_repeat_purchase_proba", "promotion_scenario",
    "simulated_net_incremental_margin", "expected_contact_cost_if_targeted", "target_rank",
    "assumed_discount_pct", "assumed_base_sales_growth", "assumed_gross_margin_pct", "contact_budget",
]
st.dataframe(priority_list[display_cols], width="stretch", height=400)
st.download_button(
    "Tải danh sách (CSV)",
    priority_list[display_cols].to_csv(index=False).encode("utf-8"),
    "priority_targeting_list.csv",
    "text/csv",
)
