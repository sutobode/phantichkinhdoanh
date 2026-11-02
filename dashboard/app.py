"""Interactive dashboard for customer value and promotion targeting.

Run from the project root with: streamlit run dashboard/app.py
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


def sidebar_filter(label: str, all_label: str, options: list[str], key: str) -> list[str]:
    """Keep the default sidebar compact while preserving multi-select filtering."""
    if st.sidebar.checkbox(all_label, value=True, key=f"{key}_all"):
        return options
    return st.sidebar.multiselect(label, options, key=key, placeholder=f"Select {label.lower()}")


def gbp(value: float) -> str:
    return "—" if pd.isna(value) else f"£{value:,.2f}"


try:
    config, scores, scenario_table, cohort, metrics = load_data()
except (FileNotFoundError, ValueError, KeyError) as exc:
    st.error(f"Final temporal-test results unavailable: {exc}. Run steps 02 → 03 → 05 first.")
    st.stop()

st.title("Customer Value & Promotion Targeting — Online Retailer")
st.caption(
    f"Final temporal test: {config['validation']['final_test_decision_date']} · "
    f"{len(scores):,} customers · {scores['repeat_purchase'].mean():.2%} repeat purchase · "
    f"PR-AUC {metrics['pr_auc'][0]:.3f} · MAE {gbp(metrics['mae'][0])}"
)
st.info(
    "All promotion outcomes are scenario-based simulations under explicit assumptions. "
    "They are not measured causal uplift or realized campaign profit."
)

# --- Sidebar filters ---
st.sidebar.header("Filters")
st.sidebar.caption("Filters apply to the sandbox and customer analysis only.")
countries = sorted(scores["country"].dropna().unique().tolist())
selected_countries = sidebar_filter("Countries", "All countries", countries, "countries")

segments = sorted(scores["rfm_segment"].dropna().unique().tolist())
selected_segments = sidebar_filter("RFM segments", "All RFM segments", segments, "segments")

recommendations = sorted(scores["final_recommendation"].dropna().unique().tolist())
selected_recs = sidebar_filter(
    "Saved recommendations", "All recommendations", recommendations, "recommendations"
)

filtered = scores[
    scores["country"].isin(selected_countries)
    & scores["rfm_segment"].isin(selected_segments)
    & scores["final_recommendation"].isin(selected_recs)
]

st.sidebar.caption(f"Showing **{len(filtered):,} of {len(scores):,}** customers")

# --- Scenario Sandbox: only policy arithmetic on fixed final-test predictions ---
st.subheader("Scenario Sandbox")
st.caption(
    f"Exploratory policy arithmetic for {len(filtered):,} filtered customers using fixed model predictions. "
    "The contact budget is reallocated within this subset; no model is retrained."
)
ref = next(s for s in config["decision"]["discount_response_scenarios"] if s["name"] == config["decision"]["reference_scenario"])
a, b, c, d = st.columns(4)
discount_pct = a.slider("Discount", 0, 40, int(round(100 * ref["discount_pct"])), 1, format="%d%%")
response_pct = b.slider(
    "Assumed sales-volume growth", 0, 100, int(round(100 * ref["response_rate"])), 1, format="%d%%"
)
margin_pct = c.slider(
    "Gross margin", 5, 80, int(round(100 * config["decision"]["gross_margin_pct"])), 1, format="%d%%"
)
budget = d.number_input("Contact budget (£)", min_value=0.0, max_value=10000.0,
                        value=float(config["decision"]["budget_total"]), step=50.0, format="%.2f")
discount = discount_pct / 100
response = response_pct / 100
margin = margin_pct / 100
sandbox, summary = decision.evaluate_proposed_scenario(filtered, config, discount, response, margin, budget)
sandbox["promotion_scenario"] = "exploratory_sandbox"
sandbox["assumed_discount_pct"] = discount
sandbox["assumed_base_sales_growth"] = response
sandbox["assumed_gross_margin_pct"] = margin
sandbox["contact_budget"] = budget

# --- Decision KPIs ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Eligible customers", f"{summary['customers_eligible']:,}")
col2.metric("Selected customers", f"{summary['customers_targeted']:,}")
col3.metric("Simulated NET margin", f"£{summary['simulated_net_incremental_margin']:,.2f}")
col4.metric("Contact cost", f"£{summary['campaign_contact_cost']:,.2f}")
if filtered.empty:
    st.warning("No customers match the current filters. Adjust the sidebar selection to evaluate a campaign.")
elif summary["customers_targeted"] == 0:
    st.success(
        "No campaign selected: no customer has positive simulated NET contribution after discount economics "
        f"and the £{config['decision']['contact_cost_per_customer']:.2f} contact cost. "
        "The No Promotion reference remains £0.00."
    )

st.divider()

# --- Scenario comparison chart ---
st.subheader("Global Policy Comparison")
st.caption(
    f"Saved results for all {len(scores):,} final-test customers; sidebar filters and sandbox controls do not apply. "
    f"Non-targeted values average {config['decision']['random_policy_simulations']:,} seeded campaigns; "
    "error bars are 2.5th–97.5th simulation percentiles, not causal confidence intervals. "
    "All policies share maximum capacity K=1,500; Proposed may abstain."
)
chart_data = scenario_table.copy()
chart_data["policy_label"] = chart_data["policy"].map({"no_promotion": "No Promotion", "non_targeted": "Non-targeted (MC mean)",
                                                        "rfm": "RFM", "proposed": "Proposed"})
chart_data["scenario_label"] = chart_data["scenario"].str.title()
chart_data["error_up"] = (chart_data["random_p975_net_margin"] - chart_data["simulated_net_incremental_margin"]).fillna(0).clip(lower=0)
chart_data["error_down"] = (chart_data["simulated_net_incremental_margin"] - chart_data["random_p025_net_margin"]).fillna(0).clip(lower=0)
fig_scenario = px.bar(
    chart_data, x="scenario_label", y="simulated_net_incremental_margin", color="policy_label", barmode="group",
    error_y="error_up", error_y_minus="error_down",
    labels={"simulated_net_incremental_margin": "Simulated NET margin (£)",
            "scenario_label": "Scenario", "policy_label": "Policy"},
    title="Simulated NET margin by policy and scenario",
    category_orders={"scenario_label": ["Conservative", "Moderate", "Aggressive"],
                     "policy_label": ["No Promotion", "Non-targeted (MC mean)", "RFM", "Proposed"]},
)
fig_scenario.update_layout(legend_title_text="Policy", margin={"t": 60, "b": 30})
st.plotly_chart(fig_scenario, width="stretch")
policy_display = scenario_table[["scenario", "policy", "discount_pct", "response_rate_reference",
                                 "customers_targeted", "campaign_contact_cost",
                                 "simulated_net_incremental_margin", "random_p025_net_margin",
                                 "random_p975_net_margin"]].copy()
policy_display["scenario"] = policy_display["scenario"].str.title()
policy_display["policy"] = policy_display["policy"].map({
    "no_promotion": "No Promotion", "non_targeted": "Non-targeted (MC mean)",
    "rfm": "RFM", "proposed": "Proposed",
})
policy_display["discount_pct"] = policy_display["discount_pct"].map(lambda value: f"{value:.0%}")
policy_display["response_rate_reference"] = policy_display["response_rate_reference"].map(lambda value: f"{value:.0%}")
for column in ("campaign_contact_cost", "simulated_net_incremental_margin",
               "random_p025_net_margin", "random_p975_net_margin"):
    policy_display[column] = policy_display[column].map(gbp)
policy_display = policy_display.rename(columns={
    "scenario": "Scenario", "policy": "Policy", "discount_pct": "Discount",
    "response_rate_reference": "Assumed Growth", "customers_targeted": "Customers Contacted",
    "campaign_contact_cost": "Contact Cost", "simulated_net_incremental_margin": "Simulated NET Margin",
    "random_p025_net_margin": "Simulation P2.5", "random_p975_net_margin": "Simulation P97.5",
})
st.dataframe(policy_display, width="stretch", hide_index=True)

st.divider()

# --- Customer analysis ---
st.subheader("Customer Analysis")
if filtered.empty:
    st.info("No customer analysis is shown because the current filters return an empty population.")
else:
    st.caption(
        f"Filtered population: {len(filtered):,} customers · predicted 90-day gross sales "
        f"{gbp(filtered['baseline_future_revenue'].sum())} · average repeat probability "
        f"{filtered['predicted_repeat_purchase_proba'].mean():.1%}."
    )
    col_a, col_b = st.columns(2)
    with col_a:
        seg_counts = filtered["rfm_segment"].value_counts().rename_axis("RFM Segment").reset_index(name="Customers")
        fig_seg = px.bar(
            seg_counts, x="RFM Segment", y="Customers", color="RFM Segment",
            title="RFM segment distribution",
            category_orders={"RFM Segment": ["Champions", "Loyal", "At-risk", "Lost"]},
        )
        fig_seg.update_layout(showlegend=False, margin={"t": 55, "b": 30})
        st.plotly_chart(fig_seg, width="stretch")
    with col_b:
        recommendation_labels = {"retain": "Retain", "nurture": "Nurture", "no-offer": "No Offer"}
        cross = pd.crosstab(sandbox["rfm_segment"], sandbox["final_recommendation"]).reset_index()
        cross_long = cross.melt(id_vars="rfm_segment", var_name="recommendation", value_name="Customers")
        cross_long["Recommendation"] = cross_long["recommendation"].map(recommendation_labels)
        cross_long = cross_long.rename(columns={"rfm_segment": "RFM Segment"})
        fig_rec = px.bar(
            cross_long, x="RFM Segment", y="Customers", color="Recommendation", barmode="stack",
            title="Sandbox recommendation by segment",
            category_orders={"RFM Segment": ["Champions", "Loyal", "At-risk", "Lost"],
                             "Recommendation": ["Retain", "Nurture", "No Offer"]},
        )
        fig_rec.update_layout(margin={"t": 55, "b": 30})
        st.plotly_chart(fig_rec, width="stretch")

st.markdown("**Observed cohort activity**")
st.caption("Global observation-history view; sidebar filters and sandbox controls do not apply.")
cohort_pivot = cohort.pivot(index="cohort_month", columns="months_since_acquisition", values="retention_rate")
fig_cohort = px.imshow(
    cohort_pivot, color_continuous_scale="Blues", aspect="auto",
    labels={"x": "Months since first observed purchase", "y": "First observed purchase cohort",
            "color": "Share active"},
    title="Monthly customer activity by first observed purchase cohort",
)
fig_cohort.update_coloraxes(colorbar_tickformat=".0%")
fig_cohort.update_layout(margin={"t": 55, "b": 30})
st.plotly_chart(fig_cohort, width="stretch")

st.divider()

# --- Actionable view ---
st.subheader("Actionable Targeting Output")
st.caption(f"Filtered sandbox selection · contact budget £{budget:,.2f} · "
           f"£{config['decision']['contact_cost_per_customer']:.2f} per contact · "
           f"maximum capacity {summary['campaign_capacity']:,} customers")

priority_list = sandbox[sandbox["targeted_under_budget_reference_scenario"]].sort_values(
    "simulated_net_incremental_margin", ascending=False
)
display_cols = [
    "Customer ID", "decision_date", "country", "rfm_segment", "final_recommendation",
    "baseline_future_revenue", "predicted_repeat_purchase_proba", "promotion_scenario",
    "simulated_net_incremental_margin", "expected_contact_cost_if_targeted", "target_rank",
    "assumed_discount_pct", "assumed_base_sales_growth", "assumed_gross_margin_pct", "contact_budget",
]
if priority_list.empty:
    st.info("No customers are selected under the current filters and sandbox assumptions.")
else:
    priority_display = priority_list[display_cols].rename(columns={
        "decision_date": "Decision Date", "country": "Country", "rfm_segment": "RFM Segment",
        "final_recommendation": "Recommendation", "baseline_future_revenue": "Forecast Gross Sales",
        "predicted_repeat_purchase_proba": "Repeat Purchase Probability",
        "promotion_scenario": "Scenario", "simulated_net_incremental_margin": "Simulated NET Margin",
        "expected_contact_cost_if_targeted": "Contact Cost", "target_rank": "Target Rank",
        "assumed_discount_pct": "Discount", "assumed_base_sales_growth": "Assumed Growth",
        "assumed_gross_margin_pct": "Gross Margin", "contact_budget": "Contact Budget",
    })
    st.dataframe(
        priority_display.style.format({
            "Forecast Gross Sales": "£{:,.2f}", "Repeat Purchase Probability": "{:.1%}",
            "Simulated NET Margin": "£{:,.2f}", "Contact Cost": "£{:,.2f}",
            "Discount": "{:.0%}", "Assumed Growth": "{:.0%}", "Gross Margin": "{:.0%}",
            "Contact Budget": "£{:,.2f}",
        }),
        width="stretch", height=400, hide_index=True,
    )
st.download_button(
    "Download targeting list (CSV)",
    priority_list[display_cols].to_csv(index=False).encode("utf-8"),
    "priority_targeting_list.csv",
    "text/csv",
)
