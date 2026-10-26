"""Interactive dashboard — Customer Value & Promotion Targeting.

3-5 KPI quyết định + 1 actionable view (danh sách khách hàng ưu tiên target),
có filter theo country/segment/recommendation.

Chạy: streamlit run dashboard/app.py
(chạy từ project root, hoặc dùng path tuyệt đối bên dưới nên chạy ở đâu cũng được)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils import load_config, processed_path, outputs_path  # noqa: E402

st.set_page_config(page_title="Customer Value & Promotion Targeting", layout="wide")


@st.cache_data
def load_data():
    config = load_config(PROJECT_ROOT / "config" / "params.yaml")
    scores = pd.read_csv(processed_path("customer_scores.csv", config))
    scenario_table = pd.read_csv(outputs_path("scenario_table.csv"))
    cohort = pd.read_csv(outputs_path("cohort_retention.csv"))
    return config, scores, scenario_table, cohort


config, scores, scenario_table, cohort = load_data()

st.title("Customer Value & Promotion Targeting — Online Retailer")
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
selected_recs = st.sidebar.multiselect("Recommendation", recommendations, default=recommendations)

filtered = scores[
    scores["country"].isin(selected_countries)
    & scores["rfm_segment"].isin(selected_segments)
    & scores["final_recommendation"].isin(selected_recs)
]

st.sidebar.markdown(f"**{len(filtered)} / {len(scores)}** khách hàng sau filter")

# --- KPI row (3-5 KPI quyết định) ---
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Số khách hàng", f"{len(filtered):,}")
col2.metric("Tổng expected future value", f"£{filtered['predicted_future_value_gbm'].sum():,.0f}")
col3.metric("Avg repeat-purchase proba", f"{filtered['predicted_repeat_purchase_proba_gbm'].mean():.1%}")
col4.metric(
    "Expected incremental margin (nurture only)",
    f"£{filtered.loc[filtered['final_recommendation']=='nurture', 'incremental_margin_reference_scenario'].sum():,.0f}",
)
retention_risk = (filtered["predicted_repeat_purchase_proba_gbm"] < 0.3).mean() if len(filtered) else 0
col5.metric("Retention risk (% proba<30%)", f"{retention_risk:.1%}")

st.divider()

# --- Scenario comparison chart ---
st.subheader("So sánh policy theo scenario (toàn bộ customer base, không áp filter)")
st.caption("blanket = gửi khuyến mãi cho TẤT CẢ khách hàng (non-targeted). "
           "targeted = chỉ gửi cho khách hàng có incremental margin dương. "
           "budget-constrained = targeted nhưng giới hạn theo ngân sách liên hệ.")
scenario_long = scenario_table.melt(
    id_vars=["scenario", "discount_pct", "response_rate_reference"],
    value_vars=["total_margin_non_targeted_blanket", "total_margin_targeted_positive_only", "total_margin_budget_constrained"],
    var_name="policy", value_name="total_margin",
)
scenario_long["policy"] = scenario_long["policy"].map({
    "total_margin_non_targeted_blanket": "Blanket (non-targeted)",
    "total_margin_targeted_positive_only": "Targeted (positive-margin)",
    "total_margin_budget_constrained": "Budget-constrained",
})
fig_scenario = px.bar(
    scenario_long, x="scenario", y="total_margin", color="policy", barmode="group",
    labels={"total_margin": "Expected incremental margin (£)", "scenario": "Scenario"},
    title="Expected incremental margin theo policy & scenario",
)
st.plotly_chart(fig_scenario, use_container_width=True)

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Phân phối RFM Segment")
    seg_counts = filtered["rfm_segment"].value_counts().reset_index()
    seg_counts.columns = ["segment", "count"]
    fig_seg = px.bar(seg_counts, x="segment", y="count", color="segment")
    st.plotly_chart(fig_seg, use_container_width=True)

with col_b:
    st.subheader("Recommendation theo Segment")
    cross = pd.crosstab(filtered["rfm_segment"], filtered["final_recommendation"]).reset_index()
    cross_long = cross.melt(id_vars="rfm_segment", var_name="recommendation", value_name="count")
    fig_rec = px.bar(cross_long, x="rfm_segment", y="count", color="recommendation", barmode="stack")
    st.plotly_chart(fig_rec, use_container_width=True)

st.divider()

# --- Cohort retention heatmap ---
st.subheader("Cohort retention (theo tháng acquisition)")
cohort_pivot = cohort.pivot(index="cohort_month", columns="months_since_acquisition", values="retention_rate")
fig_cohort = px.imshow(
    cohort_pivot, color_continuous_scale="Blues", aspect="auto",
    labels={"x": "Tháng kể từ acquisition", "y": "Cohort (tháng mua đầu)", "color": "Retention rate"},
)
st.plotly_chart(fig_cohort, use_container_width=True)

st.divider()

# --- Actionable view: danh sách khách hàng ưu tiên target ---
st.subheader("🎯 Actionable view — Danh sách khách hàng ưu tiên (nurture, theo budget)")
st.caption(f"Ngân sách liên hệ: £{config['decision']['budget_total']:.0f} "
           f"@ £{config['decision']['contact_cost_per_customer']:.2f}/khách hàng "
           f"= tối đa {int(config['decision']['budget_total'] / config['decision']['contact_cost_per_customer'])} khách hàng")

priority_list = filtered[filtered["targeted_under_budget_reference_scenario"]].sort_values(
    "incremental_margin_reference_scenario", ascending=False
)
display_cols = [
    "Customer ID", "country", "rfm_segment", "final_recommendation",
    "predicted_future_value_gbm", "predicted_repeat_purchase_proba_gbm",
    "incremental_margin_reference_scenario",
]
st.dataframe(priority_list[display_cols], use_container_width=True, height=400)
st.download_button(
    "Tải danh sách (CSV)",
    priority_list[display_cols].to_csv(index=False).encode("utf-8"),
    "priority_targeting_list.csv",
    "text/csv",
)
