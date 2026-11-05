"""Business-assumption robustness, temporal/RFM diagnostics, Monte Carlo
convergence và net-sales proxy feasibility (workstream "Assumptions & Robustness").

SECONDARY ANALYSIS — không ghi đè bất kỳ output chính nào. Không đổi temporal
split, target, công thức kinh tế, config hay RFM labels. Mọi tham số kinh tế
dưới đây là lưới KHAI BÁO TRƯỚC (không chọn theo lợi nhuận final test); mọi
growth/elasticity là giả định minh họa, không phải causal estimate.

  B. Temporal counts + rolling-origin eligibility (dùng đúng hàm của 04),
     RFM quartile boundaries, ties, recency check, cơ cấu R/F/M của At-risk.
  C. Lưới giả định g x d x b x multiplier-set + đường cong elasticity
     (Tellis 1988; Bijmolt et al. 2005) -> launch/abstain, ngưỡng hòa vốn.
  D. Monte Carlo random policy: hội tụ theo seed/số lần rút vs kỳ vọng giải tích
     E = (K/N) * sum(net_i).
  E. Feasibility của future_net_sales_proxy: cancellation có ghép được với
     purchase gốc không, và bao nhiêu thuộc về purchase TRƯỚC decision date.

Output: outputs/assumption_robustness.csv, outputs/assumption_robustness/*.csv
Chạy (sau 02→05 và 04): python src/11_assumption_robustness.py
"""
from __future__ import annotations

import copy
from importlib import import_module
from itertools import product

import numpy as np
import pandas as pd

from utils import get_logger, load_config, outputs_path, set_seed

logger = get_logger("11_assumption_robustness")

cleaning = import_module("02_cleaning_features")
baseline = import_module("03_baseline_models")
advanced = import_module("04_advanced_models")
decision = import_module("05_decision_scenario")

SEGMENTS = ["Champions", "Loyal", "At-risk", "Lost"]

# --- Lưới khai báo trước (Task C). Không tinh chỉnh theo kết quả. ---
GRID_GROSS_MARGIN = [0.30, 0.40, 0.50]
GRID_DISCOUNT = [0.05, 0.10, 0.15, 0.20]
GRID_GROWTH = [0.10, 0.20, 0.35, 0.50]
MULTIPLIER_SETS = {
    "configured": {"Champions": 0.5, "Loyal": 1.0, "At-risk": 1.3, "Lost": 0.4},
    "flat": {"Champions": 1.0, "Loyal": 1.0, "At-risk": 1.0, "Lost": 1.0},
    "compressed": {"Champions": 0.75, "Loyal": 1.0, "At-risk": 1.15, "Lost": 0.7},
    "steep": {"Champions": 0.25, "Loyal": 1.0, "At-risk": 1.6, "Lost": 0.2},
}
# Độ co giãn giá trung bình từ meta-analysis cấp THƯƠNG HIỆU (không phải cấp
# cửa hàng; chưa trừ phần brand switching) — chỉ dùng minh họa.
ELASTICITIES = {"tellis_1988_mean": -1.76, "bijmolt_2005_mean": -2.62}
MC_SEEDS = [11, 22, 33, 44, 55]
MC_DRAWS = [500, 1000, 5000]


def out(name: str):
    return outputs_path(f"assumption_robustness/{name}")


# --- Kinh tế học thuần (có unit test) ---------------------------------------

def alpha(discount: float, growth: float, multiplier: float, gross_margin: float) -> float:
    """Lãi/lỗ gộp trên mỗi £1 doanh số nền: (1 + b*m)(g - d) - g."""
    return (1 + growth * multiplier) * (gross_margin - discount) - gross_margin


def elasticity_growth(discount: float, elasticity: float) -> float:
    """Tăng sản lượng ngụ ý bởi cầu co giãn hằng số: (1-d)^elasticity - 1."""
    return (1 - discount) ** elasticity - 1


def elasticity_break_even(discount: float, gross_margin: float) -> float:
    """|elasticity| tối thiểu để alpha > 0 khi multiplier = 1:
    (1-d)^-e (g-d) = g  <=>  e = ln(g/(g-d)) / -ln(1-d)."""
    if gross_margin <= discount:
        return float("inf")
    return float(np.log(gross_margin / (gross_margin - discount)) / -np.log(1 - discount))


def analytic_random_expectation(net: np.ndarray, k: int) -> float:
    """Kỳ vọng tổng net margin của tập K phần tử chọn đều không hoàn lại."""
    return float(k / len(net) * np.sum(net))


# --- Task B -----------------------------------------------------------------

def temporal_counts(snapshots: pd.DataFrame, raw: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows = []
    counts = snapshots.groupby(["split", "decision_date"]).size()
    for (split, date), n in counts.items():
        rows.append({"design": "main", "item": f"{split} snapshot {pd.Timestamp(date).date()}", "count": int(n)})
    train, validation, _ = cleaning.temporal_partitions(snapshots, config)
    rows += [{"design": "main", "item": "validation fit = train snapshots", "count": len(train)},
             {"design": "main", "item": "final fit = train + validation snapshots", "count": len(train) + len(validation)}]

    rolling = advanced.build_rolling_snapshots(raw, config)
    schedule = advanced.rolling_origin_schedule(config)
    for origin in schedule["decision_date"].iloc[1:]:
        strict = advanced.eligible_rolling_training(rolling, origin)
        inclusive = rolling[(pd.to_datetime(rolling["decision_date"]) < origin)
                            & (pd.to_datetime(rolling["outcome_end_exclusive"]) <= origin)]
        used = sorted(pd.to_datetime(strict["decision_date"]).dt.date.unique())
        rows.append({"design": "rolling", "item": f"origin {origin.date()} n_train (code: strict <)",
                     "count": len(strict), "training_origins": ", ".join(map(str, used))})
        rows.append({"design": "rolling", "item": f"origin {origin.date()} n_train if <= were used",
                     "count": len(inclusive)})
        rows.append({"design": "rolling", "item": f"origin {origin.date()} n_evaluation",
                     "count": int((pd.to_datetime(rolling["decision_date"]) == origin).sum())})
    return pd.DataFrame(rows)


def rfm_diagnostics(final: pd.DataFrame, raw: pd.DataFrame, config: dict) -> dict[str, pd.DataFrame]:
    df = final.sort_values("Customer ID").reset_index(drop=True)
    scores = {
        "R": pd.qcut(df["recency_days"].rank(method="first"), 4, labels=[4, 3, 2, 1]).astype(int),
        "F": pd.qcut(df["frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int),
        "M": pd.qcut(df["monetary"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int),
    }
    cols = {"R": "recency_days", "F": "frequency", "M": "monetary"}
    if not (scores["R"] + scores["F"] + scores["M"]).equals(baseline.rfm_score(df)):
        raise ValueError("RFM recomputation does not match 03_baseline_models.rfm_score")

    boundaries, ties = [], []
    for dim, s in scores.items():
        values = df[cols[dim]]
        for q in sorted(s.unique()):
            v = values[s == q]
            boundaries.append({"dimension": dim, "score": q, "n": len(v), "min": v.min(), "max": v.max()})
        split_values = values.groupby(values).apply(lambda x: s[x.index].nunique())
        split_values = split_values[split_values > 1]
        ties.append({"dimension": dim, "tied_values_split_across_scores": len(split_values),
                     "customers_in_split_tie_groups": int(values.isin(split_values.index).sum()),
                     "example_values": ", ".join(map(str, split_values.index[:5]))})

    pct = {dim: df[cols[dim]].rank(method="average", pct=True) for dim in cols}
    alt = (np.ceil(pct["R"].rsub(1).add(1e-12) * 4).clip(1, 4)  # recency nhỏ -> điểm cao
           + np.ceil(pct["F"] * 4).clip(1, 4) + np.ceil(pct["M"] * 4).clip(1, 4))
    alt_seg = alt.apply(lambda x: "Champions" if x >= 10 else "Loyal" if x >= 8 else "At-risk" if x >= 6 else "Lost")
    seg = baseline.rfm_segment(df)
    tie_effect = pd.DataFrame([{
        "customers": len(df), "segment_changes_if_average_rank": int((alt_seg != seg).sum()),
        "share": float((alt_seg != seg).mean()),
        "note": "rank(method='first') phá hòa theo thứ tự dòng (Customer ID tăng dần); chỉ để định lượng, không đổi nhãn",
    }])

    t = pd.Timestamp(config["validation"]["final_test_decision_date"])
    filtered = cleaning.filter_valid_customers(raw, config)
    valid = cleaning.drop_invalid_rows(filtered[filtered["InvoiceDate"] < t], config, "rfm_check")
    last = valid.groupby("Customer ID")["InvoiceDate"].max()
    expected = (t - df["Customer ID"].map(last)).dt.days
    recency_check = pd.DataFrame([{"customers": len(df), "recency_matches_T_minus_last_valid_purchase":
                                   int((expected == df["recency_days"]).sum())}])

    df["R"], df["F"], df["M"], df["segment"] = scores["R"], scores["F"], scores["M"], seg
    at_risk = (df[df["segment"] == "At-risk"].groupby(["R", "F", "M"])
               .agg(customers=("Customer ID", "size"), repeat_rate=("repeat_purchase", "mean"),
                    mean_future_sales=("future_gross_sales", "mean"))
               .reset_index().sort_values("customers", ascending=False))
    return {"rfm_quartile_boundaries.csv": pd.DataFrame(boundaries), "rfm_ties.csv": pd.DataFrame(ties),
            "rfm_tie_effect.csv": tie_effect, "recency_check.csv": recency_check,
            "at_risk_rfm_patterns.csv": at_risk}


# --- Task C -----------------------------------------------------------------

def evaluate_policy(df: pd.DataFrame, config: dict, d: float, b: float, g: float, mult: dict) -> dict:
    dec = config["decision"]
    cfg = copy.deepcopy(config)
    cfg["decision"]["segment_response_multiplier"] = mult
    growth = decision.effective_response_rate(df, b, mult)
    net = decision.incremental_margin(df[decision.VALUE_COL], growth, d, g, dec["contact_cost_per_customer"])
    selected = decision.budget_constrained_selection(net, dec["budget_total"], dec["contact_cost_per_customer"])
    policies = decision.policy_selections(df, net, cfg)
    k = min(int(dec["budget_total"] // dec["contact_cost_per_customer"]), len(df))
    return {
        "segments_alpha_positive": ", ".join(s for s in SEGMENTS if alpha(d, b, mult[s], g) > 0) or "-",
        "customers_eligible": int((net > 0).sum()), "customers_targeted": int(selected.sum()),
        "launch": bool(selected.any()),
        "proposed_net_margin": float(net[selected].sum()),
        "random_k_expected_net_margin": analytic_random_expectation(net.to_numpy(), k),
        "rfm_k_net_margin": float(net[policies["rfm"]].sum()),
    }


def assumption_grid(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    ref = next(s for s in config["decision"]["discount_response_scenarios"]
               if s["name"] == config["decision"]["reference_scenario"])
    rows = []
    for g, d, b, (set_name, mult) in product(GRID_GROSS_MARGIN, GRID_DISCOUNT, GRID_GROWTH, MULTIPLIER_SETS.items()):
        rows.append({"family": "grid", "gross_margin": g, "discount": d, "base_growth": b,
                     "multiplier_set": set_name, "elasticity": np.nan,
                     "is_reference": (g == config["decision"]["gross_margin_pct"] and d == ref["discount_pct"]
                                      and b == ref["response_rate"] and set_name == "configured"),
                     **evaluate_policy(df, config, d, b, g, mult)})
    for (e_name, e), g, d, set_name in product(ELASTICITIES.items(), GRID_GROSS_MARGIN, GRID_DISCOUNT,
                                               ["flat", "configured"]):
        b = elasticity_growth(d, e)
        rows.append({"family": f"elasticity:{e_name}", "gross_margin": g, "discount": d, "base_growth": b,
                     "multiplier_set": set_name, "elasticity": e, "is_reference": False,
                     **evaluate_policy(df, config, d, b, g, MULTIPLIER_SETS[set_name])})
    return pd.DataFrame(rows)


def break_even_regions(grid: pd.DataFrame) -> pd.DataFrame:
    """Với mỗi (g, multiplier set, d): growth nhỏ nhất trong lưới khiến launch."""
    g = grid[grid["family"] == "grid"]
    launched = g[g["launch"]].groupby(["gross_margin", "multiplier_set", "discount"])["base_growth"].min()
    full = g.groupby(["gross_margin", "multiplier_set", "discount"]).size().index
    return (launched.reindex(full).rename("min_grid_growth_to_launch").reset_index())


def elasticity_thresholds() -> pd.DataFrame:
    rows = []
    for g, d in product(GRID_GROSS_MARGIN, GRID_DISCOUNT):
        rows.append({"gross_margin": g, "discount": d,
                     "required_abs_elasticity_flat_multiplier": elasticity_break_even(d, g),
                     **{f"growth_implied_{k}": elasticity_growth(d, e) for k, e in ELASTICITIES.items()},
                     "break_even_growth_flat": d / (g - d)})
    return pd.DataFrame(rows)


# --- Task D -----------------------------------------------------------------

def monte_carlo_convergence(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    dec = config["decision"]
    k = min(int(dec["budget_total"] // dec["contact_cost_per_customer"]), len(df))
    scenarios = [s for s in dec["discount_response_scenarios"] if s["name"] == dec["reference_scenario"]]
    scenarios.append({"name": "illustrative_light", "discount_pct": 0.05, "response_rate": 0.20})
    rows = []
    for s in scenarios:
        growth = decision.effective_response_rate(df, s["response_rate"], dec["segment_response_multiplier"])
        net = decision.incremental_margin(df[decision.VALUE_COL], growth, s["discount_pct"],
                                          dec["gross_margin_pct"], dec["contact_cost_per_customer"])
        exact = analytic_random_expectation(net.to_numpy(), k)
        for seed, n in product(MC_SEEDS, MC_DRAWS):
            r = decision.monte_carlo_random_policy(net, k, n, seed)
            rows.append({"scenario": s["name"], "seed": seed, "draws": n, "analytic_mean": exact,
                         "mc_mean": r["random_mean_net_margin"],
                         "rel_error_vs_analytic": abs(r["random_mean_net_margin"] - exact) / abs(exact),
                         "p025": r["random_p025_net_margin"], "p975": r["random_p975_net_margin"]})
    return pd.DataFrame(rows)


# --- Task E -----------------------------------------------------------------

def net_proxy_feasibility(raw: pd.DataFrame, final: pd.DataFrame, config: dict) -> pd.DataFrame:
    t = pd.Timestamp(config["validation"]["final_test_decision_date"])
    end = t + pd.Timedelta(days=config["validation"]["outcome_horizon_days"])
    filtered = cleaning.filter_valid_customers(raw, config)
    filtered = filtered[filtered["Customer ID"].isin(final["Customer ID"])]
    inv = filtered["Invoice"].astype(str)
    is_cancel = inv.str.startswith("C")
    purchases = filtered[~is_cancel & (filtered["Quantity"] > 0) & (filtered["Price"] > 0)]
    cancels = filtered[is_cancel & (filtered["InvoiceDate"] >= t) & (filtered["InvoiceDate"] < end)].copy()

    purchase_invoices = set(purchases["Invoice"].astype(str))
    cancels["invoice_number_matches_purchase"] = cancels["Invoice"].astype(str).str[1:].isin(purchase_invoices)

    p = purchases[["Customer ID", "StockCode", "InvoiceDate"]].sort_values("InvoiceDate")
    c = cancels.reset_index().sort_values("InvoiceDate")
    matched = pd.merge_asof(c, p.rename(columns={"InvoiceDate": "last_purchase_date"}),
                            left_on="InvoiceDate", right_on="last_purchase_date",
                            by=["Customer ID", "StockCode"], direction="backward")
    value = -(matched["Quantity"] * matched["Price"])
    origin = np.select(
        [matched["last_purchase_date"].isna(), matched["last_purchase_date"] < t],
        ["no_prior_purchase_same_customer_stockcode", "prior_purchase_before_decision_date"],
        "prior_purchase_inside_outcome_window")
    matched["origin"] = origin
    lag = (matched["InvoiceDate"] - matched["last_purchase_date"]).dt.days

    gross = purchases[(purchases["InvoiceDate"] >= t) & (purchases["InvoiceDate"] < end)]
    gross_value = float((gross["Quantity"] * gross["Price"]).sum())
    rows = [{"item": "cancellation rows in final-test outcome window", "value": len(matched)},
            {"item": "cancelled value (£)", "value": float(value.sum())},
            {"item": "gross sales in outcome window (£)", "value": gross_value},
            {"item": "cancelled value / gross sales", "value": float(value.sum()) / gross_value},
            {"item": "cancellation invoice number (without 'C') equals a purchase invoice",
             "value": float(cancels["invoice_number_matches_purchase"].mean())},
            {"item": "median lag cancellation - latest same-SKU purchase (days)", "value": float(lag.median())}]
    for name, grp in value.groupby(matched["origin"]):
        rows.append({"item": f"value share: {name}", "value": float(grp.sum() / value.sum())})
        rows.append({"item": f"row count: {name}", "value": int(grp.size)})
    return pd.DataFrame(rows)


def main() -> None:
    config = load_config()
    set_seed(config["seed"])
    raw = cleaning.load_raw(config)
    snapshots = baseline.load_dataset(config)
    df = decision.load_scores(config)
    _, _, final = cleaning.temporal_partitions(snapshots, config)

    tc = temporal_counts(snapshots, raw, config)
    tc.to_csv(out("temporal_counts.csv"), index=False)
    logger.info("B. Temporal counts:\n%s", tc.to_string(index=False))
    for name, frame in rfm_diagnostics(final, raw, config).items():
        frame.to_csv(out(name), index=False)
        logger.info("B. %s:\n%s", name, frame.head(15).to_string(index=False))

    grid = assumption_grid(df, config)
    grid.to_csv(outputs_path("assumption_robustness.csv"), index=False)
    regions = break_even_regions(grid)
    regions.to_csv(out("break_even_regions.csv"), index=False)
    thresholds = elasticity_thresholds()
    thresholds.to_csv(out("elasticity_thresholds.csv"), index=False)
    logger.info("C. Grid: %d rows, launch in %d | reference row:\n%s", len(grid), int(grid["launch"].sum()),
                grid[grid["is_reference"]].to_string(index=False))
    logger.info("C. Min grid growth to launch:\n%s",
                regions.pivot_table(index=["gross_margin", "multiplier_set"], columns="discount",
                                    values="min_grid_growth_to_launch").to_string())
    logger.info("C. Elasticity thresholds:\n%s", thresholds.round(3).to_string(index=False))
    logger.info("C. Elasticity rows:\n%s", grid[grid["family"] != "grid"][[
        "family", "gross_margin", "discount", "base_growth", "multiplier_set", "customers_targeted",
        "proposed_net_margin", "random_k_expected_net_margin"]].round(3).to_string(index=False))

    mc = monte_carlo_convergence(df, config)
    mc.to_csv(out("monte_carlo_convergence.csv"), index=False)
    logger.info("D. Monte Carlo convergence:\n%s",
                mc.groupby(["scenario", "draws"]).agg(analytic=("analytic_mean", "first"),
                                                      mc_min=("mc_mean", "min"), mc_max=("mc_mean", "max"),
                                                      max_rel_error=("rel_error_vs_analytic", "max"))
                .round(4).to_string())

    feas = net_proxy_feasibility(raw, final, config)
    feas.to_csv(out("net_proxy_feasibility.csv"), index=False)
    logger.info("E. Net proxy feasibility:\n%s", feas.to_string(index=False))


if __name__ == "__main__":
    main()
