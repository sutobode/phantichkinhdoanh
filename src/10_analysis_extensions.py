"""Analysis extensions — 5 phân tích bổ sung cho decision layer (nhánh hung).

  E1. Two-part value model: v = P(mua) x E[sales | mua]. Linear trên doanh số
      zero-inflated cho 30% dự báo âm; mô hình hai phần tránh việc đó. Chỉ là
      nguồn SO SÁNH — không thay nguồn chính đã chọn trước (Linear baseline).
  E2. Tornado cho CẢ 4 segment multiplier (x0.7 / x1.3) + multiplier hòa vốn.
  E3. Biến thể product-only: bỏ dòng không phải sản phẩm (POST, M, AMAZONFEE,
      ...) khỏi raw, dựng lại snapshot, fit lại baseline, so quyết định.
  E4. Cỡ mẫu A/B test để đo mức tăng doanh số quanh ngưỡng hòa vốn.
  E5. Monte Carlo trên toàn dải giả định (hàm thuần, có unit test).

Mọi con số margin vẫn là scenario-based, non-causal (Scope Guardrail).
Yêu cầu: đã chạy 02, 03, 04 (và 09 cho bảng ablation gốc).
Chạy: python src/10_analysis_extensions.py
"""
from __future__ import annotations

import copy
from importlib import import_module

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import average_precision_score, mean_absolute_error, root_mean_squared_error

from utils import get_logger, load_config, outputs_path, processed_path, set_seed

logger = get_logger("10_analysis_extensions")

cleaning = import_module("02_cleaning_features")
baseline = import_module("03_baseline_models")
decision = import_module("05_decision_scenario")
extensions = import_module("09_scenario_extensions")

FEATURE_COLS = baseline.FEATURE_COLS
TARGET = "future_gross_sales"
SEGMENTS = ["Champions", "Loyal", "At-risk", "Lost"]


def out(name: str):
    return outputs_path(f"analysis_extensions/{name}")


# --- E1. Two-part value model ----------------------------------------------

def smearing_factor(residuals: np.ndarray) -> float:
    """Duan smearing: E[exp(e)] ước lượng bằng trung bình mẫu, sửa bias khi
    quay về thang gốc từ hồi quy trên log1p."""
    return float(np.mean(np.exp(residuals)))


def fit_two_part(fit_df: pd.DataFrame, eval_df: pd.DataFrame, config: dict) -> dict[str, np.ndarray]:
    """Trả về dự báo v >= 0 cho eval_df theo 2 biến thể phần "mức chi":
    linear trên log1p (có smearing) và GBM trên doanh số của người mua."""
    classifier, _ = baseline.baseline_models(config)
    clf = clone(classifier).fit(fit_df[FEATURE_COLS], fit_df["repeat_purchase"])
    p_buy = clf.predict_proba(eval_df[FEATURE_COLS])[:, 1]

    buyers = fit_df[fit_df[TARGET] > 0]
    y_log = np.log1p(buyers[TARGET].to_numpy())
    # Feature cũng log1p (đều >= 0) và log-dự báo bị chặn ở max của train: hồi
    # quy log trên feature thô (monetary tới £484k) ngoại suy exp() ra hàng triệu.
    lin = LinearRegression().fit(np.log1p(buyers[FEATURE_COLS]), y_log)
    smear = smearing_factor(y_log - lin.predict(np.log1p(buyers[FEATURE_COLS])))
    log_pred = np.minimum(lin.predict(np.log1p(eval_df[FEATURE_COLS])), y_log.max())
    spend_lin = np.expm1(log_pred) * smear

    gbm = GradientBoostingRegressor(random_state=config["seed"]).fit(buyers[FEATURE_COLS], buyers[TARGET])
    spend_gbm = gbm.predict(eval_df[FEATURE_COLS])

    return {
        "two_part_loglinear": np.clip(p_buy * spend_lin, 0, None),
        "two_part_gbm": np.clip(p_buy * spend_gbm, 0, None),
    }


def regression_metrics(y: np.ndarray, v: np.ndarray, top_k: int) -> dict:
    order = np.argsort(-v, kind="stable")[:top_k]
    best = np.sort(y)[::-1][:top_k].sum()
    return {
        "mae": mean_absolute_error(y, v),
        "rmse": root_mean_squared_error(y, v),
        "median_abs_error": float(np.median(np.abs(y - v))),
        "share_negative_raw": np.nan,
        f"top{top_k}_sales_capture": float(y[order].sum() / best) if best > 0 else np.nan,
        "spearman": pd.Series(v).corr(pd.Series(y), method="spearman"),
    }


def two_part_evaluation(snapshots: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Đánh giá đúng thiết kế báo cáo: validation (fit train) và final test
    (fit train+validation). Không dùng nhãn final test để chọn gì."""
    train, validation, test = cleaning.temporal_partitions(snapshots, config)
    k = int(config["decision"]["budget_total"] // config["decision"]["contact_cost_per_customer"])
    rows, final_values = [], None
    for role, fit_df, eval_df in (("validation", train, validation),
                                  ("final_test", pd.concat([train, validation], ignore_index=True), test)):
        y = eval_df[TARGET].to_numpy()
        lin_raw = LinearRegression().fit(fit_df[FEATURE_COLS], fit_df[TARGET]).predict(eval_df[FEATURE_COLS])
        values = {"linear_baseline (báo cáo)": np.clip(lin_raw, 0, None), **fit_two_part(fit_df, eval_df, config)}
        for name, v in values.items():
            m = regression_metrics(y, v, k)
            if name.startswith("linear"):
                m["share_negative_raw"] = float((lin_raw < 0).mean())
            rows.append({"split": role, "value_model": name, **m})
        if role == "final_test":
            final_values = pd.DataFrame({"Customer ID": eval_df["Customer ID"].to_numpy(), **values})
    return pd.DataFrame(rows), final_values


def ablation_with_two_part(df: pd.DataFrame, final_values: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Chạy lại ablation của 09 với thêm 2 nguồn two-part (cùng rule, chấm trên
    doanh số thật)."""
    merged = df.merge(final_values.drop(columns=["linear_baseline (báo cáo)"]), on="Customer ID",
                      how="left", validate="one_to_one")
    if merged[["two_part_loglinear", "two_part_gbm"]].isna().any().any():
        raise ValueError("Two-part predictions không khớp final-test population")
    k = extensions.capacity(merged, config)
    rng = np.random.default_rng(config["seed"])
    scenarios = config["decision"]["discount_response_scenarios"] + extensions.ILLUSTRATIVE_SCENARIOS
    sources = {
        "no_model_constant": pd.Series(merged[decision.VALUE_COL].mean(), index=merged.index),
        "linear_baseline (báo cáo)": merged[decision.VALUE_COL],
        "gbm_advanced": merged["predicted_future_value_gbm"].clip(lower=0),
        "two_part_loglinear": merged["two_part_loglinear"],
        "two_part_gbm": merged["two_part_gbm"],
        "oracle_actual_sales": merged[TARGET],
    }
    rows = []
    for s in scenarios:
        real = extensions.net_margin(merged, merged[TARGET], s, config).to_numpy()
        oracle = extensions.expected_realized_with_ties(real, real, k, rng)
        for name, value in sources.items():
            score = extensions.net_margin(merged, value, s, config).to_numpy()
            realized = extensions.expected_realized_with_ties(score, real, k, rng)
            rows.append({"scenario": s["name"], "value_source": name,
                         "customers_eligible": int((score > 0).sum()),
                         "customers_targeted": int(min((score > 0).sum(), k)),
                         "simulated_net_margin_on_actual_sales": realized,
                         "share_of_oracle": realized / oracle if oracle > 0 else np.nan})
    return pd.DataFrame(rows)


# --- E2. Tornado cho cả 4 segment multiplier -------------------------------

def break_even_multiplier(discount: float, growth: float, gross_margin: float) -> float:
    """m* sao cho (1 + growth*m)(g - d) - g = 0 (bỏ qua chi phí liên hệ)."""
    if growth <= 0 or gross_margin <= discount:
        return float("nan")
    return discount / ((gross_margin - discount) * growth)


def segment_tornado(df: pd.DataFrame, config: dict, scenarios: list[dict]) -> pd.DataFrame:
    dec = config["decision"]
    base_mult = dec["segment_response_multiplier"]

    def run(scenario, mult):
        cfg = copy.deepcopy(config)
        cfg["decision"]["segment_response_multiplier"] = mult
        net = extensions.net_margin(df, df[decision.VALUE_COL], scenario, cfg)
        sel = decision.budget_constrained_selection(net, dec["budget_total"], dec["contact_cost_per_customer"])
        return int(sel.sum()), float(net[sel].sum())

    rows = []
    for s in scenarios:
        base_n, base_total = run(s, base_mult)
        m_star = break_even_multiplier(s["discount_pct"], s["response_rate"], dec["gross_margin_pct"])
        for seg in SEGMENTS:
            for side, factor in (("low", 0.7), ("high", 1.3)):
                mult = dict(base_mult)
                mult[seg] = base_mult[seg] * factor
                n, total = run(s, mult)
                rows.append({
                    "scenario": s["name"], "segment": seg, "side": side,
                    "multiplier": mult[seg], "base_multiplier": base_mult[seg],
                    "break_even_multiplier": m_star,
                    "base_profitable": base_mult[seg] > m_star,
                    "customers_targeted": n, "delta_customers": n - base_n,
                    "net_margin": total, "delta_net_margin": total - base_total,
                })
    return pd.DataFrame(rows)


# --- E3. Product-only variant ----------------------------------------------

def is_product(raw: pd.DataFrame) -> pd.Series:
    """StockCode sản phẩm bắt đầu bằng 5 chữ số; còn lại là phí/điều chỉnh."""
    return raw["StockCode"].astype(str).str.match(r"^\d{5}")


def product_only_variant(raw: pd.DataFrame, base_scores: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = is_product(raw)
    snapshots = cleaning.build_temporal_snapshots(raw[mask].copy(), config)
    scores, metrics, _, _ = baseline.evaluate_temporal_models(snapshots, config, *baseline.baseline_models(config),
                                                              source="baseline")
    scores = decision.prepare_scores(scores, config)

    summary = [{
        "variant": "primary (all StockCodes)", "raw_rows": len(raw),
        "final_test_customers": len(base_scores),
        "final_test_pr_auc": average_precision_score(base_scores["repeat_purchase"],
                                                     base_scores["predicted_repeat_purchase_proba"]),
        "final_test_mae": mean_absolute_error(base_scores[TARGET], base_scores[decision.VALUE_COL]),
    }, {
        "variant": "product_only", "raw_rows": int(mask.sum()),
        "final_test_customers": len(scores),
        "final_test_pr_auc": metrics["final_test"]["pr_auc"][0],
        "final_test_mae": metrics["final_test"]["mae"][0],
    }]

    rows = []
    scenarios = config["decision"]["discount_response_scenarios"] + extensions.ILLUSTRATIVE_SCENARIOS
    dec = config["decision"]
    for s in scenarios:
        picks = {}
        for name, frame in (("primary", base_scores), ("product_only", scores)):
            net = extensions.net_margin(frame, frame[decision.VALUE_COL], s, config)
            sel = decision.budget_constrained_selection(net, dec["budget_total"], dec["contact_cost_per_customer"])
            picks[name] = set(frame.loc[sel, "Customer ID"])
            rows.append({"scenario": s["name"], "variant": name, "customers_targeted": int(sel.sum()),
                         "net_margin": float(net[sel].sum())})
        overlap = (len(picks["primary"] & picks["product_only"]) / len(picks["primary"])
                   if picks["primary"] else np.nan)
        for r in rows[-2:]:
            r["list_overlap_with_primary"] = overlap
    return pd.DataFrame(summary), pd.DataFrame(rows)


# --- E4. Cỡ mẫu A/B test ---------------------------------------------------

def sample_size_per_arm(sigma: float, delta: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Hai nhóm độc lập, so trung bình, hai phía: n = 2 (z_{1-a/2}+z_{1-b})^2 s^2 / delta^2."""
    if delta <= 0 or sigma < 0:
        raise ValueError("delta must be positive and sigma nonnegative")
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return int(np.ceil(2 * (z * sigma / delta) ** 2))


def ab_test_sample_sizes(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Mức tăng cần phát hiện = mức tăng HÒA VỐN của doanh số (d/(g-d)) — test
    phải đủ lực để phân biệt "lãi" với "lỗ". sigma lấy từ doanh số 90 ngày
    THỰC TẾ của final test theo segment (quan sát, không phải giả định),
    kèm bản winsorize p99 để giảm ảnh hưởng whale."""
    dec = config["decision"]
    g = dec["gross_margin_pct"]
    rows = []
    for d in sorted({s["discount_pct"] for s in dec["discount_response_scenarios"]
                     + extensions.ILLUSTRATIVE_SCENARIOS}):
        lift = d / (g - d)
        for seg in SEGMENTS:
            y = df.loc[df["rfm_segment"] == seg, TARGET].to_numpy()
            y_w = np.minimum(y, np.quantile(y, 0.99))
            for label, values in (("raw", y), ("winsorized_p99", y_w)):
                mu, sigma = values.mean(), values.std(ddof=1)
                n = sample_size_per_arm(sigma, lift * mu) if mu > 0 else np.nan
                rows.append({
                    "discount_pct": d, "segment": seg, "variance_basis": label,
                    "segment_customers_available": len(y),
                    "baseline_mean_90d_sales": mu, "sd_90d_sales": sigma,
                    "break_even_relative_lift": lift, "detectable_abs_lift": lift * mu,
                    "n_per_arm": n, "n_total_two_arms": 2 * n if n == n else np.nan,
                    "feasible_in_one_cycle": bool(n == n and 2 * n <= len(y)),
                })
    return pd.DataFrame(rows)


# --- E5. Monte Carlo -------------------------------------------------------

def monte_carlo_assumptions(v: np.ndarray, seg_idx: np.ndarray, base_mult: np.ndarray, config: dict,
                            n_draws: int, mode: str, seed: int) -> pd.DataFrame:
    """Rút đồng thời discount, growth, gross margin, 4 multiplier (xU(0.5,1.5)).
    mode="linked": growth bám discount theo 3 scenario cấu hình (x U(0.7,1.3));
    mode="independent": rút độc lập (stress test)."""
    dec = config["decision"]
    scen = sorted(dec["discount_response_scenarios"], key=lambda s: s["discount_pct"])
    sd, sb = [s["discount_pct"] for s in scen], [s["response_rate"] for s in scen]
    g0, c = dec["gross_margin_pct"], dec["contact_cost_per_customer"]
    k = int(dec["budget_total"] // c)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_draws):
        d = rng.uniform(min(sd), max(sd))
        b = np.interp(d, sd, sb) * rng.uniform(0.7, 1.3) if mode == "linked" else rng.uniform(min(sb), max(sb))
        g = rng.uniform(g0 * 0.75, g0 * 1.25)
        mult = base_mult * rng.uniform(0.5, 1.5, size=len(base_mult))
        alpha = (1 + b * mult) * (g - d) - g
        net = v * alpha[seg_idx] - c
        pos = np.sort(net[net > 0])[::-1][:k]
        rows.append({"discount": d, "growth": b, "gross_margin": g, "max_alpha": alpha.max(),
                     "n_selected": len(pos), "proposed_net_margin": pos.sum(),
                     "random_k_net_margin": net.mean() * min(k, len(net))})
    return pd.DataFrame(rows)


def summarize_monte_carlo(sims: pd.DataFrame) -> dict:
    picked = sims["n_selected"] > 0
    return {
        "P(policy chọn 0 khách)": float((~picked).mean()),
        "P(random K khách lỗ)": float((sims["random_k_net_margin"] < 0).mean()),
        "P(proposed >= random)": float((sims["proposed_net_margin"] >= sims["random_k_net_margin"]).mean()),
        "Median margin khi có chọn (£)": float(sims.loc[picked, "proposed_net_margin"].median()) if picked.any() else 0.0,
        "P95 margin proposed (£)": float(sims["proposed_net_margin"].quantile(0.95)),
    }


def main() -> None:
    config = load_config()
    set_seed(config["seed"])
    df = extensions.load_inputs(config)

    # E1
    snapshots = baseline.load_dataset(config)
    two_part_metrics, final_values = two_part_evaluation(snapshots, config)
    two_part_metrics.to_csv(out("two_part_metrics.csv"), index=False)
    logger.info("E1. Two-part value model:\n%s", two_part_metrics.round(4).to_string(index=False))
    ablation = ablation_with_two_part(df, final_values, config)
    ablation.to_csv(out("ablation_with_two_part.csv"), index=False)
    logger.info("E1. Ablation (share of oracle trên doanh số thật):\n%s",
                ablation.pivot(index="value_source", columns="scenario", values="share_of_oracle")
                .round(3).to_string())

    # E2
    scenarios = config["decision"]["discount_response_scenarios"] + extensions.ILLUSTRATIVE_SCENARIOS
    tornado = segment_tornado(df, config, scenarios)
    tornado.to_csv(out("segment_multiplier_tornado.csv"), index=False)
    logger.info("E2. Segment multiplier tornado:\n%s", tornado[[
        "scenario", "segment", "side", "multiplier", "break_even_multiplier", "customers_targeted",
        "delta_customers", "net_margin", "delta_net_margin"]].round(3).to_string(index=False))

    # E3
    raw = cleaning.load_raw(config)
    po_summary, po_decisions = product_only_variant(raw, df, config)
    po_summary.to_csv(out("product_only_summary.csv"), index=False)
    po_decisions.to_csv(out("product_only_decisions.csv"), index=False)
    logger.info("E3. Product-only:\n%s\n%s", po_summary.round(4).to_string(index=False),
                po_decisions.round(3).to_string(index=False))

    # E4
    ab = ab_test_sample_sizes(df, config)
    ab.to_csv(out("ab_test_sample_size.csv"), index=False)
    logger.info("E4. A/B sample size:\n%s", ab[[
        "discount_pct", "segment", "variance_basis", "segment_customers_available", "baseline_mean_90d_sales",
        "break_even_relative_lift", "n_per_arm", "feasible_in_one_cycle"]].round(3).to_string(index=False))

    # E5
    seg_idx = df["rfm_segment"].map({s: i for i, s in enumerate(SEGMENTS)}).to_numpy()
    base_mult = np.array([config["decision"]["segment_response_multiplier"][s] for s in SEGMENTS])
    summary = {}
    for mode in ("linked", "independent"):
        sims = monte_carlo_assumptions(df[decision.VALUE_COL].to_numpy(), seg_idx, base_mult, config,
                                       5000, mode, config["seed"])
        sims.to_csv(out(f"monte_carlo_{mode}.csv"), index=False)
        summary[mode] = summarize_monte_carlo(sims)
    mc = pd.DataFrame(summary)
    mc.to_csv(out("monte_carlo_summary.csv"))
    logger.info("E5. Monte Carlo (5000 draws):\n%s", mc.round(3).to_string())


if __name__ == "__main__":
    main()
