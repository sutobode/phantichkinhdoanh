"""Kiểm chứng độc lập các con số trong BA-report.pdf từ raw data.

BA-report.pdf dùng thiết kế KHÁC pipeline 02→05 trong repo: snapshot theo
decision date T, horizon 90 ngày, chia train/validation/final-test theo thời
gian, Linear Regression làm nguồn dự báo cho promotion, và công thức margin
  ΔM_i = v_i[(1 + b·m_z)(g − d) − g] − c        (Eq. 1 trong báo cáo)
Script này dựng lại đúng thiết kế đó (theo mô tả Section 3, 5, 7, 8) và so
từng con số với giá trị báo cáo công bố.

Chạy: python src/08_verify_ba_report.py
"""
from __future__ import annotations

from importlib import import_module

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils import get_logger, load_config, outputs_path


def clip_nonneg(predictions: np.ndarray) -> np.ndarray:
    return np.clip(predictions, 0, None)


logger = get_logger("08_verify_ba_report")

cleaning = import_module("02_cleaning_features")
baseline = import_module("03_baseline_models")
FEATURE_COLS = baseline.FEATURE_COLS
CLEAN_CFG = {"cleaning": {"drop_cancelled_invoices": True, "drop_non_positive_quantity": True,
                          "drop_non_positive_price": True}}
HORIZON = pd.Timedelta(days=90)
SCHEDULE = {"train1": "2010-12-01", "train2": "2011-03-01", "validation": "2011-06-01", "test": "2011-09-10"}

G, C, BUDGET = 0.40, 0.50, 750.0
MULT = {"Champions": 0.5, "Loyal": 1.0, "At-risk": 1.3, "Lost": 0.4}
SCENARIOS = {"Conservative": (0.05, 0.10), "Moderate": (0.10, 0.20), "Aggressive": (0.20, 0.35)}

checks: list[dict] = []


def check(section: str, item: str, reported: float, reproduced: float, tol: float) -> None:
    ok = abs(reported - reproduced) <= tol
    checks.append({"section": section, "item": item, "reported": reported,
                   "reproduced": round(float(reproduced), 4), "match": ok})


def snapshot(raw_f: pd.DataFrame, decision_date: str) -> pd.DataFrame:
    """Feature từ giao dịch t < T, outcome từ [T, T+90 ngày)."""
    t = pd.Timestamp(decision_date)
    obs = raw_f[raw_f["InvoiceDate"] < t]
    fut = raw_f[(raw_f["InvoiceDate"] >= t) & (raw_f["InvoiceDate"] < t + HORIZON)]
    return_rate = cleaning.compute_return_rate(obs)
    obs_clean = cleaning.drop_invalid_rows(obs, CLEAN_CFG, f"obs<{decision_date}")
    fut_clean = cleaning.drop_invalid_rows(fut, CLEAN_CFG, f"fut@{decision_date}")
    feats = cleaning.compute_rfm_features(obs_clean, return_rate, t)
    labels = cleaning.compute_labels(fut_clean, feats["Customer ID"])
    df = feats.merge(labels, on="Customer ID").rename(columns={"future_gross_sales": "future_value"})
    df["rfm_segment"] = baseline.rfm_segment(df)

    # Biến thể net (chỉ dùng cho robustness, không phải thiết kế của báo cáo):
    # net = mua - trả trong horizon (dòng Price>0), clip >= 0; return share
    # lịch sử = giá trị trả / giá trị mua trong t < T (không leak).
    fut_p = fut[fut["Price"] > 0]
    net = (fut_p["Quantity"] * fut_p["Price"]).groupby(fut_p["Customer ID"]).sum()
    df["net_future_value"] = df["Customer ID"].map(net).fillna(0.0).clip(lower=0).values
    obs_p = obs[obs["Price"] > 0]
    value = obs_p["Quantity"] * obs_p["Price"]
    is_cancel = obs_p["Invoice"].astype(str).str.startswith("C")
    bought = value[~is_cancel & (obs_p["Quantity"] > 0)].groupby(obs_p["Customer ID"]).sum()
    returned = (-value[is_cancel]).groupby(obs_p["Customer ID"]).sum()
    share = (returned.reindex(bought.index).fillna(0) / bought).clip(0, 1)
    df["obs_return_share"] = df["Customer ID"].map(share).fillna(0.0).values
    return df


def fit_models(train: pd.DataFrame):
    X, y_cls, y_reg = train[FEATURE_COLS], train["repeat_purchase"], train["future_value"]
    logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)).fit(X, y_cls)
    gbm_c = GradientBoostingClassifier(random_state=42).fit(X, y_cls)
    linear = LinearRegression().fit(X, y_reg)
    gbm_r = GradientBoostingRegressor(random_state=42).fit(X, y_reg)
    return logreg, gbm_c, linear, gbm_r


def evaluate(split: str, train: pd.DataFrame, test: pd.DataFrame, reported: dict) -> np.ndarray:
    logreg, gbm_c, linear, gbm_r = fit_models(train)
    X, y_cls, y_reg = test[FEATURE_COLS], test["repeat_purchase"], test["future_value"]
    p_lr, p_gb = logreg.predict_proba(X)[:, 1], gbm_c.predict_proba(X)[:, 1]
    raw_lin = linear.predict(X)
    v_lin, v_gb = clip_nonneg(raw_lin), clip_nonneg(gbm_r.predict(X))
    naive_v = np.full(len(test), train["future_value"].mean())
    got = {
        "AP naive": y_cls.mean(),
        "AP logistic": average_precision_score(y_cls, p_lr),
        "ROC logistic": roc_auc_score(y_cls, p_lr),
        "AP GBM": average_precision_score(y_cls, p_gb),
        "ROC GBM": roc_auc_score(y_cls, p_gb),
        "MAE naive": mean_absolute_error(y_reg, naive_v),
        "RMSE naive": mean_squared_error(y_reg, naive_v) ** 0.5,
        "MAE linear": mean_absolute_error(y_reg, v_lin),
        "RMSE linear": mean_squared_error(y_reg, v_lin) ** 0.5,
        "MAE GBM": mean_absolute_error(y_reg, v_gb),
        "RMSE GBM": mean_squared_error(y_reg, v_gb) ** 0.5,
    }
    for k, rep in reported.items():
        check(f"6.1 {split}", k, rep, got[k], tol=0.002 if ("AP" in k or "ROC" in k) else 1.0)
    if split == "final test":
        check("6.1 final test", "Số dự báo Linear âm bị clip", 1583, int((raw_lin < 0).sum()), tol=0)
        check("6.1 final test", "Mean dự báo Linear (£)", 379.00, v_lin.mean(), tol=1.0)
    return v_lin


def margin(v: np.ndarray, seg: pd.Series, d: float, b: float, g: float = G) -> np.ndarray:
    m = seg.map(MULT).to_numpy()
    return v * ((1 + b * m) * (g - d) - g) - C


def proposed(delta: np.ndarray) -> tuple[int, float]:
    max_n = int(BUDGET // C)
    pos = np.sort(delta[delta > 0])[::-1][:max_n]
    return len(pos), pos.sum()


def main() -> None:
    config = load_config()
    raw = cleaning.load_raw(config)

    # --- Section 2.1: data quality ---
    check("2.1", "Tổng số dòng", 1_067_371, len(raw), tol=0)
    check("2.1", "Dòng thiếu Customer ID", 243_007, raw["Customer ID"].isna().sum(), tol=0)
    check("2.1", "Dòng trùng lặp hoàn toàn", 34_335, raw.duplicated().sum(), tol=0)

    raw_f = cleaning.filter_valid_customers(raw, config)
    snaps = {name: snapshot(raw_f, date) for name, date in SCHEDULE.items()}

    # --- Table 4: số khách hàng mỗi snapshot ---
    for name, rep in zip(SCHEDULE, [4266, 4537, 4933, 5281]):
        check("Table 4", f"Số khách {name}", rep, len(snaps[name]), tol=0)

    test = snaps["test"]
    check("2.2", "Repeat rate final test (%)", 43.36, 100 * test["repeat_purchase"].mean(), tol=0.01)
    check("2.2", "Mean future gross (£)", 543.11, test["future_value"].mean(), tol=0.01)
    check("Table 1", "Mean historical gross (£)", 2705.44, test["monetary"].mean(), tol=0.01)
    check("Table 1", "Median historical gross (£)", 804.02, test["monetary"].median(), tol=0.01)
    top = test["future_value"].sort_values(ascending=False)
    check("2.2", "Top 53 chiếm % future gross", 37.58, 100 * top.iloc[:53].sum() / top.sum(), tol=0.01)

    # --- Table 3: RFM segment ---
    for seg, n, rr in [("Champions", 1507, 76.51), ("Loyal", 1065, 46.57),
                       ("At-risk", 1136, 33.01), ("Lost", 1573, 16.91)]:
        sub = test[test["rfm_segment"] == seg]
        check("Table 3", f"Số khách {seg}", n, len(sub), tol=0)
        check("Table 3", f"Repeat rate {seg} (%)", rr, 100 * sub["repeat_purchase"].mean(), tol=0.01)

    # --- Table 5/6: model performance ---
    evaluate("validation", pd.concat([snaps["train1"], snaps["train2"]]), snaps["validation"], {
        "AP naive": 0.3286, "AP logistic": 0.6953, "ROC logistic": 0.8060, "AP GBM": 0.7189, "ROC GBM": 0.8150,
        "MAE naive": 513.04, "RMSE naive": 2057.12, "MAE linear": 294.53, "RMSE linear": 1030.92,
        "MAE GBM": 277.66, "RMSE GBM": 1120.81,
    })
    v = evaluate("final test", pd.concat([snaps["train1"], snaps["train2"], snaps["validation"]]), test, {
        "AP naive": 0.4336, "AP logistic": 0.7502, "ROC logistic": 0.7900, "AP GBM": 0.7691, "ROC GBM": 0.7987,
        "MAE naive": 636.34, "RMSE naive": 3288.87, "MAE linear": 395.69, "RMSE linear": 2016.14,
        "MAE GBM": 400.40, "RMSE GBM": 2167.29,
    })
    seg = test["rfm_segment"]

    # --- Table 7: hệ số margin lớn nhất (thuần số học) ---
    for name, rep in zip(SCENARIOS, [-0.0045, -0.0220, -0.1090]):
        d, b = SCENARIOS[name]
        alpha = max((1 + b * m) * (G - d) - G for m in MULT.values())
        check("Table 7", f"Max coefficient {name}", rep, alpha, tol=1e-4)

    # --- Table 8: policy comparison ---
    rng = np.random.default_rng(42)
    rfm_score = (pd.qcut(test["recency_days"].rank(method="first"), 4, labels=[4, 3, 2, 1]).astype(int)
                 + pd.qcut(test["frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
                 + pd.qcut(test["monetary"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int))
    rfm_top = (pd.DataFrame({"s": rfm_score, "m": test["monetary"]})
               .sort_values(["s", "m"], ascending=False).index[:1500])
    rfm_pos = test.index.get_indexer(rfm_top)
    for name, (rep_rand, rep_rfm) in zip(SCENARIOS, [(-17417.59, -55333.45), (-37407.57, -118314.35),
                                                    (-90677.15, -277865.97)]):
        d, b = SCENARIOS[name]
        delta = margin(v, seg, d, b)
        rand = np.mean([delta[rng.choice(len(delta), 1500, replace=False)].sum() for _ in range(1000)])
        check("Table 8", f"Non-targeted {name}", rep_rand, rand, tol=0.01 * abs(rep_rand))
        check("Table 8", f"RFM-based {name}", rep_rfm, delta[rfm_pos].sum(), tol=0.01 * abs(rep_rfm))
        check("Table 8", f"Proposed {name} (số khách chọn)", 0, proposed(delta)[0], tol=0)

    # --- Section 8.2: sensitivity points ---
    for d, b, rep_n, rep_m in [(0.05, 0.20, 1500, 7175.92), (0.10, 0.30, 664, 1424.27), (0.15, 0.50, 616, 962.39)]:
        n, total = proposed(margin(v, seg, d, b))
        check("8.2", f"d={d:.0%} b={b:.0%} số khách", rep_n, n, tol=0)
        check("8.2", f"d={d:.0%} b={b:.0%} margin (£)", rep_m, total, tol=1.0)
    check("8.2", "d=10% b=35% margin (£)", 3990.80, proposed(margin(v, seg, 0.10, 0.35))[1], tol=1.0)
    check("8.2", "g=50% margin (£)", 143.77, proposed(margin(v, seg, 0.10, 0.20, g=0.50))[1], tol=1.0)

    # --- Table 9: median break-even theo segment (Moderate) ---
    d, b = SCENARIOS["Moderate"]
    pos = v > 0
    m = seg.map(MULT).to_numpy()
    b_star = (1 / m) * ((G + C / np.where(pos, v, np.nan)) / (G - d) - 1)
    d_star = G - (G + C / np.where(pos, v, np.nan)) / (1 + b * m)
    for s, n, rb, rd in [("Champions", 1503, 67.38, 3.54), ("Loyal", 951, 34.47, 6.38),
                         ("At-risk", 779, 26.75, 7.91), ("Lost", 465, 90.49, 2.17)]:
        mask = pos & (seg == s).to_numpy()
        check("Table 9", f"{s} số khách v>0", n, mask.sum(), tol=0)
        check("Table 9", f"{s} median growth cần (%)", rb, 100 * np.median(b_star[mask]), tol=0.05)
        check("Table 9", f"{s} median discount tối đa (%)", rd, 100 * np.median(d_star[mask]), tol=0.05)

    # --- 7.4: recommendation labels ---
    delta_ref = margin(v, seg, d, b)
    retain = (delta_ref <= 0) & seg.isin(["Champions", "Loyal"]).to_numpy() & (v > 0)
    check("7.4", "Số khách Retain", 2454, retain.sum(), tol=0)
    check("7.4", "Số khách Nurture", 0, (delta_ref > 0).sum(), tol=0)

    result = pd.DataFrame(checks)
    path = outputs_path("ba_report_verification.csv")
    result.to_csv(path, index=False)
    logger.info("Kết quả kiểm chứng (%d/%d khớp):\n%s", result["match"].sum(), len(result), result.to_string(index=False))

    fit_df = pd.concat([snaps["train1"], snaps["train2"], snaps["validation"]])
    robustness_gross_vs_net(fit_df, test, v)
    robustness_monte_carlo(v, seg)


def robustness_gross_vs_net(fit_df: pd.DataFrame, test: pd.DataFrame, v_gross: np.ndarray) -> None:
    """Phản biện (1): gross vs net. Với α < 0 (3 scenario báo cáo), ΔM = v·α − c
    âm với MỌI v >= 0, nên đổi v (gross -> net) không thể làm ai được chọn.
    Chỉ ở scenario thay thế có α > 0 thì định nghĩa value mới ảnh hưởng."""
    X = test[FEATURE_COLS]
    v_net = clip_nonneg(LinearRegression().fit(fit_df[FEATURE_COLS], fit_df["net_future_value"]).predict(X))
    variants = {
        "gross (báo cáo)": v_gross,
        "net label (retrain)": v_net,
        "gross x (1 - return share)": v_gross * (1 - test["obs_return_share"].to_numpy()),
    }
    seg = test["rfm_segment"]
    rows = []
    for d, b in [(0.05, 0.10), (0.10, 0.20), (0.20, 0.35), (0.05, 0.20), (0.10, 0.30), (0.15, 0.50)]:
        base_sel = None
        for name, val in variants.items():
            delta = margin(val, seg, d, b)
            order = np.argsort(-delta)
            sel = set(order[: proposed(delta)[0]])
            base_sel = sel if base_sel is None else base_sel
            n, total = proposed(delta)
            rows.append({"discount": d, "growth": b, "variant": name, "n_selected": n, "margin": total,
                         "overlap_with_gross": len(sel & base_sel) / len(base_sel) if base_sel else np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(outputs_path("robustness/ba_gross_vs_net.csv"), index=False)
    logger.info("Robustness — gross vs net (phương pháp BA-report):\n%s", df.to_string(index=False))


def robustness_monte_carlo(v: np.ndarray, seg: pd.Series, n_draws: int = 5000) -> None:
    """Phản biện (2): hệ số là giả định. Rút ngẫu nhiên đồng thời discount,
    growth, gross margin và 4 segment multiplier (x U(0.5,1.5), không ép thứ
    tự) -> kết luận "không khuyến mãi" và "không target tràn lan" vững tới đâu."""
    rng = np.random.default_rng(42)
    segs = list(MULT)
    seg_idx = seg.map({s: i for i, s in enumerate(segs)}).to_numpy()
    base = np.array([MULT[s] for s in segs])
    sc_d = [0.05, 0.10, 0.20]
    sc_b = [0.10, 0.20, 0.35]
    out = {}
    for mode in ["linked", "independent"]:
        rows = []
        for _ in range(n_draws):
            d = rng.uniform(0.05, 0.20)
            b = np.interp(d, sc_d, sc_b) * rng.uniform(0.7, 1.3) if mode == "linked" else rng.uniform(0.10, 0.35)
            g = rng.uniform(0.30, 0.50)
            mult = base * rng.uniform(0.5, 1.5, size=4)
            alpha = (1 + b * mult) * (g - d) - g
            delta = v * alpha[seg_idx] - C
            n, total = proposed(delta)
            rand_mean = delta.mean() * 1500
            rows.append({"n_selected": n, "proposed_margin": total, "random_1500_margin": rand_mean,
                         "max_alpha": alpha.max()})
        sims = pd.DataFrame(rows)
        out[mode] = {
            "P(policy không chọn ai)": (sims["n_selected"] == 0).mean(),
            "P(mọi segment có α < 0)": (sims["max_alpha"] < 0).mean(),
            "P(chọn ngẫu nhiên 1.500 khách bị lỗ)": (sims["random_1500_margin"] < 0).mean(),
            "Median margin khi có chọn (£)": sims.loc[sims["n_selected"] > 0, "proposed_margin"].median(),
            "P95 margin proposed (£)": sims["proposed_margin"].quantile(0.95),
            "P(proposed >= random)": (sims["proposed_margin"] >= sims["random_1500_margin"]).mean(),
        }
    summary = pd.DataFrame(out)
    summary.to_csv(outputs_path("robustness/ba_monte_carlo.csv"))
    logger.info("Robustness — Monte Carlo %d draws (phương pháp BA-report):\n%s", n_draws, summary.to_string())


if __name__ == "__main__":
    main()
