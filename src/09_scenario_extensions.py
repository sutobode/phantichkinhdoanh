"""Scenario extensions — trả lời câu hỏi "model đóng góp gì vào quyết định?".

Ba kịch bản cấu hình trong config/params.yaml đều có hệ số
  alpha_z = (1 + b*m_z)(g - d) - g < 0  cho MỌI segment,
nên proposed policy chọn 0 khách bất kể dự báo. Script này bổ sung (KHÔNG thay
đổi pipeline 02→07 hay config):

  A1. Kịch bản minh họa neo theo ngưỡng hòa vốn (growth vừa vượt
      d/(g-d)/m_At-risk) -> bảng so sánh No-promo / Random / RFM / Proposed
      bằng đúng build_scenario_table của 05_decision_scenario.py.
  A2. Ablation nguồn dự báo: giữ nguyên rule chọn, chỉ thay v_i bằng
      constant / historical monetary / Linear (baseline) / GBM (advanced) /
      oracle (doanh số thật) -> đo phần giá trị do model tạo ra.
  A3. Realized replay: danh sách chọn theo DỰ BÁO được tính lại bằng
      future_gross_sales THỰC TẾ của final test (vẫn dưới giả định
      discount/growth — không phải causal effect).

Mọi con số vẫn là scenario-based, non-causal (Scope Guardrail của đề).

Yêu cầu: đã chạy 03 (baseline) và 04 (advanced) để có customer_test_predictions*.csv.
Chạy: python src/09_scenario_extensions.py
"""
from __future__ import annotations

import copy
from importlib import import_module

import numpy as np
import pandas as pd

from utils import get_logger, load_config, outputs_path, processed_path, set_seed

logger = get_logger("09_scenario_extensions")
decision = import_module("05_decision_scenario")

VALUE_COL = decision.VALUE_COL
ACTUAL_COL = "future_gross_sales"
N_TIE_SHUFFLES = 200

# Neo theo ngưỡng hòa vốn của segment At-risk (m=1.3, g=0.40):
#   d=5%  -> cần b > 0.05/0.35/1.3 = 11.0%   ; chọn 20%
#   d=10% -> cần b > 0.10/0.30/1.3 = 25.6%   ; chọn 35%
#   d=15% -> cần b > 0.15/0.25/1.3 = 46.2%   ; chọn 50%
# Trùng các điểm đã báo cáo ở Section 8.2 (sensitivity), không phải giả định mới.
ILLUSTRATIVE_SCENARIOS = [
    {"name": "illustrative_light", "discount_pct": 0.05, "response_rate": 0.20},
    {"name": "illustrative_standard", "discount_pct": 0.10, "response_rate": 0.35},
    {"name": "illustrative_deep", "discount_pct": 0.15, "response_rate": 0.50},
]


def out(name: str):
    return outputs_path(f"scenario_extensions/{name}")


def load_inputs(config: dict) -> pd.DataFrame:
    """Baseline scores (đã validate bởi 05.prepare_scores) + GBM prediction
    cùng final-test population, ghép theo Customer ID."""
    base = decision.load_scores(config)
    adv_path = processed_path("customer_test_predictions_advanced.csv", config)
    adv = pd.read_csv(adv_path)[["Customer ID", "predicted_future_value"]]
    adv = adv.rename(columns={"predicted_future_value": "predicted_future_value_gbm"})
    df = base.merge(adv, on="Customer ID", how="left", validate="one_to_one")
    if df["predicted_future_value_gbm"].isna().any():
        raise ValueError("Advanced predictions không khớp final-test population — chạy lại 04")
    return df


def net_margin(df: pd.DataFrame, value: pd.Series, scenario: dict, config: dict) -> pd.Series:
    dec = config["decision"]
    growth = decision.effective_response_rate(df, scenario["response_rate"], dec["segment_response_multiplier"])
    return decision.incremental_margin(value, growth, scenario["discount_pct"], dec["gross_margin_pct"],
                                       dec["contact_cost_per_customer"])


def capacity(df: pd.DataFrame, config: dict) -> int:
    dec = config["decision"]
    return min(int(dec["budget_total"] // dec["contact_cost_per_customer"]), len(df))


def break_even_table(config: dict, scenarios: list[dict]) -> pd.DataFrame:
    dec = config["decision"]
    g = dec["gross_margin_pct"]
    rows = []
    for s in scenarios:
        d, b = s["discount_pct"], s["response_rate"]
        for seg, m in dec["segment_response_multiplier"].items():
            alpha = (1 + b * m) * (g - d) - g
            rows.append({
                "scenario": s["name"], "segment": seg, "discount_pct": d, "base_growth": b,
                "effective_growth": b * m, "required_effective_growth": d / (g - d),
                "alpha_per_gbp": alpha, "min_value_to_cover_contact": (dec["contact_cost_per_customer"] / alpha
                                                                      if alpha > 0 else np.nan),
            })
    return pd.DataFrame(rows)


def realized_policy_table(df: pd.DataFrame, scenarios: list[dict], config: dict) -> pd.DataFrame:
    """A1 + A3: chọn danh sách bằng margin DỰ BÁO, chấm điểm bằng margin THỰC TẾ."""
    k = capacity(df, config)
    rows = []
    for s in scenarios:
        pred = net_margin(df, df[VALUE_COL], s, config)
        real = net_margin(df, df[ACTUAL_COL], s, config)
        policies = decision.policy_selections(df, pred, config)
        oracle = decision.budget_constrained_selection(real, config["decision"]["budget_total"],
                                                       config["decision"]["contact_cost_per_customer"])
        oracle_total = float(real[oracle].sum())
        for policy, selected in policies.items():
            n = int(selected.sum())
            if policy == "non_targeted":
                # Kỳ vọng của 1 tập ngẫu nhiên K phần tử = K * mean (chính xác, không cần mô phỏng)
                pred_total, real_total = k * pred.mean(), k * real.mean()
            else:
                pred_total, real_total = float(pred[selected].sum()), float(real[selected].sum())
            rows.append({
                "scenario": s["name"], "discount_pct": s["discount_pct"], "base_growth": s["response_rate"],
                "policy": policy, "customers_targeted": n,
                "simulated_net_margin_predicted": pred_total,
                "simulated_net_margin_on_actual_sales": real_total,
                "oracle_net_margin_on_actual_sales": oracle_total,
                "share_of_oracle": real_total / oracle_total if oracle_total > 0 else np.nan,
            })
    return pd.DataFrame(rows)


def expected_realized_with_ties(score: np.ndarray, real: np.ndarray, k: int, rng) -> float:
    """Rule chọn: margin dự báo > 0, top-K. Khi nhiều khách có cùng score (vd
    v = hằng số), thứ tự trong nhóm hòa là tùy ý -> lấy kỳ vọng qua nhiều lần
    xáo trộn thay vì phụ thuộc thứ tự Customer ID."""
    eligible = np.flatnonzero(score > 0)
    if len(eligible) == 0:
        return 0.0
    if len(np.unique(np.round(score[eligible], 9))) == len(eligible):
        pick = eligible[np.argsort(-score[eligible], kind="stable")[:k]]
        return float(real[pick].sum())
    totals = []
    for _ in range(N_TIE_SHUFFLES):
        perm = rng.permutation(eligible)
        pick = perm[np.argsort(-score[perm], kind="stable")[:k]]
        totals.append(real[pick].sum())
    return float(np.mean(totals))


def prediction_source_ablation(df: pd.DataFrame, scenarios: list[dict], config: dict) -> pd.DataFrame:
    """A2: cùng rule, chỉ đổi nguồn v_i. Các nguồn không-model được scale về
    cùng tổng với dự báo Linear để ngưỡng c/alpha so sánh công bằng."""
    k = capacity(df, config)
    rng = np.random.default_rng(config["seed"])
    total_pred = df[VALUE_COL].sum()
    sources = {
        "no_model_constant": pd.Series(total_pred / len(df), index=df.index),
        "historical_monetary_scaled": df["monetary"] * total_pred / df["monetary"].sum(),
        "linear_baseline (báo cáo)": df[VALUE_COL],
        "gbm_advanced": df["predicted_future_value_gbm"].clip(lower=0),
        "oracle_actual_sales": df[ACTUAL_COL],
    }
    rows = []
    for s in scenarios:
        real = net_margin(df, df[ACTUAL_COL], s, config).to_numpy()
        oracle_total = expected_realized_with_ties(real, real, k, rng)
        for name, value in sources.items():
            score = net_margin(df, value, s, config).to_numpy()
            realized = expected_realized_with_ties(score, real, k, rng)
            rows.append({
                "scenario": s["name"], "value_source": name,
                "customers_eligible": int((score > 0).sum()),
                "customers_targeted": int(min((score > 0).sum(), k)),
                "simulated_net_margin_on_actual_sales": realized,
                "share_of_oracle": realized / oracle_total if oracle_total > 0 else np.nan,
            })
    return pd.DataFrame(rows)


def main() -> None:
    config = load_config()
    set_seed(config["seed"])
    df = load_inputs(config)
    configured = config["decision"]["discount_response_scenarios"]
    all_scenarios = configured + ILLUSTRATIVE_SCENARIOS
    logger.info("Final-test customers: %d | capacity K=%d", len(df), capacity(df, config))

    be = break_even_table(config, all_scenarios)
    be.to_csv(out("break_even_alpha.csv"), index=False)
    logger.info("Hệ số alpha theo segment:\n%s", be.to_string(index=False))

    # A1: dùng nguyên build_scenario_table của 05 cho kịch bản minh họa
    illus_cfg = copy.deepcopy(config)
    illus_cfg["decision"]["discount_response_scenarios"] = ILLUSTRATIVE_SCENARIOS
    illus = decision.build_scenario_table(df, illus_cfg)
    illus.to_csv(out("illustrative_scenario_table.csv"), index=False)
    logger.info("A1. Illustrative scenarios (predicted):\n%s", illus[[
        "scenario", "policy", "customers_targeted", "simulated_net_incremental_margin"]].to_string(index=False))

    realized = realized_policy_table(df, all_scenarios, config)
    realized.to_csv(out("policy_realized_replay.csv"), index=False)
    logger.info("A3. Policy replay on actual final-test sales:\n%s", realized.to_string(index=False))

    ablation = prediction_source_ablation(df, all_scenarios, config)
    ablation.to_csv(out("prediction_source_ablation.csv"), index=False)
    logger.info("A2. Prediction-source ablation:\n%s", ablation.to_string(index=False))


if __name__ == "__main__":
    main()
