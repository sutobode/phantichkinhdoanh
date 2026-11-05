"""Kiểm tra output đã sinh so với giá trị kỳ vọng (docs/VERIFICATION_HANDOFF.md,
docs/assumption_register.md). Chỉ ĐỌC output, không chạy model.

Chạy từ gốc repo sau 06 → 04 → 08 → 09 → 10 → 11:
  python scripts/check_results.py            # in bảng PASS/FAIL, exit 1 nếu có FAIL
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
rows: list[dict] = []


def check(name: str, actual, expected, tol: float = 0.0) -> None:
    try:
        ok = abs(float(actual) - float(expected)) <= tol
    except (TypeError, ValueError):
        ok = actual == expected
    rows.append({"check": name, "actual": actual, "expected": expected, "tol": tol, "status": "PASS" if ok else "FAIL"})


def guarded(name: str, fn) -> None:
    try:
        fn()
    except Exception as exc:  # file thiếu / cột đổi -> FAIL rõ ràng thay vì crash
        rows.append({"check": name, "actual": f"ERROR: {exc}", "expected": "-", "tol": "-", "status": "FAIL"})


def qa():
    text = (OUT / "final_qa_report.md").read_text(encoding="utf-8")
    check("final QA: pytest PASSED", "pytest: PASSED" in text, True)
    check("final QA: invariants PASSED", "Invariant checks: PASSED" in text, True)


def verification():
    v = pd.read_csv(OUT / "ba_report_verification.csv")
    check("BA-report numbers reproduced", int(v["match"].sum()), 78)


def models():
    m = pd.read_csv(OUT / "temporal_model_comparison.csv").set_index(["split", "model"])
    check("final AP logistic", round(m.loc[("final_test", "baseline"), "pr_auc"], 4), 0.7502, 0.0002)
    check("final MAE linear", round(m.loc[("final_test", "baseline"), "mae"], 2), 395.69, 1.0)
    check("final AP GBM", round(m.loc[("final_test", "advanced"), "pr_auc"], 4), 0.7691, 0.0005)


def scenarios():
    s = pd.read_csv(OUT / "scenario_table.csv")
    proposed = s[s["policy"] == "proposed"]
    check("configured scenarios: proposed selects 0", int(proposed["customers_targeted"].sum()), 0)
    rfm = s[(s["policy"] == "rfm") & (s["scenario"] == "moderate")]["simulated_net_incremental_margin"].iloc[0]
    check("moderate RFM net margin", round(rfm, 2), -118314.35, 1.0)


def extensions():
    a = pd.read_csv(OUT / "scenario_extensions/prediction_source_ablation.csv")
    light = a[(a["scenario"] == "illustrative_light") & a["value_source"].str.startswith("linear")]
    check("ablation light: linear share of oracle", round(light["share_of_oracle"].iloc[0], 3), 0.785, 0.01)
    t = pd.read_csv(OUT / "analysis_extensions/two_part_metrics.csv")
    lin = t[(t["split"] == "final_test") & t["value_model"].str.startswith("linear")]
    check("two-part table: linear final MAE", round(lin["mae"].iloc[0], 2), 395.69, 1.0)


def assumptions():
    g = pd.read_csv(OUT / "assumption_robustness.csv")
    grid = g[g["family"] == "grid"]
    check("assumption grid size", len(grid), 192)
    check("assumption grid launches", int(grid["launch"].sum()), 70, 2)
    check("reference combination abstains", bool(grid[grid["is_reference"]]["launch"].iloc[0]), False)
    check("proposed >= random K in every combination",
          bool((grid["proposed_net_margin"] >= grid["random_k_expected_net_margin"]).all()), True)
    tc = pd.read_csv(OUT / "assumption_robustness/temporal_counts.csv").set_index("item")["count"]
    check("main final fit snapshots", int(tc["final fit = train + validation snapshots"]), 13736)
    check("rolling 2011-09-10 n_train", int(tc["origin 2011-09-10 n_train (code: strict <)"]), 19760)
    check("rolling 2011-03-01 n_train", int(tc["origin 2011-03-01 n_train (code: strict <)"]), 6024)
    tie = pd.read_csv(OUT / "assumption_robustness/rfm_tie_effect.csv")
    check("RFM segment changes under average rank", int(tie["segment_changes_if_average_rank"].iloc[0]), 193)
    rc = pd.read_csv(OUT / "assumption_robustness/recency_check.csv")
    check("recency check matches", int(rc.iloc[0, 1]), 5281)
    mc = pd.read_csv(OUT / "assumption_robustness/monte_carlo_convergence.csv")
    check("MC 5000 draws max rel error < 1%", bool(mc[mc["draws"] == 5000]["rel_error_vs_analytic"].max() < 0.01), True)
    f = pd.read_csv(OUT / "assumption_robustness/net_proxy_feasibility.csv").set_index("item")["value"]
    check("cancelled value / gross (final test)", round(f["cancelled value / gross sales"], 4), 0.0355, 0.001)


if __name__ == "__main__":
    for name, fn in [("qa", qa), ("verification", verification), ("models", models), ("scenarios", scenarios),
                     ("extensions", extensions), ("assumptions", assumptions)]:
        guarded(name, fn)
    table = pd.DataFrame(rows)
    sys.stdout.reconfigure(encoding="utf-8")
    print(table.to_string(index=False))
    n_fail = int((table["status"] == "FAIL").sum())
    print(f"\nRESULT: {len(table) - n_fail}/{len(table)} PASS")
    sys.exit(1 if n_fail else 0)
