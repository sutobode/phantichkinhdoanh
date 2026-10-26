"""Week 6 — Final QA & reproducibility check.

Chạy baseline pipeline (02 -> 03 -> 05; optional 04 for advanced), pytest, và kiểm
tra các invariant quan trọng (không cột trùng, không giá trị âm vô lý, số
liệu nhất quán giữa các file) trước khi nộp bài.

Chạy: python src/06_final_qa.py
"""
from __future__ import annotations

import subprocess
import json
import sys
from importlib import import_module
from pathlib import Path

import pandas as pd

from utils import (PROJECT_ROOT, CONFIG_PATH, get_logger, load_config, outputs_path, processed_path,
                   raw_path, validate_final_test_predictions, build_run_manifest, sha256_file)

logger = get_logger("06_final_qa")

STEPS = ["02_cleaning_features", "03_baseline_models", "05_decision_scenario"]


def run_pipeline_from_scratch() -> list[str]:
    steps = STEPS.copy()
    if load_config()["decision"]["value_prediction_source"] == "advanced":
        steps.insert(2, "04_advanced_models")
    for step in steps:
        logger.info("Chạy %s ...", step)
        module = import_module(step)
        module.main()
    logger.info("Completed temporal pipeline: %s", steps)
    return steps


def run_pytest() -> bool:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(PROJECT_ROOT / "tests"), "-v"],
        capture_output=True, text=True,
    )
    logger.info("pytest output:\n%s", result.stdout[-3000:])
    if result.returncode != 0:
        logger.error("pytest FAILED:\n%s", result.stderr[-2000:])
    return result.returncode == 0


def check_invariants(config: dict) -> list[str]:
    """Trả về list các vấn đề phát hiện được (rỗng nếu mọi thứ OK)."""
    issues = []

    features = pd.read_csv(processed_path("customer_features.csv", config))
    labels = pd.read_csv(processed_path("customer_labels.csv", config))
    scores = pd.read_csv(processed_path("customer_test_scores.csv", config))
    try:
        validate_final_test_predictions(scores, config)
    except ValueError as exc:
        issues.append(str(exc))
    if scores["baseline_future_revenue"].isna().any() or (scores["baseline_future_revenue"] < 0).any():
        issues.append("Negative/missing business input revenue")
    if ((scores["baseline_future_revenue"] <= 0) & scores["final_recommendation"].eq("nurture")).any():
        issues.append("Non-positive revenue must not receive a nurture recommendation")

    # 1. Không cột trùng (_x/_y) trong customer_scores.csv
    dup_cols = [c for c in scores.columns if c.endswith("_x") or c.endswith("_y")]
    if dup_cols:
        issues.append(f"customer_scores.csv có cột trùng do merge lỗi: {dup_cols}")

    # 2. Số khách hàng nhất quán giữa features/labels/scores
    if not (len(features) == len(labels) == len(scores)):
        issues.append(
            f"Số khách hàng KHÔNG khớp giữa 3 file: features={len(features)}, "
            f"labels={len(labels)}, scores={len(scores)}"
        )

    # 3. recency_days / tenure_days không được âm
    if (features["recency_days"] < 0).any() or (features["tenure_days"] < 0).any():
        issues.append("Có recency_days hoặc tenure_days âm — lỗi tính as_of/window boundary")

    # 4. return_rate phải trong [0, 1]
    if not features["return_rate"].between(0, 1).all():
        issues.append("Có return_rate ngoài khoảng [0, 1]")

    # 5. final_recommendation chỉ có 3 giá trị hợp lệ
    valid_recs = {"retain", "nurture", "no-offer"}
    invalid = set(scores["final_recommendation"].unique()) - valid_recs
    if invalid:
        issues.append(f"final_recommendation có giá trị không hợp lệ: {invalid}")

    # 6. repeat_purchase label phải nhất quán với future_value > 0
    merged = labels.copy()
    inconsistent = merged[(merged["future_gross_sales"] > 0) != (merged["repeat_purchase"] == 1)]
    if len(inconsistent) > 0:
        issues.append(f"{len(inconsistent)} dòng label repeat_purchase không khớp future_value>0")

    # 7. scenario_table.csv tồn tại và có >= 3 scenario (Baseline Required của đề)
    scenario_table = pd.read_csv(outputs_path("scenario_table.csv"))
    if scenario_table["scenario"].nunique() < 3:
        issues.append(f"scenario_table.csv chỉ có {len(scenario_table)} scenario (yêu cầu >= 3)")

    decision = import_module("05_decision_scenario")
    try:
        pd.testing.assert_frame_equal(scenario_table, decision.build_scenario_table(scores, config),
                                      check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-8)
    except (AssertionError, ValueError) as exc:
        issues.append(f"Scenario provenance/cost/capacity mismatch: {exc}")
    for _, policies in scenario_table.groupby("scenario"):
        capacity_k = min(int(config["decision"]["budget_total"] // config["decision"]["contact_cost_per_customer"]), len(scores))
        baselines = policies[policies["policy"].isin(["non_targeted", "rfm"])]
        no_promotion = policies[policies["policy"].eq("no_promotion")]
        if (set(policies["policy"]) != {"no_promotion", "non_targeted", "rfm", "proposed"}
                or not policies["campaign_capacity"].eq(capacity_k).all()
                or not baselines["customers_targeted"].eq(capacity_k).all()
                or not policies["customers_targeted"].between(0, capacity_k).all()
                or (policies["campaign_contact_cost"] > config["decision"]["budget_total"]).any()):
            issues.append("Policies must share maximum capacity; Random/RFM use K, Proposed uses at most K")
        if not no_promotion[["customers_targeted", "campaign_contact_cost", "simulated_net_incremental_margin"]].eq(0).all().all():
            issues.append("No Promotion must be exactly zero")
    break_even_segment, break_even_summary = decision.build_break_even_tables(scores, config)
    for name, expected in (
        ("break_even_by_segment.csv", break_even_segment),
        ("break_even_summary.csv", break_even_summary),
        ("response_discount_surface.csv", decision.response_discount_surface(scores, config)),
    ):
        try:
            pd.testing.assert_frame_equal(pd.read_csv(outputs_path(name)), expected,
                                          check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-8)
        except AssertionError as exc:
            issues.append(f"Decision analysis output mismatch ({name}): {exc}")
    return issues


def write_run_manifest(config: dict) -> None:
    scores = pd.read_csv(processed_path("customer_test_scores.csv", config))
    raw_file = raw_path("online_retail_II.xlsx", config)
    quality = pd.read_csv(outputs_path("data_quality.csv"))
    manifest = build_run_manifest(config, raw_file, CONFIG_PATH, quality.iloc[0]["raw_rows"],
                                  scores, PROJECT_ROOT / "requirements.txt")
    path = outputs_path("run_manifest.json")
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    saved = json.loads(path.read_text(encoding="utf-8"))
    if (saved != manifest or saved["raw_data_sha256"] != sha256_file(raw_file)
            or saved["config_sha256"] != sha256_file(CONFIG_PATH)):
        raise ValueError("Run manifest failed round-trip/hash validation")


def main() -> None:
    config = load_config()

    logger.info("=== BƯỚC 1: Chạy lại toàn bộ pipeline từ đầu ===")
    steps = run_pipeline_from_scratch()

    logger.info("=== BƯỚC 2: Chạy pytest ===")
    tests_passed = run_pytest()

    logger.info("=== BƯỚC 3: Kiểm tra invariants ===")
    issues = check_invariants(config)
    if tests_passed and not issues:
        write_run_manifest(config)

    report_lines = [
        "# Final QA Report",
        "",
        f"- Temporal pipeline re-run: OK ({len(steps)} steps; source={config['decision']['value_prediction_source']})",
        f"- pytest: {'PASSED' if tests_passed else 'FAILED — xem log outputs/logs/06_final_qa.log'}",
        f"- Invariant checks: {'PASSED — không phát hiện vấn đề' if not issues else 'FAILED'}",
        f"- Run manifest: {'GENERATED and hash-validated — outputs/run_manifest.json' if tests_passed and not issues else 'NOT GENERATED for this run'}",
    ]
    if issues:
        report_lines.append("\n## Vấn đề phát hiện được:")
        for issue in issues:
            report_lines.append(f"- ❌ {issue}")
    report_path = PROJECT_ROOT / "outputs" / "final_qa_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    logger.info("Đã ghi final QA report: %s", report_path)

    if not tests_passed or issues:
        logger.error("❌ FINAL QA FAILED — xem %s trước khi nộp bài.", report_path)
        sys.exit(1)
    logger.info("✅ FINAL QA PASSED — project sẵn sàng nộp bài.")


if __name__ == "__main__":
    main()
