# Current results — real UCI baseline, Phase 1 + Phase 2

The corrected baseline has been rerun on UCI Online Retail II. Current evidence:

- `models/temporal_registry.json`: 5,281 final-test customers, repeat rate 43.36%;
  Logistic PR-AUC 0.750184 / ROC-AUC 0.789977; clipped Linear MAE £395.69 / RMSE
  £2,016.14. Naive PR-AUC 0.433630; MAE £636.34 / RMSE £3,288.87.
- `data/processed/customer_test_scores.csv`: current final customer-level evaluation
  table, explicitly allowed by `.gitignore`; includes temporal provenance.
- `outputs/data_quality.csv`: 1,067,371 raw rows, 34,335 exact duplicates beyond first;
  policy retain. `outputs/temporal_schedule.csv` records the established 90-day design.
- `outputs/scenario_table.csv`: No Promotion, Monte Carlo non-targeted, RFM and
  Proposed on the same final-test population. 1,000 seeded K=1,500 random campaigns
  replace the old single-draw primary benchmark. Random-policy intervals are not causal CIs.

| Scenario | Non-targeted MC mean NET | RFM NET | Proposed / No Promotion |
|---|---:|---:|---:|
| Conservative | -£17,417.59 | -£55,333.45 | £0 / £0 |
| Moderate | -£37,407.57 | -£118,314.35 | £0 / £0 |
| Aggressive | -£90,677.15 | -£277,865.97 | £0 / £0 |

The no-campaign recommendation remains unchanged. Positive results in the
exploratory sensitivity surface do not change the three main scenario assumptions.

## Reproducibility and decision evidence

Run 02 → 03 → 05, or `python src/06_final_qa.py` for the complete baseline rerun,
tests, invariants and manifest. `outputs/run_manifest.json` records actual raw,
config and requirements SHA-256 hashes, timestamp, Git commit/dirty state, Python,
platform, temporal design/source and counts. Unchanged input bytes yield stable hashes.

Phase 2 produces `break_even_by_segment.csv`, `break_even_summary.csv`,
`response_discount_surface.csv`, and `figures/response_discount_surface.png`.
The existing tornado table/figure remain current. Dashboard scenario controls act
only on fixed final-test predictions in memory; no models are retrained.

## Historical evidence, not current results

`data/processed/customer_scores.csv`, `models/registry.json`,
`outputs/rolling_origin_validation.csv`, unprefixed `calibration_*.csv`,
`shap_feature_importance.csv`, `error_by_segment.csv`,
`ethics_bias_check_by_country.csv`, and `figures/calibration_curve.png` belong to
the pre-remediation design if present. They are not used to support current claims.
No advanced-model work is part of Phase 1 + Phase 2.
