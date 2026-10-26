# Results status — temporal/economic remediation

The corrected UCI pipeline has **not been rerun in the remediation environment**:
`data/raw/online_retail_II.xlsx`, openpyxl and Pandera are unavailable. Do not report
the old numerical artifacts as results of the corrected methodology.

## Superseded artifacts

- `data/processed/customer_scores.csv`: old same-snapshot, mixed training/test scores;
  includes unconstrained negative value predictions. Never loaded by corrected policy/dashboard.
- `models/registry.json`: old random-customer-holdout metrics, not temporal metrics.
- `outputs/rolling_origin_validation.csv`: old repeated random-customer splits,
  not genuine rolling temporal evaluation.
- `outputs/calibration_*.csv`, `shap_feature_importance.csv`, `error_by_segment.csv`,
  `ethics_bias_check_by_country.csv`, `figures/calibration_curve.png`: old diagnostics.
- `outputs/scenario_table.csv`, `tornado_sensitivity.csv`, `figures/tornado_sensitivity.png`:
  old economics until replaced by corrected step 05. Old `uplift_vs_blanket` is
  not a causal metric and is absent from the new schema.
- `outputs/cohort_retention.csv`: old observation window until replaced by step 02.
- The old `outputs/final_qa_report.md` is not evidence of a corrected full-data pass.

These historical files are retained for traceability, not silently relabeled as new.

## Corrected results to generate

Run 02 → 03 → 05 with the default baseline source. New final-test-only prediction
and score filenames prevent fallback to legacy data. Model metrics are written to
`models/temporal_registry.json`; dates to `outputs/temporal_schedule.csv` and
duplicate audit to `outputs/data_quality.csv`. Step 05 overwrites the scenario and
sensitivity tables only after validating final-test provenance. Optional step 04
writes separate advanced predictions and `temporal_*` diagnostics.

The no-campaign conclusion for the unchanged three scenarios is an algebraic
consequence of the corrected economics: all coefficients are negative before
contact cost for every segment. Thus for any nonnegative revenue predictions,
all equal-N policies contact zero and have £0 simulated NET change. This is not
an empirically regenerated UCI campaign result. Corrected model accuracy, population
counts, duplicate count and sensitivity magnitudes remain pending.
