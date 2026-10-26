# Traceability Map

This supersedes the former mixed-sample/blanket-policy numerical recommendations.
Corrected UCI results are pending raw-data execution; old artifacts are identified
in `outputs/RESULTS_STATUS.md`. No synthetic test metric is a capstone result.

## Data → snapshots → validation

- **Source:** UCI Online Retail II workbook; `docs/data_dictionary.md` records URL,
  access date, cleaning, duplicate-retention policy and gross-positive-sales semantics.
- **Implementation:** `src/02_cleaning_features.py::audit_duplicates`,
  `build_temporal_snapshots`, `snapshot_schedule`, `temporal_partitions`.
- **Assumptions/dates:** expanding history from 2009-12-01; train decisions
  2010-12-01/2011-03-01, validation 2011-06-01, final test 2011-09-10; horizon 90 days.
- **Validation:** raw coverage and schedule checks reject incomplete outcome windows
  or training labels finishing at/after the later decision. Tests mutate final
  outcomes and verify final predictions do not change.
- **Outputs:** `outputs/data_quality.csv`, `outputs/temporal_schedule.csv`,
  `data/processed/customer_snapshots.csv`.

## Prediction → final-test policy inputs

- **Implementation:** `src/03_baseline_models.py::evaluate_temporal_models` fits
  earlier train snapshots for validation, then train+validation for final test.
- **Baseline:** train-only StandardScaler + LogisticRegression; LinearRegression;
  RFM and naive comparators. No random customer split.
- **Optional:** `src/04_advanced_models.py` uses the same temporal evaluator for GBM,
  preserving calibration, SHAP and subgroup diagnostics.
- **Outputs:** `models/temporal_registry.json`,
  `data/processed/customer_test_predictions.csv`, optional
  `data/processed/customer_test_predictions_advanced.csv`.
- **Validation:** `src/utils.py::validate_final_test_predictions` rejects legacy,
  mixed-snapshot and deployment rows. Model source is fixed in configuration,
  not selected by final-test accuracy. Clipped predictions are evaluated and used
  in business logic; unconstrained predictions remain diagnostic fields.

## Economics → no campaign under the current assumptions

- **Inputs:** final-test clipped future gross sales under historical behavior,
  not untreated revenue; `config/params.yaml::decision` contains all assumptions.
- **Implementation:** `src/05_decision_scenario.py::incremental_margin`:
  `v*((1+r)*(g-d)-g) - contact_cost`. Underlying unit COGS is unchanged.
- **Assumptions:** g=0.40; d/r pairs 0.05/0.10, 0.10/0.20, 0.20/0.35;
  segment multipliers 0.5/1.0/1.3/0.4; contact cost £0.50.
- **Conclusion:** the maximum pre-contact coefficient in each scenario is negative
  (-0.0045, -0.022, -0.109). Thus zero selected contacts, £0 contact cost and £0
  simulated NET campaign change follow for any nonnegative predictions. This is
  algebra under assumptions, not an observed UCI campaign result.
- **Tests:** known promoted-revenue/COGS accounting example, contact-cost deduction,
  negative forecast clipping and absence of sign-reversal recommendations.

## Fair policy comparison → decisions/dashboard

- **Implementation:** `policy_selections` and `build_scenario_table` compare seeded
  random, RFM score/monetary ranking, and descending positive NET contributions.
- **Constraint:** same actual N <= floor(£750/£0.50)=1,500 for every policy within a
  scenario. Random selection is seeded and customer order canonicalized.
- **Outputs:** `outputs/scenario_table.csv` (three scenarios × three policies),
  `data/processed/customer_test_scores.csv` (decision time/source/scenario,
  before-contact and NET contribution, cost, rank, action, selection).
- **Actions:** positive net and value → nurture; positive-value Champions/Loyal
  with nonpositive net → retain; otherwise no-offer. No intervention effect is
  inferred from these labels.
- **Dashboard:** `dashboard/app.py` accepts final-test scores only; shows equal-N
  NET comparisons. Optional API in `src/serve.py` is explicitly deployment/demo,
  has no campaign-wide budget allocation, and never supplies evaluation metrics.

## Sensitivity and verification

- `tornado_sensitivity` varies response 0.10–0.35, discount 0.05–0.20, gross margin
  0.30–0.50, At-risk multiplier 0.91–1.69 around moderate; recomputes selected
  budget-constrained NET margin. Table schema is retained.
- Outputs: `outputs/tornado_sensitivity.csv`, `outputs/figures/tornado_sensitivity.png`.
- `tornado_bar_bounds` guarantees nonnegative widths for reversed endpoints.
- `src/06_final_qa.py`: baseline 02→03→05 (04 only for advanced source), tests,
  temporal provenance, nonnegative values and recomputed scenario consistency.
- Relevant tests: `tests/test_features.py`, `tests/test_temporal_evaluation.py`,
  `tests/test_decision_scenario.py`, `tests/test_pipeline_idempotency.py`.
