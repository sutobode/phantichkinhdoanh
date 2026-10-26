# Traceability Map

This supersedes the former mixed-sample/blanket-policy numerical recommendations.
The real UCI baseline rerun is complete: 5,281 final-test customers, 43.36% repeat
purchase; PR-AUC 0.750184, ROC-AUC 0.789977, MAE £395.69, RMSE £2,016.14.
Evidence: `models/temporal_registry.json` and `data/processed/customer_test_scores.csv`.
Historical artifacts are identified in `outputs/RESULTS_STATUS.md`.

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
- **Observed counts:** 1,067,371 raw rows; 34,335 exact duplicates beyond first,
  retained. `customer_test_scores.csv` is the current customer-level evaluation
  artifact, explicitly allowed by `.gitignore`; intermediate processed files stay ignored.

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
  simulated NET campaign change follow for the proposed policy with any nonnegative
  predictions. Random/RFM use capacity K and incur simulated losses when K>0;
  the rerun confirms this using final-test predictions. This is a simulation on
  real-data forecasts, not an observed campaign effect.
- **Tests:** known promoted-revenue/COGS accounting example, contact-cost deduction,
  negative forecast clipping and absence of sign-reversal recommendations.

## Fair policy comparison → decisions/dashboard

- **Implementation:** `monte_carlo_random_policy`, `policy_selections` and
  `build_scenario_table` compare No Promotion (£0), 1,000 seeded uniform random
  K-subsets, RFM score/monetary ranking, and descending positive NET contributions.
- **Constraint:** same maximum K=min(floor(£750/£0.50), final-test population size).
  Random/RFM use K; proposed ranks positive NET contributions and uses at most K,
  possibly zero. Actual contact cost is deducted separately. Random selection is
  seeded and customer order canonicalized. All share scenario assumptions.
- **Outputs:** `outputs/scenario_table.csv` (three scenarios × four policies),
  `data/processed/customer_test_scores.csv` (decision time/source/scenario,
  before-contact and NET contribution, cost, rank, action, selection).
- **Main non-targeted benchmark:** MC mean with 2.5th–97.5th random-policy simulation
  percentiles; not a causal CI. Conservative -£17,417.59 [-£21,707.94, -£14,023.76];
  moderate -£37,407.57 [-£46,583.42, -£30,081.21]; aggressive -£90,677.15
  [-£112,076.51, -£73,360.59]. RFM totals respectively -£55,333.45, -£118,314.35,
  -£277,865.97; Proposed and No Promotion £0. The old single draw is no longer the benchmark.
- **Actions:** positive net and value → nurture; positive-value Champions/Loyal
  with nonpositive net → retain; otherwise no-offer. No intervention effect is
  inferred from these labels.
- **Dashboard:** `dashboard/app.py` accepts final-test scores only; shows equal-maximum-capacity
  NET comparisons. Optional API in `src/serve.py` is explicitly deployment/demo,
  has no campaign-wide budget allocation, and never supplies evaluation metrics.

## Sensitivity and verification

- `tornado_sensitivity` varies response 0.10–0.35, discount 0.05–0.20, gross margin
  0.30–0.50, At-risk multiplier 0.91–1.69 around moderate; recomputes selected
  budget-constrained NET margin. Table schema is retained.
- Outputs: `outputs/tornado_sensitivity.csv`, `outputs/figures/tornado_sensitivity.png`.
- Observed conditional simulation: discount 5% → £7,175.92; growth 35% → £3,990.80;
  gross margin 50% → £143.77; At-risk multiplier 1.69 → £15.05, each varied alone.
- `build_break_even_tables` uses the analytical zero-NET equations and exports
  `outputs/break_even_by_segment.csv` / `break_even_summary.csv`. At-risk median
  required base growth at 10% discount is 26.75%; median supportable discount at
  20% base growth is 7.91%, conditional on positive predicted value.
- `response_discount_surface` exports 20 cells plus one heatmap. First profitable
  tested combinations: 5%/20% growth (£7,175.92), 10%/30% (£1,424.27), 15%/50%
  (£962.39); none at 20% discount through 50% growth. All outputs are scenario-labeled.
- `evaluate_proposed_scenario` powers the dashboard sandbox in memory. Controls
  do not retrain models or change saved global comparisons. Filtered-population scope
  and exploratory/non-causal interpretation are explicitly displayed.
- `tornado_bar_bounds` guarantees nonnegative widths for reversed endpoints.
- `src/06_final_qa.py`: baseline 02→03→05 (04 only for advanced source), tests,
  temporal provenance, nonnegative values and recomputed scenario consistency.
- On success, QA writes and validates `outputs/run_manifest.json`: programmatic
  raw/config/requirements SHA-256, timestamp, Git commit/dirty state, Python/platform,
  final decision/horizon/source, raw rows and final customers. Hash stability is tested.
- Relevant tests: `tests/test_features.py`, `tests/test_temporal_evaluation.py`,
  `tests/test_decision_scenario.py`, `tests/test_pipeline_idempotency.py`.
