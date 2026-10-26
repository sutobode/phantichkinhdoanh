# Model / Analysis Card

## Business decision

Which customers should receive retention/promotion attention without wasting
discounts? Outputs are retain / nurture / no-offer and a contact-budget selection.
No campaign is a valid recommendation when every simulated net contribution is negative.

## Unit, targets and horizon

- Unit: customer snapshot with at least one valid observation purchase.
- Classification: `repeat_purchase = int(future_gross_sales > 0)`.
- Regression: `future_gross_sales`, positive non-cancelled sales over **90 days**,
  including zero for no purchase. Not net sales after returns, profit or lifetime CLV.
- Expanding observation begins 2009-12-01 and stops strictly before each decision.

| Role | Decision dates | Outcome end (exclusive) |
|---|---|---|
| Train | 2010-12-01; 2011-03-01 | 2011-03-01; 2011-05-30 |
| Validation | 2011-06-01 | 2011-08-30 |
| Final test | 2011-09-10 | 2011-12-09 |

Full inclusive windows are in `docs/problem_statement.md`. Raw coverage is checked;
the potentially incomplete final recorded day (2011-12-09) is not used in labels.

## Features

Seven numeric inputs, calculated only from each observation window:
`recency_days, frequency, monetary, avg_order_value, n_distinct_products,
tenure_days, return_rate`. Formulas: `docs/data_dictionary.md`.
Country and RFM are diagnostic/decision fields, not predictive inputs.

## Models and preprocessing

- Required classifier: StandardScaler → LogisticRegression(max_iter=1000, seed=42).
  Scaling is fitted only on training snapshots inside the model pipeline.
- Required regressor: LinearRegression.
- Naive comparators: historical training repeat rate / mean future gross sales.
- Optional comparison: GradientBoostingClassifier and GradientBoostingRegressor,
  sklearn defaults with seed=42. No advanced model is required for scenarios.
- Regression evaluation and simulation use `max(prediction, 0)`. Raw predictions
  are retained separately for diagnostics; clipping prevents negative-value sign reversal.
- No hyperparameter search. `decision.value_prediction_source` defaults to baseline
  and must be fixed before inspecting final-test metrics. No automatic winner claim.

## Temporal validation

1. Fit on both train snapshots to evaluate the later validation snapshot.
2. Refit the same predeclared model on train + validation snapshots, whose labels
   are completely observed by 2011-08-30, before final decision 2011-09-10.
3. Predict only the final temporal-test snapshot for final metrics and policy simulation.

The same customer may appear at earlier origins; that is acceptable historical
information. Final-test labels never enter fitting, scaling or selection.
There is **no random same-snapshot customer holdout**. This replaces the old
cross-sectional evaluation; old scores/intervals are not temporal-test evidence.

Metrics: PR-AUC (average precision), ROC-AUC, MAE and RMSE; 1,000 seeded customer
bootstrap resamples conditional on the fitted model. Calibration CSV is generated
for the baseline. Optional step 04 retains comparative reliability plotting, SHAP,
RFM-segment error analysis and country-group probability-error diagnostics.

## Current result status

**Corrected UCI predictive metrics and population counts are pending a raw-data
rerun.** The remediation environment lacks `data/raw/online_retail_II.xlsx`,
openpyxl and Pandera; no replacement accuracy figures are invented.
Synthetic integration tests verify mechanics, not predictive performance.

Corrected runs write `models/temporal_registry.json` (separate validation/final-test
metrics), `outputs/temporal_model_comparison.csv` when advanced is run, and
`data/processed/customer_test_predictions*.csv`. Old `models/registry.json` and
unprefixed diagnostic artifacts are historical only; see `outputs/RESULTS_STATUS.md`.

## Promotion assumptions and economics

- `v` / `baseline_future_revenue`: clipped predicted future gross sales under
  historical behavior, not a no-promotion counterfactual.
- `g` / `gross_margin_pct`: 0.40; baseline COGS = v(1-g).
- `d` / `discount_pct`: 0.05, 0.10, 0.20 across conservative/moderate/aggressive.
- `response_rate`: 0.10, 0.20, 0.35, **assumed proportional increase in pre-discount
  sales volume**, not a response probability. Effective r multiplies by segment:
  Champions 0.5, Loyal 1.0, At-risk 1.3, Lost 0.4. These are not learned effects.
- Unit product costs do not change with discount: promoted revenue v(1+r)(1-d),
  promoted COGS v(1+r)(1-g), baseline margin vg.
- Before-contact incremental margin = `v*((1+r)*(g-d)-g)`.
- NET incremental margin = before-contact margin − £0.50/contact.
- Contact budget £750; maximum 1,500. Discounts affect profit, not this contact budget.

Policies on the same final-test population: seeded random non-targeted, descending
RFM sum (then monetary, then customer ID), and descending positive NET margin.
All share maximum K=min(floor(contact budget/contact cost), evaluation population).
Random and RFM contact K customers regardless of proposed eligibility. Proposed
selects only positive NET contributions, ranked descending, up to K and may select
zero. Actual contact costs are deducted for each policy. NET margin/contact is
reported as 0 by convention for a zero-contact policy. No over-budget blanket
comparison is used.

With the unchanged three scenarios, even the highest multiplier produces negative
before-contact coefficients: -0.0045, -0.022, -0.109 respectively. Thus for **any
nonnegative v**, all three proposed campaigns select zero customers, spend £0 and
have £0 simulated NET change. Random/RFM still use K contacts and have negative
simulated NET totals when K>0. The proposed-minus-random difference therefore
shows the simulated value of abstaining; exact baseline totals require the final
test predictions. This algebraic implication is not an empirically rerun UCI result and not evidence
that promotion never works. Assumptions were not adjusted to manufacture profit.

Sensitivity retains the four endpoint ranges: response 0.10–0.35, discount
0.05–0.20, gross margin 0.30–0.50, At-risk multiplier 0.91–1.69. It now measures
budget-constrained NET margin. Reversed low/high outcomes have nonnegative plot
widths. Margin bootstrap conditions on predictions/assumptions, not causal uncertainty.

## Risks, limitations and ethics/bias

- No randomized treatment/control: every promotion impact is **simulated under
  assumptions**, not causal uplift, true ROI or observed incremental profit.
- Historical promotions may already affect sales; historical forecasts are not untreated outcomes.
- Sales exclude returns/cancellations; duplicates are audited and retained, not
  assumed erroneous. This may overstate economically realized revenue.
- Ninety-day value is not lifetime value. High spenders/outliers and time drift
  can materially affect predictions; clipping fixes domain, not calibration/accuracy.
- RFM quartiles use only the current snapshot's known features, including its full
  customer population. RFM is not a predictive input; rank ties depend on row order.
- Segment growth multipliers encode assumptions, not measured responsiveness.
- Optional country diagnostics pool countries with fewer than 20 customers.
  They are descriptive error checks, not proof of fairness or absence of bias.
  Earlier country-error results are superseded and must be recomputed.
- Customer bootstrap intervals do not capture time/regime/model-selection uncertainty.
- Only one final temporal origin; no causal experiment or deployment validation.

## Reproducibility

Configuration: `config/params.yaml`. Baseline: 02 → 03 → 05; optional 04 runs after
02 and before an advanced-source 05. Tests enforce boundaries, label availability,
test-label independence, policy provenance, clipping, financial arithmetic, equal
maximum capacity (not equal actual contact counts), deterministic scenarios and
tornado geometry. See README for commands.
