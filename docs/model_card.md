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

The real UCI baseline rerun evaluates **5,281 final-test customers**, with a
**43.36%** repeat-purchase rate. Raw rows: **1,067,371**; exact duplicates beyond
first: **34,335**, retained. Synthetic tests validate mechanics separately.

| Final-test model | PR-AUC | ROC-AUC | MAE | RMSE |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.7501843981 | 0.7899770637 | — | — |
| Linear Regression, clipped | — | — | £395.6872412 | £2,016.1352842 |
| Naive | 0.4336299943 | — | £636.3386447 | £3,288.8721982 |

Source: `models/temporal_registry.json`; validation metrics and bootstrap intervals
are stored separately there. No new advanced models are fitted in Phase 1 + Phase 2.

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

Policies on the same final-test population: Monte Carlo non-targeted, descending
RFM sum (then monetary, then customer ID), and descending positive NET margin.
All share maximum K=min(floor(contact budget/contact cost), evaluation population).
Random and RFM contact K customers regardless of proposed eligibility. Proposed
selects only positive NET contributions, ranked descending, up to K and may select
zero. Actual contact costs are deducted for each policy. NET margin/contact is
reported as 0 by convention for a zero-contact policy. No over-budget blanket
comparison is used.
**No Promotion = £0** is the additional zero-contact decision reference, not a model.
Non-targeted uses 1,000 independent uniform K-subsets without replacement within
each campaign (seed 42, canonical customer order). The same seeded draws are used
across scenarios. Its primary metric is the Monte Carlo mean. Reported 2.5th/97.5th
percentiles describe random-policy variation, not causal confidence or forecast uncertainty.

With the unchanged three scenarios, even the highest multiplier produces negative
before-contact coefficients: -0.0045, -0.022, -0.109 respectively. Thus for **any
nonnegative v**, all three proposed campaigns select zero customers, spend £0 and
have £0 simulated NET change. Random/RFM still use K contacts and have negative
simulated NET totals when K>0. The proposed-minus-random difference therefore
shows the simulated value of abstaining. The real-data policy rerun confirms:

| Scenario | Non-targeted MC mean NET | Random-policy 2.5th–97.5th percentiles | RFM NET | Proposed / No Promotion |
|---|---:|---|---:|---:|
| Conservative | -£17,417.59 | [-£21,707.94, -£14,023.76] | -£55,333.45 | £0 / £0 |
| Moderate | -£37,407.57 | [-£46,583.42, -£30,081.21] | -£118,314.35 | £0 / £0 |
| Aggressive | -£90,677.15 | [-£112,076.51, -£73,360.59] | -£277,865.97 | £0 / £0 |

Random/RFM each contact 1,500 customers and spend £750. Proposed and No Promotion
contact zero and spend zero. Source: `outputs/scenario_table.csv`. These replace
the earlier single-draw random benchmark, not the unchanged predictive results.
They are conditional simulations on real-data forecasts, not observed promotion
effects or evidence that promotion never works. Main assumptions remain unchanged.

Sensitivity retains the four endpoint ranges: response 0.10–0.35, discount
0.05–0.20, gross margin 0.30–0.50, At-risk multiplier 0.91–1.69. It now measures
budget-constrained NET margin. Reversed low/high outcomes have nonnegative plot
widths. Margin bootstrap conditions on predictions/assumptions, not causal uncertainty.
Moderate one-at-a-time results: discount 5% → £7,175.92; base growth 35% → £3,990.80;
gross margin 50% → £143.77; At-risk multiplier 1.69 → £15.05.

### Break-even and response × discount frontier

For v>0, required effective growth is `(g+c/v)/(g-d)-1` when g>d; required base
growth divides this by the positive segment multiplier. Maximum discount at fixed
effective growth is `g-(g+c/v)/(1+r)`. Equality produces zero NET margin, whereas
targeting requires strictly positive margin. Invalid cases return NaN; negative
discount bounds are retained to show that even no discount cannot cover contact cost.

At 10% discount, among positive-value customers the median required base growth is:
At-risk **26.75%**, Loyal **34.47%**, Champions **67.38%**, Lost **90.49%**.
For At-risk customers at the configured 20% base growth, median supportable discount
is **7.91%**, below the proposed 10%. Quantiles exclude undefined values: 3,698 of
5,281 customers have positive predicted sales (779 At-risk). Eligibility shares
still use all customers. Sources: `outputs/break_even_by_segment.csv`, `break_even_summary.csv`.

The 20-cell exploratory surface holds margin=40%, budget=£750 and multipliers fixed.
First profitable tested grid points by discount: 5%/20% base growth (1,500 contacts,
£7,175.92); 10%/30% (664, £1,424.27); 15%/50% (616, £962.39). At 20% discount no
campaign qualifies through 50% growth. This is a discrete assumption frontier,
not a measured causal relationship. Outputs: `outputs/response_discount_surface.csv`
and `outputs/figures/response_discount_surface.png`.

The dashboard sandbox recomputes the same policy arithmetic on filtered final-test
scores without retraining, with explicitly labeled exploratory scope and five
decision KPIs. Saved global comparisons/cohorts are not changed by its controls.

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
Final QA generates and hash-validates `outputs/run_manifest.json`, recording actual
raw/config/requirements SHA-256 hashes, Git commit and dirty state, Python/platform,
timestamp, temporal design, prediction source, raw rows and final customer count.
Python remains 3.11; validate the activated environment with `python --version`
and `pip check`. The pinned dependencies have not been broadened or upgraded.
