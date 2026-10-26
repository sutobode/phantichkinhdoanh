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
- Phase 3 challengers: one `GradientBoostingClassifier` and one
  `GradientBoostingRegressor`, sklearn defaults with seed=42. No grid search.
- Regression evaluation and simulation use `max(prediction, 0)`. Raw predictions
  are retained separately for diagnostics; clipping prevents negative-value sign reversal.
- `decision.value_prediction_source` remains frozen at baseline for the Phase 1/2
  business layer. Phase 3 model selection uses only validation results; final-test
  performance cannot change the selection.

## Temporal validation

1. Fit on both train snapshots to evaluate the later validation snapshot.
2. Refit the same predeclared model on train + validation snapshots, whose labels
   are completely observed by 2011-08-30, before final decision 2011-09-10.
3. Predict only the final temporal-test snapshot for final metrics and policy simulation.

The same customer may appear at earlier origins; that is acceptable historical
information. Final-test labels never enter fitting, scaling or selection.
There is **no random same-snapshot customer holdout**. This replaces the old
cross-sectional evaluation; old scores/intervals are not temporal-test evidence.

Baseline metrics use PR-AUC (average precision), ROC-AUC, MAE and RMSE with 1,000
seeded customer bootstrap resamples conditional on the fitted model. Phase 3 adds
Brier score, median/p90 absolute regression error, compact calibration diagnostics,
Tree SHAP attribution and secondary rolling-origin evaluation.

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
are stored separately there.

### Phase 3 advanced comparison and selection

Selection rules were fixed in `config/params.yaml` before final-test comparison:

- Classification: select GBM only if validation PR-AUC gains at least **0.01** and
  Brier score does not worsen by more than **0.005**.
- Regression: select GBM only if validation MAE improves by at least **5%** and
  p90 absolute error does not worsen by more than **5%**.

| Split / model pair | PR-AUC | ROC-AUC | Brier | MAE | RMSE | Median abs. error | P90 abs. error |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation baseline | 0.695268 | 0.806039 | 0.161353 | £294.53 | £1,030.92 | £112.47 | £616.99 |
| Validation GBM | 0.718923 | 0.814990 | 0.155854 | £277.66 | £1,120.81 | £99.74 | £551.24 |
| Final-test baseline | 0.750184 | 0.789977 | 0.199542 | £395.69 | £2,016.14 | £120.42 | £776.47 |
| Final-test GBM | 0.769056 | 0.798659 | 0.199442 | £400.40 | £2,167.29 | £103.34 | £792.10 |

The validation rule selects both GBM challengers: classifier PR-AUC gained 0.02365
with lower Brier score; regressor MAE improved 5.73% and p90 error fell 10.66%.
The final test supports the classifier choice, but GBM regression MAE/RMSE are worse
than Linear Regression despite lower median error. This is temporal instability,
not a reason to select retrospectively on final test. The interpretable baseline
remains the primary reference and frozen business input; GBM is retained as the
Phase 3 challenger, not promoted into policy calculations. The paired GBMs are also
substantially more expensive to fit, while inference remains small for both.
Exact timings are environment/run-dependent and remain in
`outputs/temporal_model_comparison.csv` rather than being treated as stable metrics.

### Calibration

Phase 3 reports Brier score and ECE using ten equal-width probability bins; ECE is
the customer-count-weighted absolute gap between mean predicted probability and
observed positive rate in nonempty bins. A single sigmoid calibrator was tested.
For validation, the base model fits the first train origin and the sigmoid fits the
second; for final evaluation, the base fits both train origins and the sigmoid fits
validation. Final-test labels are never used to fit or select calibration.

Validation Brier changed from **0.161353** to **0.161257**, an improvement of only
0.000096 versus the predeclared 0.002 requirement (PR-AUC 0.695511). Calibration
was therefore **not retained**, regardless of its final-test Brier of 0.198787 versus
0.199542 uncalibrated. Sources: `outputs/calibration_metrics.csv`,
`outputs/calibration_table.csv`, `outputs/figures/temporal_calibration_curve.png`.

### Predictive attribution

Tree SHAP was computed for the advanced classifier on a deterministic seed-42 sample
of 750 final-test customers. Top mean absolute attributions are frequency 0.6412,
recency 0.5734, monetary 0.2154, tenure 0.1896 and average order value 0.1215.
These explain predictive contribution in this fitted model; they do **not** identify
causes, treatment effects or promotion response. Sources:
`outputs/advanced_feature_importance.csv` and `outputs/figures/shap_summary.png`.

### Secondary temporal robustness

Five evaluation origins (2010-09-01 through 2011-09-10) use expanding prior
snapshots whose 90-day labels end strictly before each origin. This analysis is
secondary and does not replace the main schedule.

| Model family | Mean PR-AUC (range) | Mean MAE (range) |
|---|---:|---:|
| Baseline | 0.7179 (0.6598–0.7879) | £517.79 (£335.03–£690.04) |
| GBM | 0.7242 (0.6617–0.7897) | £514.06 (£302.21–£716.31) |

GBM classification is higher at all five origins, but gains vary and are often
small. Regression is mixed: GBM has lower MAE at two of five origins. Both families
vary materially over time, with no monotonic decay; continued temporal monitoring
is warranted. Source: `outputs/rolling_origin_metrics.csv` and
`outputs/figures/rolling_origin_performance.png`.

### Duplicate-policy robustness

The primary analysis retains 34,335 exact duplicate rows beyond first because the
source does not establish whether identical lines are errors or legitimate repeated
line items. A secondary `drop_exact_duplicates` analysis removes them, leaving
1,033,036 rows, while reusing the same temporal dates, baseline models and business
assumptions. It writes only namespaced artifacts and does not replace primary outputs.

| Duplicate policy | Final customers / repeat rate | PR-AUC | ROC-AUC | MAE | RMSE |
|---|---:|---:|---:|---:|---:|
| Retain (primary) | 5,281 / 43.36% | 0.750184 | 0.789977 | £395.69 | £2,016.14 |
| Drop exact duplicates | 5,281 / 43.36% | 0.750191 | 0.789971 | £396.54 | £2,066.55 |

Under duplicate removal, Proposed contacts/margin remain 0/£0 for Conservative,
Moderate and Aggressive. Non-targeted Monte Carlo mean losses become £16,612.70,
£35,638.18 and £86,340.30 respectively. The recommendation remains **do not launch
the configured campaigns**. The conclusion is qualitatively stable, while regression
RMSE changes quantitatively; this does not prove which duplicate policy is correct.
Sources: `outputs/duplicate_robustness_summary.csv` and
`outputs/duplicate_robustness_business_metrics.csv`.

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
  assumed erroneous. This may overstate economically realized revenue; the separate
  removal sensitivity changes regression error but not the campaign decision.
- Ninety-day value is not lifetime value. High spenders/outliers and time drift
  can materially affect predictions; clipping fixes domain, not calibration/accuracy.
- RFM quartiles use only the current snapshot's known features, including its full
  customer population. RFM is not a predictive input; rank ties depend on row order.
- Segment growth multipliers encode assumptions, not measured responsiveness.
- Optional country diagnostics pool countries with fewer than 20 customers.
  They are descriptive error checks, not proof of fairness or absence of bias.
  Earlier country-error results are superseded and must be recomputed.
- Customer bootstrap intervals do not capture time/regime/model-selection uncertainty.
- One main final temporal origin plus five secondary robustness origins; neither is
  a causal experiment or deployment validation.

## Reproducibility

Configuration: `config/params.yaml`. Baseline: 02 → 03 → 05; Phase 3 step 04 runs
after 02/03 and remains independent of step 05. Secondary duplicate sensitivity is
step 07 and never overwrites primary outputs. Tests enforce boundaries, label availability,
test-label independence, policy provenance, clipping, financial arithmetic, equal
maximum capacity (not equal actual contact counts), deterministic scenarios,
validation-only selection/calibration, rolling-origin provenance, deterministic
attribution sampling and tornado geometry. See README for commands.
Final QA generates and hash-validates `outputs/run_manifest.json`, recording actual
raw/config/requirements SHA-256 hashes, Git commit and dirty state, Python/platform,
timestamp, temporal design, prediction source, raw rows and final customer count.
Python remains 3.11; validate the activated environment with `python --version`
and `pip check`. The pinned dependencies have not been broadened or upgraded.
