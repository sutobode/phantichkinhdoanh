# Report Evidence Pack — Customer Value & Promotion Targeting for an Online Retailer

**Purpose:** source-grounded evidence for a later academic Business Analytics report; this is not the final report. Extracted from the current checkout on 2026-11-02. No analytical pipeline, QA suite, model fitting, or chart generation was run during extraction.

**Reading convention:** **Observed** denotes transactions, historical features, and realized holdout labels; **Predictive** denotes forecasts and predictive evaluation; **Assumed** denotes configured economics; **Simulated** denotes conditional policy arithmetic. GBP amounts are nominal dataset-currency amounts. Tables round currency to two decimals and most metrics to six decimals; cited CSV/JSON fields retain full precision. A citation applying to a table is stated immediately before or after it. `path::function/section` identifies implementation or artifact keys; CSV citations identify row keys and columns. PDF citations identify the visible section and printed page.

## 0. Evidence index and provenance

### 0.1 Checkout versus analytical-run provenance

| Item | Verified evidence |
|---|---|
| Current Git HEAD | `bb27c8516ac4e3dbc1680db92dcc1032dc2b3666` (`git rev-parse HEAD`, extraction session) |
| Working tree **before** writing this pack | `git status --short --untracked-files=all` returned only `?? "Mathematical Modelling Project - Slides.pdf"`; no tracked modifications. This pre-existing untracked PDF was not opened or modified by the extraction. |
| Final workspace observation | HEAD remained unchanged; `git status --short --untracked-files=all` returned only `?? docs/REPORT_EVIDENCE_PACK.md`. The earlier untracked slides PDF was no longer present in the root directory. No extraction command removed or changed it; this independently occurring workspace change must not be attributed to the evidence-pack creation. |
| Latest recorded baseline run | `baseline_temporal_20261027T013845Z`, short commit `f2164e8`; `models/temporal_registry.json:1063-1136` |
| Latest recorded advanced run | `advanced_temporal_20261026T182001Z`, short commit `334a105`; `models/temporal_registry.json:495-604` |
| QA manifest timestamp | `2026-10-27T01:40:28.830108+00:00`; `outputs/run_manifest.json:2` |
| Manifest analytical revision/state | `f2164e82d6a639ccd72ff16ba21473edb227a8ec`, `git_worktree_dirty=true`; `outputs/run_manifest.json:3-4` |
| Recorded analytical environment | Python `3.11.3`; `Windows-10-10.0.26300-SP0`; `outputs/run_manifest.json:5-6` |
| Final business prediction source | `baseline`, decision `2011-09-10`, horizon 90 days; `config/params.yaml:26-37,58-59`; `outputs/run_manifest.json:9-13` |
| Final policy-input lineage | `data/processed/customer_test_predictions.csv` → `data/processed/customer_test_scores.csv`; `src/05_decision_scenario.py::load_scores,prepare_scores,main` |
| Official instructor requirements | Present: `phantichkinhdoanh.pdf`, two PDF pages, printed pages 2–3; visually read. Required package, temporal validation, baseline, scenario economics, and non-causal guardrails are explicit. The executive brief has a maximum of two pages; no full-report page limit or citation style is specified there. |

The manifest records an **analytical run**, not the later release checkout. Recent history includes `c7f7b2c` (“Add executive brief generation script and update documentation with calibration evidence”) and current `bb27c85` (“update project”). Do not relabel the manifest as a run at current HEAD or infer a clean current tree from an older audit. The registry is append-only and contains multiple runs; use the latest baseline entry for baseline intervals and the matching latest advanced entry/current comparison CSV for challengers. The baseline point estimates are consistent across those sources. Sources: `src/03_baseline_models.py::register_run`; `README.md:178-181`; extraction-session Git observations.

### 0.2 Checksums and dependencies

| Input/artifact | SHA-256 | Status |
|---|---|---|
| `data/raw/online_retail_II.xlsx` | `bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980` | Recorded manifest hash; raw workbook was not rescanned/rehashed in this extraction. |
| `config/params.yaml` | `e698fcbefef5bd36eced6b0cdf9473e11f8aaace62298f2f8ff9be794b52cc97` | Recorded manifest hash; read-only byte hash during extraction matches. |
| `requirements.txt` | `d0680e1efe8b818f3b0771eda4199a4114b8cb0d76fe5f24d663d571ce3e12ca` | Recorded manifest hash; read-only byte hash during extraction matches. |
| `data/processed/customer_test_scores.csv` | `d4d1aef08a2d30f082c49acab6a933c3f70a989367787797e6903408c79b1484` | **New derived checksum** during extraction; not a field in the historical manifest. |

Source: `outputs/run_manifest.json:7-14`; checks calculated as `hashlib.sha256(Path(path).read_bytes()).hexdigest()` for the three small text/score artifacts above. No artifact was overwritten.

Pinned versions (`requirements.txt:1-16`): pandas 3.0.5; NumPy 2.4.6; scikit-learn 1.9.1; matplotlib 3.11.1; seaborn 0.13.2; openpyxl 3.1.5; PyYAML 6.0.2; SciPy 1.17.1; pytest 8.3.4; SHAP 0.51.0; pandera 0.35.0; FastAPI 0.162.0; uvicorn 0.54.0; joblib 1.6.0; Streamlit 1.66.1; Plotly 7.1.0. These are **pins**, not a complete historical `pip freeze`. A read-only runtime inspection confirmed local scikit-learn 1.9.1 when checking default estimator parameters (§5). No fresh dependency-compatibility test was run.

### 0.3 Authoritative source map

| Evidence | Primary path(s) | Use |
|---|---|---|
| Dataset definition and dates | `config/params.yaml`; `src/02_cleaning_features.py`; `src/data_validation.py` | Cleaning, windows, features, labels, schema |
| Dataset counts and schedule | `outputs/data_quality.csv`; `outputs/temporal_schedule.csv`; dated real-data blocks in `outputs/logs/02_cleaning_features.log` | Raw/duplicate counts, window removal counts, boundaries |
| Final customer evaluation | `data/processed/customer_test_scores.csv` | Schema and aggregate-only inspection; current customer population, labels, predictions, decisions |
| Baseline evaluation | `models/temporal_registry.json::baseline_temporal_20261027T013845Z.metrics` | Validation/final metrics and bootstrap intervals |
| Advanced evaluation/selection | `outputs/temporal_model_comparison.csv`; `outputs/model_selection.csv`; `src/04_advanced_models.py` | Challenger metrics and validation-only gates |
| Calibration and interpretation | `outputs/calibration_metrics.csv`; `outputs/calibration_table.csv`; `outputs/advanced_feature_importance.csv` | Current calibration and Tree SHAP |
| Temporal robustness | `outputs/rolling_origin_metrics.csv` | Secondary five-origin evaluation |
| Business decision | `src/05_decision_scenario.py`; `outputs/scenario_table.csv` | Fixed-cost economics, Monte Carlo and policy outcomes |
| Sensitivity | `outputs/break_even_by_segment.csv`; `outputs/break_even_summary.csv`; `outputs/tornado_sensitivity.csv`; `outputs/response_discount_surface.csv` | Analytical break-even and assumption exploration |
| Duplicate robustness | `src/07_duplicate_robustness.py`; `outputs/duplicate_robustness_summary.csv`; `outputs/duplicate_robustness_business_metrics.csv` | Namespaced secondary sensitivity |
| Dashboard | `dashboard/app.py` | Controls, five KPIs, charts, actionable download |
| QA/reproducibility | `src/06_final_qa.py`; `src/utils.py`; `outputs/final_qa_report.md`; `outputs/logs/06_final_qa.log`; `outputs/run_manifest.json` | Historical verification and manifest generation |
| Secondary narrative | `README.md`; `docs/problem_statement.md`; `docs/data_dictionary.md`; `docs/model_card.md`; `docs/executive_brief.md`; `docs/traceability_map.md`; `slides/outline.md` | Interpret only after checking code/current outputs |

**Historical exclusions:** `data/processed/customer_scores.csv`, `models/registry.json`, `outputs/rolling_origin_validation.csv`, `outputs/calibration_logreg.csv`, `outputs/calibration_gbm.csv`, `outputs/shap_feature_importance.csv`, `outputs/error_by_segment.csv`, `outputs/ethics_bias_check_by_country.csv`, and `outputs/figures/calibration_curve.png` are not current temporal-test evidence. `PLAN.md` is an archived work log. Current calibration is **not** excluded: `temporal_calibration_logreg.csv`, `calibration_metrics.csv`, and `calibration_table.csv` are current. Source: `outputs/RESULTS_STATUS.md:37-48`; `README.md:3-10`.

## 1. Research problem and decision framing

**Exact instructor business question:** “Which customers should the retailer prioritize for retention or promotion to increase expected future margin without over-discounting?” Source: `phantichkinhdoanh.pdf::Customer Value & Promotion Targeting / Business Question` (printed p.3). Repository formulation agrees: `docs/problem_statement.md:3-7`.

| Element | Report-ready evidence |
|---|---|
| Decision-maker | Marketing/CRM decision-maker (`docs/executive_brief.md:3`). |
| Unit of analysis | Customer × decision date, requiring at least one valid purchase in the observation history; not a transaction, invoice, or randomly held-out customer (`src/02_cleaning_features.py::build_snapshot,compute_rfm_features`). |
| Predictive question | Will an existing customer buy in the next 90 days, and how much **gross positive future sales** will that customer generate? (`compute_labels`; `config/params.yaml:32-37`). |
| Prescriptive question | Under explicit discount, proportional sales-growth, margin and contact-cost assumptions, which customers have positive simulated incremental NET margin, subject to a contact budget? No contact is allowed (`src/05_decision_scenario.py::incremental_margin,budget_constrained_selection`). |
| Observation and outcomes | Expanding history from 2009-12-01; features strictly before each decision; labels in `[T,T+90 days)` (§3). |
| Capacity and cost | £750 **contact** budget; £0.50/contact; `K=min(floor(B/c),N)=1,500` for 5,281 final-test customers. Discount cost is embedded in margin, not charged to this contact budget (`config/params.yaml:76-96`; `outputs/scenario_table.csv::campaign_capacity`). |
| Decisions | Positive value/NET → `nurture`; positive-value Champions/Loyal with nonpositive NET → `retain`; otherwise `no-offer`. Recommendation and actual budget selection are separate (`src/05_decision_scenario.py::assign_recommendation,main`). |
| Guardrails | Final temporal-test predictions only; preselected baseline source; nonfinite values rejected; negative forecasts clipped; same maximum capacity across policies; no final-label fitting; no assumed causal interpretation (§3, §7). |

`future_gross_sales` is the sum of positive-quantity, positive-price, non-cancelled sales over the future horizon. It is **not net return-adjusted revenue, profit, lifetime CLV, or an identified no-promotion outcome**. Classification probability is not promotion-response probability. There is no randomized promotion treatment/control and no estimated causal uplift, realized ROI, measured incremental profit, or verified savings. Sources: `src/02_cleaning_features.py::drop_invalid_rows,compute_labels`; `src/05_decision_scenario.py:7-20`; `phantichkinhdoanh.pdf::Scope Guardrails`.

## 2. Dataset, acquisition, cleaning, and descriptive analytics

### 2.1 Acquisition and raw measurement

| Item | Evidence and qualification |
|---|---|
| Dataset | UCI Machine Learning Repository, **Online Retail II**; two years of transactions from a UK-based non-store retailer (`phantichkinhdoanh.pdf::Primary Dataset`, printed p.3). |
| Dataset page | `https://archive.ics.uci.edu/dataset/502/online+retail+ii` (`docs/data_dictionary.md:5-6`). |
| Direct download URL | `https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip` (`config/params.yaml:9`). |
| Recorded download date | **2026-10-26**, recorded consistently in `config/params.yaml:10` and `docs/data_dictionary.md:8`. This is not missing; an independent download receipt is not recorded. |
| Workbook/sheets | `data/raw/online_retail_II.xlsx`; “Year 2009-2010” and “Year 2010-2011” (`docs/data_dictionary.md:9`). Code loads all sheets and concatenates them (`src/02_cleaning_features.py::load_raw`). Sheet-specific row counts are **NOT AVAILABLE** in the inspected current outputs. |
| Size | 1,067,371 rows, 8 columns (`outputs/data_quality.csv::raw_rows`; `outputs/logs/data_validation.log:11`). |
| Granularity | Transaction **line item**: invoice identifier, stock code, quantity and unit price. Multiple product lines can share an invoice; neither invoice ID nor customer ID is a raw-row unique key. This follows the raw fields and distinct-invoice aggregation (`src/data_validation.py::RAW_SCHEMA`; `src/02_cleaning_features.py::compute_rfm_features`). |
| Date coverage | Recorded calendar span **2009-12-01 through 2011-12-09** (`PLAN.md:101`, historical acquisition record; `docs/problem_statement.md:24,38`; `outputs/monthly_transaction_profile.csv::month`). Current code checks start/end coverage before building snapshots. Exact raw minimum/maximum **timestamps** are **NOT AVAILABLE** in current inspected logs/summary artifacts; no workbook scan was performed. |
| Last day | Final labels exclude 2011-12-09 because the last recorded calendar day may be partial (`src/02_cleaning_features.py:216-222`). |
| External joins | None in the snapshot construction; feature/label join is one-to-one by Customer ID and snapshot uniqueness is checked by Customer ID × decision date (`build_snapshot,temporal_partitions`). |

Raw dictionary (semantic types, not inferred Excel cell storage):

| Field | Meaning/type | Validation / caveat |
|---|---|---|
| `Invoice` | String invoice identifier | Coerced string, non-null; prefix `C` marks cancellation. |
| `StockCode` | String product identifier | Coerced string, non-null; distinct count is a feature. |
| `Description` | Product description, string | Nullable; not a predictor. |
| `Quantity` | Integer units on a line | Non-null, coerced integer; nonpositive values subsequently excluded from valid purchases. |
| `InvoiceDate` | Transaction timestamp | Non-null timestamp; time-window field. |
| `Price` | Float unit price in GBP | Non-null, coerced float; nonpositive values subsequently excluded. |
| `Customer ID` | Nullable numeric identifier | Coerced float at validation, allowed non-null range 10000–99999; missing IDs dropped, remaining IDs converted to integer. |
| `Country` | Country string | Non-null, coerced string; first observed valid country retained per customer, not a predictor. |

Sources: `docs/data_dictionary.md:14-25`; `src/data_validation.py:18-31`; `src/02_cleaning_features.py::filter_valid_customers,compute_rfm_features`. Schema allows extra columns (`strict=False`); current validation log reports exactly eight.

### 2.2 Cleaning counts and non-overlap rules

**Dataset-level observed audit:** 243,007 rows have missing Customer ID and are excluded globally; 34,335 exact duplicate rows **beyond the first occurrence** are retained in the primary analysis. Duplicate audit occurs on original concatenated rows before schema coercion. Duplicate rows can overlap missing-ID/invalid-transaction categories, so they are **not another sequential removal count**. Sources: `outputs/logs/02_cleaning_features.log:7022-7023`; `outputs/data_quality.csv::exact_duplicate_rows,duplicate_policy`; `src/02_cleaning_features.py:243-252`.

**New derived calculations:** known-ID rows = `1,067,371−243,007 = 824,364`; missing-ID percentage = `243,007/1,067,371×100 = 22.766873%`; duplicate percentage = `34,335/1,067,371×100 = 3.216782%`. Inputs are the audit counts above, not a new raw-data cleaning run.

Per-window counts from the **latest primary real-data block**, `outputs/logs/02_cleaning_features.log:7024-7055` (2026-10-27 08:38 local log timestamps):

| Decision T | Window | Known-ID rows before window cleaning | C-prefix cancellation removals | Additional Quantity ≤0 removals | Additional Price ≤0 removals | Valid rows |
|---|---|---:|---:|---:|---:|---:|
| 2010-12-01 | Observation | 402,731 | 9,541 | 0 | 30 | 393,160 |
| 2010-12-01 | Outcome | 83,928 | 2,103 | 0 | 8 | 81,817 |
| 2011-03-01 | Observation | 486,659 | 11,644 | 0 | 38 | 474,977 |
| 2011-03-01 | Outcome | 79,005 | 1,771 | 0 | 6 | 77,228 |
| 2011-06-01 | Observation | 566,587 | 13,429 | 0 | 44 | 553,114 |
| 2011-06-01 | Outcome | 81,724 | 1,843 | 0 | 8 | 79,873 |
| 2011-09-10 | Observation | 659,661 | 15,587 | 0 | 52 | 644,022 |
| 2011-09-10 | Outcome | 164,078 | 3,150 | 0 | 19 | 160,909 |

Within a row, removals are sequential and non-overlapping: cancellation first, then remaining nonpositive quantity, then remaining nonpositive price. **Do not sum expanding-snapshot rows:** historical transactions recur in later observation windows, and previous outcome windows later become history. The separate `final_test/cohort` log repeats the final observation counts and is not another removal. Zero *additional* quantity removals does not mean the raw workbook had no negative quantities. Whole-workbook, mutually exclusive cancellation/quantity/price totals and a full raw-field missingness table are **NOT AVAILABLE** in the inspected current artifacts. Outcome transaction counts include customers new to that window; labels are subsequently restricted to observation customers (`compute_labels`).

Returns/cancellations are not matched to original invoices or netted from sales. `return_rate` measures distinct cancellation invoices divided by all distinct invoices in the observation window **before** purchase cleaning; it is not a monetary return ratio. Exact identical lines are not proven data errors; the separate drop-duplicate sensitivity tests dependence on that choice (§10). Sources: `src/02_cleaning_features.py::compute_return_rate,drop_invalid_rows,compute_labels`; `docs/data_dictionary.md:41-56`.

### 2.3 Current descriptive evidence

The following score-table summaries are **new derived calculations**, using only aggregate inspection of `data/processed/customer_test_scores.csv`. They describe the 5,281-customer **final snapshot**, not all raw transactions or all ever-observed customers. Method: pandas `read_csv`; `size/nunique/value_counts/groupby`; `describe(percentiles=[.25,.5,.75,.9,.95,.99])`; no customer-level records were printed. The score checksum is in §0.2.

| Quantity | Aggregate | Evidence field/formula |
|---|---:|---|
| Unique evaluation customers | 5,281; 5,281 rows | `Customer ID.nunique()`, `len` |
| Repeat purchasers | 2,290 / 5,281 = 43.362999% | `repeat_purchase.sum()/len`; also registry `naive_pr_auc` |
| Non-purchasers | 2,991 | `5,281−2,290`; target is zero |
| UK-labelled customers | 4,813 | `country.value_counts()['United Kingdom']`; country is first valid observation country |
| Positive clipped forecasts | 3,698 | `(predicted_future_value>0).sum()`; corroborated by `break_even_summary.csv::valid_break_even_customers` |
| Negative raw forecasts clipped to zero | 1,583 | `(predicted_future_value_raw<0).sum()` |
| Saved moderate recommendations | `retain` 2,454; `no-offer` 2,827; `nurture` 0 | `final_recommendation.value_counts()` |
| Missing score fields | None except all 5,281 `target_rank` values blank | `isna().sum()`; blank rank is deliberate ineligibility, not failed feature imputation |

| Final-snapshot quantity | Observed expanding-history `monetary` (£) | Observed next-90-day `future_gross_sales` (£) | Predicted next-90-day `predicted_future_value` (£) |
|---|---:|---:|---:|
| Mean | 2,705.44 | 543.11 | 379.00 |
| 25th percentile | 323.86 | 0.00 | 0.00 |
| Median | 804.02 | 0.00 | 97.52 |
| 75th percentile | 2,118.75 | 435.47 | 271.99 |
| 90th percentile | 5,164.88 | 1,135.76 | 721.00 |
| 95th percentile | 8,729.38 | 1,881.86 | 1,283.31 |
| 99th percentile | 28,050.95 | 5,409.34 | 4,405.25 |
| Maximum | 484,615.10 | 123,737.11 | 75,348.29 |

**New derived high-value skew:** top `ceil(0.01×5,281)=53` customers, independently ranked by each observed value column, account for **30.831129%** of historical `monetary` and **37.575579%** of final `future_gross_sales`: `column.nlargest(53).sum()/column.sum()`. This is descriptive concentration, not treatment responsiveness. Historical frequency median is 3 invoices (mean 5.745692; maximum 309); recency median is 163 days; cancellation-invoice-share median is 0 (mean 0.121119). Source fields: score-table `frequency`, `recency_days`, `return_rate`; same aggregate method.

**Existing cohort evidence:** `outputs/cohort_retention.csv`, generated from the clean **final observation** history, defines first-observed-purchase-month cohorts and monthly activity, not uninterrupted survival or true lifetime acquisition. For December 2009, cohort size 955: month 1 active 337 (35.287958%), month 6 active 360 (37.696335%), month 12 active 359 (37.591623%), month 20 active 244 (25.549738%). January 2010 cohort size 383: month 1 active 79 (20.626632%). Row keys are `(cohort_month,months_since_acquisition)`; columns `n_active_customers,cohort_size,retention_rate`; implementation `src/02_cleaning_features.py::compute_cohort_view,main`. **New derived** count of distinct cohort months is 22 (2009-12–2011-09), and sum of month-0 cohort sizes is 5,281. September 2011 activity is only through September 9; its cells are partial-month activity and should not be compared as complete-month retention. Missing heatmap cells are not automatically zero retention.

`outputs/monthly_transaction_profile.csv` supplies raw monthly distinct invoices/customers: November 2010 = 3,669/1,683; November 2011 = 3,462/1,711; December 2011 = 1,015/686 (partial month). Its function performs no cancellation/price cleaning (`src/01_eda.py::monthly_transaction_profile`), and it is an optional source-EDA artifact outside the final-QA rerun. Use only as source-level activity context; monthly unique customers cannot be summed into unique dataset customers.

**Existing visual support:** dashboard cohort heatmap and segment bars. Static gross-sales histogram, Lorenz/Pareto plot, raw missingness chart and cleaned monthly-sales chart are **NOT AVAILABLE**; no chart was created for this pack (§12).

## 3. Temporal dataset construction and leakage prevention

### 3.1 Exact main timeline

All timestamps are midnight, boundaries half-open, history start `2009-12-01 00:00:00`. No timezone transformation is implemented in the window logic.

| Role | Decision T / observation end exclusive | Observation start inclusive | Label start inclusive | Label end exclusive | Snapshot customers |
|---|---|---|---|---|---:|
| Train 1 | 2010-12-01 | 2009-12-01 | 2010-12-01 | 2011-03-01 | 4,266 |
| Train 2 | 2011-03-01 | 2009-12-01 | 2011-03-01 | 2011-05-30 | 4,537 |
| Validation | 2011-06-01 | 2009-12-01 | 2011-06-01 | 2011-08-30 | 4,933 |
| Final test | 2011-09-10 | 2009-12-01 | 2011-09-10 | 2011-12-09 | 5,281 |

Dates: `outputs/temporal_schedule.csv::decision_date,split,outcome_end_exclusive`; `config/params.yaml:13-32`. First/second train sample counts are the corresponding `n_evaluation` snapshot counts in `outputs/rolling_origin_metrics.csv` (same `build_snapshot` and history), corroborated by their sum 8,803 in `models/temporal_registry.json::latest baseline.metrics.validation.n_train`. Validation and final counts are explicit in that registry and `outputs/temporal_model_comparison.csv`. These are customer-snapshot counts, not distinct people across all training origins.

| Evaluation | Fitting snapshots | Fitting rows | Latest fitted label end exclusive | Evaluation rows |
|---|---|---:|---|---:|
| Validation | Train 1 + Train 2 | 8,803 | 2011-05-30 < 2011-06-01 | 4,933 |
| Final test | Train 1 + Train 2 + Validation | 13,736 | 2011-08-30 < 2011-09-10 | 5,281 |

Source: `src/03_baseline_models.py::evaluate_temporal_models`; `models/temporal_registry.json:1081-1136`. Refit uses the same declared specifications; the final labels are for evaluation only. Earlier customers can reappear: expanding history has no inactivity-based cohort deletion and prior customers remain eligible once they have a valid purchase. **New derived set implication**, not a scanned ID-overlap table: with this nested eligibility, all 4,266 Train-1 customers reappear in Train 2; all 4,537 Train-2 customers reappear in validation; all 4,933 validation customers reappear in final test. Thus 8,803 fitting rows are not 8,803 different customers. Source logic: `build_snapshot`, `compute_rfm_features`; synthetic confirmation of permitted repeated IDs: `tests/test_temporal_evaluation.py:95-113`.

### 3.2 Leakage controls and uncertainty units

| Control | Implemented behavior | Verification reference |
|---|---|---|
| Features available at T | Split first; observation `start≤InvoiceDate<T`; cancellation share computed from observation only | `src/02_cleaning_features.py::split_windows,build_snapshot`; `tests/test_features.py:73-139` |
| Outcome definition | Independent cleaning within `[T,T+90)`; missing customer outcome → zero | `compute_labels`; `tests/test_features.py:142-157` |
| Complete horizon | Raw normalized last day must reach exclusive label end; potentially partial last day excluded | `build_temporal_snapshots`; `tests/test_temporal_evaluation.py:69-72` |
| Fitting chronology | Main training outcomes must end **strictly before** validation/final decision | `snapshot_schedule`, `evaluate_temporal_models`; `tests/test_temporal_evaluation.py:51-66` |
| Unique snapshot keys | Reject duplicated Customer ID × decision date; reject incorrect schedule | `temporal_partitions` |
| Train-only scaler | StandardScaler inside logistic pipeline, fitted separately on each fitting partition | `src/03_baseline_models.py::baseline_models`; `tests/test_temporal_evaluation.py:110-112` |
| Final-label independence | Mutating final labels does not alter predictions | `tests/test_temporal_evaluation.py:95-106`; `tests/test_advanced_models.py:46-62` |
| Validation-only selection | Final rows ignored by model selection; calibration adoption also uses validation | `src/04_advanced_models.py::select_models,evaluate_calibration`; `tests/test_advanced_models.py:65-98` |
| Policy provenance | Reject empty/mixed/duplicate/legacy/demo rows; require configured final dates, design and earlier fitted-label end; verify source and finite values | `src/utils.py::validate_final_test_predictions`; `src/05_decision_scenario.py::prepare_scores`; `tests/test_decision_scenario.py:107-131` |

Baseline metric uncertainty uses 1,000 seed-42 **customer resamples within a single evaluation snapshot**, size N with replacement, keeping the already fitted predictions fixed. Reported arrays are `[point,2.5th percentile,97.5th percentile]`; invalid metric resamples are skipped. It is not refitting/bootstrap model-selection uncertainty, not a block/time bootstrap, and not a cross-origin customer-cluster bootstrap. Repeated identities in fitting origins are allowed and are not grouped away or downweighted. Source: `src/03_baseline_models.py::bootstrap_ci,evaluate_temporal_models`; `config/params.yaml:4,33`.

**Important boundary nuance:** Train 1 label end equals Train 2 decision (2011-03-01); main validation/final fitting remains valid because both are earlier than the evaluation date. Sigmoid calibration allows base-fit label end **equal to** calibration decision (`>` rejection), but calibration-label end must be strictly before evaluation (`>=` rejection). Secondary rolling training is stricter (`label end < origin`), and excludes Train 1 at the 2011-03-01 rolling evaluation. Do not claim every fit/calibration transition is strictly separated by a positive time gap. Sources: `src/04_advanced_models.py:162-178,314-322`.

## 4. Features, RFM/cohorts, targets

### 4.1 All actual predictive inputs

The following **seven**, in this order, are used by both baseline and advanced classifier/regressor. Country, Customer ID, RFM score/segment, cohort, decisions and labels are not model inputs. Sources: `src/03_baseline_models.py:81-84`; `src/04_advanced_models.py:31-33`.

| Field | Exact calculation and units | Availability / null handling | Source |
|---|---|---|---|
| `recency_days` | `(T−max(InvoiceDate)).days` over valid observation purchases; integer elapsed days, not calendar-date subtraction | Before T; at least one valid purchase guarantees an extrema value; no imputer | `src/02_cleaning_features.py:123-125` |
| `frequency` | `nunique(Invoice)` among valid observation lines | Before T; ≥1; no imputer | Same, line 126 |
| `monetary` | `sum(Revenue)`, `Revenue=Quantity×Price`, GBP gross positive historical sales | Before T; purchases have positive quantity/price; no imputer | `drop_invalid_rows:116`; `compute_rfm_features:127` |
| `avg_order_value` | `sum(Revenue)/nunique(Invoice)`, GBP/invoice | Before T; denominator ≥1; no imputer | Same, line 128 |
| `n_distinct_products` | `nunique(StockCode)` among valid observation lines | Before T; schema requires StockCode; no imputer | Same, line 129 |
| `tenure_days` | `(T−min(InvoiceDate)).days`; days since **first observed** valid purchase | Before T; not proven lifetime tenure; no imputer | Same, line 130 |
| `return_rate` | `nunique(C-prefixed Invoice)/nunique(all Invoice)` per customer in uncleaned observation window after missing-ID removal | Before T; missing cancellation numerator reindexed to 0; missing joined rate filled 0.0; ratio of invoice counts, not line/quantity/value returns | `compute_return_rate:85-91`; `compute_rfm_features:133-134` |

Schema validation precedes primary feature building and rejects missing required non-null raw fields. No general feature imputation, log transform, winsorization, feature selection, or country encoding appears in the fitted pipelines. `country` is the first value in the grouped valid observation rows (`compute_rfm_features:131`), not a resolved lifetime location.

### 4.2 Segmentation and descriptive cohorts

RFM is computed independently in each evaluation snapshot using `pd.qcut(feature.rank(method='first'),4,labels=...)`. Recency labels are `[4,3,2,1]`; frequency and monetary labels `[1,2,3,4]`. Sum yields 3–12. There are no fixed GBP/day cut points: cutoffs depend on that snapshot's ranks; ties are broken by row order. RFM uses contemporaneously available features from the full evaluation population, not future labels, and is not a predictive input. Source: `src/03_baseline_models.py:58-78,154-156`.

| Segment | RFM sum rule | Final customers | Positive predicted sales | Assumed growth multiplier | Observed final repeat rate |
|---|---|---:|---:|---:|---:|
| Champions | ≥10 | 1,507 | 1,503 | 0.5 | 76.509622% |
| Loyal | 8–9 | 1,065 | 951 | 1.0 | 46.572770% |
| At-risk | 6–7 | 1,136 | 779 | 1.3 | 33.010563% |
| Lost | 3–5 | 1,573 | 465 | 0.4 | 16.910362% |

Counts/positive-value counts: `outputs/break_even_by_segment.csv::scenario=moderate,segment; customers,valid_break_even_customers`; multipliers: `config/params.yaml:81-86`. Repeat percentages are **new derived calculations**: score-table `groupby('rfm_segment').repeat_purchase.mean()×100`. Segment labels are descriptive rules, **not observed churn states or learned treatment-response types**.

Distinct meanings: RFM = known-history segmentation; cohort retention = observed monthly purchasing among first-observed cohorts; `predicted_repeat_purchase_proba` = classifier probability of any positive future sales; `predicted_future_value` = unconditional clipped regression forecast of 90-day gross sales, including customers who buy nothing. The policy uses the regression forecast directly; it does **not** multiply it by repeat probability. Source: `src/05_decision_scenario.py::prepare_scores,incremental_margin`.

### 4.3 Score artifact schema

Aggregate-only CSV inspection confirmed **31 columns**, 5,281 unique customers and the provenance values in §0/§3. Types below are read-time CSV-inferred types, not a formal storage schema; date strings become timestamps when validated.

| Type/group | Exact fields |
|---|---|
| Integer identifiers/history/classification | `Customer ID`, `recency_days`, `frequency`, `n_distinct_products`, `tenure_days`, `repeat_purchase`, `rfm_score` |
| Float observed history/target | `monetary`, `avg_order_value`, `return_rate`, `future_gross_sales` |
| String history/segmentation | `country`, `rfm_segment` |
| String temporal provenance | `decision_date`, `outcome_end_exclusive`, `split`, `prediction_source`, `evaluation_design`, `training_outcome_end_exclusive` |
| Float forecasts | `predicted_repeat_purchase_proba`, `predicted_future_value_raw`, `predicted_future_value`, `baseline_future_revenue` |
| Float assumed/simulated quantities | `effective_response_rate_reference`, `simulated_gross_incremental_margin`, `simulated_net_incremental_margin`, `expected_contact_cost_if_targeted` |
| String decisions | `promotion_scenario`, `final_recommendation` |
| Boolean selection | `targeted_under_budget_reference_scenario` |
| Nullable rank | `target_rank` (all blank, read as float64; writer uses nullable integer) |

Source: `data/processed/customer_test_scores.csv::header,dtypes,isna aggregates` (**new derived schema inspection**); writers `src/03_baseline_models.py:154-162`, `src/05_decision_scenario.py:425-432`. Saved scenario is `moderate`, source `baseline`, split `final_test`, design `expanding_temporal_snapshots`, training outcome end `2011-08-30`.

## 5. Models, preprocessing, validation and model selection

### 5.1 Specifications and baselines

| Model/comparator | Confirmed implementation |
|---|---|
| Naive classification | Constant **fitting-partition** repeat rate for every evaluation customer. Its average precision equals the evaluation prevalence; it is not trained on evaluation labels. No stored naive ROC-AUC or bootstrap CI. |
| Naive future sales | Constant **fitting-partition mean** `future_gross_sales`; mean is recomputed when final fitting adds validation. |
| RFM baseline | Behavioral segmentation and business ranking, not a separately fitted classification/regression predictor with reported PR-AUC/MAE. |
| Logistic Regression | `make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,random_state=42))`; seven raw numeric features; scale fitted only on fitting rows. |
| Linear Regression | `LinearRegression()` on the same seven numeric features, **without StandardScaler**. Raw prediction retained; evaluated value `max(raw,0)`. |
| GBM classifier | `GradientBoostingClassifier(random_state=42)`; all other parameters library defaults. |
| GBM regressor | `GradientBoostingRegressor(random_state=42)`; all other parameters library defaults; values clipped for evaluation. |

Sources: `src/03_baseline_models.py:114-151`; `src/04_advanced_models.py:46-49,72-116`. There is one GBM candidate per task, no hyperparameter grid search and no final-test tuning. Classification PR-AUC is **average precision** (`average_precision_score`), not trapezoidal integration of a plotted PR curve. Regression errors are in GBP per customer over the 90-day outcome. Advanced p90 error is `np.quantile(abs(y−prediction),0.90)` (`src/04_advanced_models.py::_metrics`).

**Verified library defaults:** read-only `get_params()` inspection under local scikit-learn **1.9.1** (the pinned version), constructing but never fitting estimators, supplements the explicit code specifications:

- StandardScaler: `copy=True, with_mean=True, with_std=True`.
- Logistic: `C=1.0, solver='lbfgs', tol=0.0001, fit_intercept=True, class_weight=None, dual=False, l1_ratio=0.0, penalty='deprecated', intercept_scaling=1, n_jobs=None, verbose=0, warm_start=False`, with explicit `max_iter=1000, random_state=42`. Preserve the actual 1.9.1 default API rather than reporting an older version's `penalty='l2'` constructor literal.
- Linear: `copy_X=True, fit_intercept=True, n_jobs=None, positive=False, tol=1e-6`.
- Both GBMs: `n_estimators=100, learning_rate=0.1, max_depth=3, subsample=1.0, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0, ccp_alpha=0.0, init=None, n_iter_no_change=None, validation_fraction=0.1, tol=0.0001, verbose=0, warm_start=False, criterion='deprecated', random_state=42`. Classifier `loss='log_loss'`; regressor `loss='squared_error', alpha=0.9`. `validation_fraction` is not an active early-stopping split when `n_iter_no_change=None`.

Evidence pointers for default inspection: `.venv/Lib/site-packages/sklearn/preprocessing/_data.py::StandardScaler`, `.venv/Lib/site-packages/sklearn/linear_model/_logistic.py::LogisticRegression`, `.venv/Lib/site-packages/sklearn/linear_model/_base.py::LinearRegression`, `.venv/Lib/site-packages/sklearn/ensemble/_gb.py::GradientBoostingClassifier,GradientBoostingRegressor`; method `get_params()`; version pin `requirements.txt:3`. These are current dependency defaults, not a serialized historical hyperparameter audit.

### 5.2 Validation-only selection and fixed business source

| Task | Gate, fixed in config | Validation result | Outcome |
|---|---|---|---|
| Classification | GBM AP gain ≥0.01 **and** Brier increase ≤0.005 | AP gain 0.0236544317; Brier change −0.0054985360 | GBM selected as advanced challenger |
| Regression | Relative MAE gain ≥0.05 **and** relative p90-error increase ≤0.05 | MAE gain 0.0572890839; p90 change −0.1065546884 | GBM selected as advanced challenger |
| Sigmoid calibration | Validation Brier gain ≥0.002 **and** AP loss ≤0.005 | Brier gain only 0.0000959933; AP rises slightly | **Not retained** |

Sources: `config/params.yaml:41-48`; `src/04_advanced_models.py::select_models,evaluate_calibration`; `outputs/model_selection.csv::task,validation_gain,guardrail_value`; `outputs/calibration_metrics.csv::model,split,brier_score,pr_auc,calibration_retained`. The Brier gain subtraction is a **new derived calculation** from uncalibrated and sigmoid validation rows. Gates are deterministic adoption rules, not significance tests.

`decision.value_prediction_source='baseline'` is independent of advanced benchmark selection and remains frozen. `04` writes its GBM pair into a separately named advanced prediction file; it does not route the policy through the selected model automatically or replace baseline outputs. Final business results are therefore baseline-based, even though both challengers pass validation. This preserves the predeclared interpretable reference and prevents retrospective final-test source selection. Sources: `src/04_advanced_models.py:394-424`; `src/05_decision_scenario.py:43-60`; `docs/model_card.md:95-104`.

### 5.3 Calibration, attribution and secondary temporal evaluation

- **Calibration candidate:** a Platt-style sigmoid implemented as another `LogisticRegression(random_state=42)` fitted on the baseline logistic pipeline's `decision_function` scores. Validation candidate: base fits Train 1, sigmoid fits Train 2, evaluate validation. Final candidate: base fits both train origins, sigmoid fits validation, evaluate final. It therefore uses different base-fit training allocations from the uncalibrated comparator; the difference cannot be attributed solely to a transform of an otherwise identical final model. It is not GBM calibration and is not isotonic calibration. Source: `src/04_advanced_models.py:162-178,202-248`.
- **Calibration diagnostics:** ten equal-width bins over [0,1], empty bins omitted; ECE = `sum(n_bin×abs(observed_rate−mean_probability))/sum(n_bin)`. No bin uncertainty intervals. Source: `calibration_table,expected_calibration_error`.
- **SHAP:** `shap.TreeExplainer(final_advanced_classifier).shap_values(X_sample)`; deterministic sample of 750 final customers, canonical Customer ID order, pandas seed-42 sampling; mean absolute attribution per feature and beeswarm plot. No regressor SHAP was generated. Default explainer output is used; repository artifacts do not label a probability-unit scale, so do not report the magnitudes as probability changes or causal effects. Source: `src/04_advanced_models.py:251-285`; `outputs/advanced_feature_importance.csv::method,sample_size`.
- **Rolling origins:** six configured snapshots beginning 2010-06-01 yield five evaluation origins; each model is refitted using eligible prior snapshots with label ends strictly before the origin. This secondary design has more early training origins than the main design (§6.5). Source: `config/params.yaml:50-56`; `src/04_advanced_models.py:288-352`.
- **Uncertainty availability:** baseline bootstrap intervals exist (§6.2). Current advanced comparison, calibration, SHAP, duplicate sensitivity and rolling CSVs provide point summaries, **not** corresponding bootstrap confidence intervals. No paired confidence interval or significance test for baseline–GBM metric differences is present.

## 6. Predictive results — current authoritative numbers

### 6.1 Validation and final test versus naive

| Split | Evaluation N | Observed prevalence / naive AP | Logistic AP | Logistic ROC-AUC | Linear MAE (£) | Linear RMSE (£) | Naive MAE (£) | Naive RMSE (£) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation, 2011-06-01 | 4,933 | 0.3286032840 | 0.6952684045 | 0.8060389081 | 294.53 | 1,030.92 | 513.04 | 2,057.12 |
| Final test, 2011-09-10 | 5,281 | 0.4336299943 | 0.7501843981 | 0.7899770637 | 395.69 | 2,016.14 | 636.34 | 3,288.87 |

Source: `models/temporal_registry.json::run_id=baseline_temporal_20261027T013845Z,metrics.{validation,final_test}.{n_evaluation,naive_pr_auc,pr_auc,roc_auc,mae,rmse,naive_mae,naive_rmse}` (`:1081-1136`). Prevalence equality follows constant-score average precision; final observed rate independently agrees with score aggregate. Naive ROC-AUC was not stored; a constant score has theoretical ROC-AUC 0.5 when both classes are present, not a separately measured artifact here. Exact fitting-mean constant values were not exported in the current small summary artifacts.

### 6.2 Baseline 95% percentile bootstrap intervals

| Split | Metric | Point | 2.5th percentile | 97.5th percentile |
|---|---|---:|---:|---:|
| Validation | AP | 0.695268 | 0.670952 | 0.717693 |
| Validation | ROC-AUC | 0.806039 | 0.792754 | 0.818855 |
| Validation | MAE (£) | 294.53 | 269.80 | 324.40 |
| Validation | RMSE (£) | 1,030.92 | 735.33 | 1,352.39 |
| Final test | AP | 0.750184 | 0.731190 | 0.768797 |
| Final test | ROC-AUC | 0.789977 | 0.777637 | 0.802009 |
| Final test | MAE (£) | 395.69 | 350.89 | 453.53 |
| Final test | RMSE (£) | 2,016.14 | 1,188.64 | 2,805.03 |

Source: same latest registry entry, metric arrays `[point,lo,hi]`. Interpretation is conditional fitted-model customer-bootstrap uncertainty (§3.2), not temporal generalization or causal uncertainty. The large RMSE interval is consistent with the observed high-value tail, without identifying its causes.

### 6.3 GBM versus baseline

| Split / family | AP | ROC-AUC | Brier | MAE (£) | RMSE (£) | Median absolute error (£) | P90 absolute error (£) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation / baseline | 0.695268 | 0.806039 | 0.161353 | 294.53 | 1,030.92 | 112.47 | 616.99 |
| Validation / GBM | 0.718923 | 0.814990 | 0.155854 | 277.66 | 1,120.81 | 99.74 | 551.24 |
| Final / baseline | 0.750184 | 0.789977 | 0.199542 | 395.69 | 2,016.14 | 120.42 | 776.47 |
| Final / GBM | 0.769056 | 0.798659 | 0.199442 | 400.40 | 2,167.29 | 103.34 | 792.10 |

Source: `outputs/temporal_model_comparison.csv::(split,model),pr_auc,roc_auc,brier_score,mae,rmse,median_absolute_error,p90_absolute_error`; matching advanced registry run in §0. **GBM regression reverses on final-test MAE**, and its RMSE is worse on both validation and final test. Its lower median final error does not offset the reported mean/tail error metrics. Classifier ranking improves on final AP/ROC-AUC, but the Brier difference is small and ECE is worse (§6.4); no claim of statistically established superiority is supported.

Recorded computational cost, for the classifier/regressor **pair**: validation baseline/GBM fitting 0.044726/1.872781 seconds; final fitting 0.017731/2.932948 seconds; final inference 0.630922/2.043533 ms per 1,000 rows. Source: same CSV `fit_seconds,inference_ms_per_1000`; timing logic `src/04_advanced_models.py:87-102`. These are single-run environment-dependent timings, not repeated performance benchmarks.

### 6.4 Calibration and predictive attribution

| Split / classifier | Brier | ECE, 10 equal-width bins | AP | ROC-AUC | Sigmoid retained? |
|---|---:|---:|---:|---:|---|
| Validation / logistic | 0.161353 | 0.026149 | 0.695268 | 0.806039 | Not applicable |
| Validation / GBM | 0.155854 | 0.018123 | 0.718923 | 0.814990 | Not applicable |
| Validation / logistic sigmoid | 0.161257 | 0.026548 | 0.695511 | 0.806516 | **No** |
| Final / logistic | 0.199542 | 0.116981 | 0.750184 | 0.789977 | Not applicable |
| Final / GBM | 0.199442 | 0.130112 | 0.769056 | 0.798659 | Not applicable |
| Final / logistic sigmoid | 0.198787 | 0.107230 | 0.745961 | 0.787062 | **No** |

Source: `outputs/calibration_metrics.csv::(model,split),brier_score,ece_equal_width_10_bins,pr_auc,roc_auc,calibration_retained`. Final sigmoid Brier cannot override its failed validation adoption gate. Baseline final bin 0 has N=1,420, mean forecast 0.056749 versus observed 0.151408; bin 2 has N=747, forecast 0.252357 versus observed 0.428380. These illustrate underprediction in those bins, not universal calibration failure for every customer. Source: `outputs/calibration_table.csv::model=logistic_regression,split=final_test,bin∈{0,2}; n,mean_predicted_probability,observed_positive_rate`. Figure: `outputs/figures/temporal_calibration_curve.png`. ECE depends on binning and has no stored CI.

| GBM classifier feature | Mean absolute Tree SHAP attribution |
|---|---:|
| `frequency` | 0.641238 |
| `recency_days` | 0.573394 |
| `monetary` | 0.215441 |
| `tenure_days` | 0.189631 |
| `avg_order_value` | 0.121503 |
| `n_distinct_products` | 0.040199 |
| `return_rate` | 0.009874 |

Source: `outputs/advanced_feature_importance.csv::feature,mean_abs_attribution`, all `model=gradient_boosting_classifier,task=classification,method=tree_shap,sample_size=750`. These magnitudes explain predictive contribution within this fitted challenger. Mean absolute values give neither a signed effect nor a causal intervention ranking; correlated RFM features can share predictive information. Figure: `outputs/figures/shap_summary.png`.

### 6.5 Secondary rolling-origin robustness

Every cell below is from `outputs/rolling_origin_metrics.csv::(decision_date,model,metric),value`; N fields are `n_train,n_evaluation`. These metrics do not replace §6.1.

| Evaluation origin | Fitting rows | Evaluation N | Family | AP | ROC-AUC | MAE (£) | RMSE (£) |
|---|---:|---:|---|---:|---:|---:|---:|
| 2010-09-01 | 2,703 | 3,321 | Baseline | 0.787876 | 0.725144 | 556.91 | 2,313.79 |
| 2010-09-01 | 2,703 | 3,321 | GBM | 0.789703 | 0.723115 | 525.66 | 1,807.80 |
| 2010-12-01 | 6,024 | 4,266 | Baseline | 0.659845 | 0.776004 | 609.32 | 2,180.74 |
| 2010-12-01 | 6,024 | 4,266 | GBM | 0.661687 | 0.773705 | 617.95 | 2,232.95 |
| 2011-03-01 | 6,024 | 4,537 | Baseline | 0.679890 | 0.772415 | 690.04 | 2,757.94 |
| 2011-03-01 | 6,024 | 4,537 | GBM | 0.685027 | 0.766718 | 716.31 | 2,607.05 |
| 2011-06-01 | 14,827 | 4,933 | Baseline | 0.706138 | 0.811765 | 335.03 | 1,091.10 |
| 2011-06-01 | 14,827 | 4,933 | GBM | 0.718009 | 0.810949 | 302.21 | 1,126.20 |
| 2011-09-10 | 19,760 | 5,281 | Baseline | 0.755962 | 0.792269 | 397.64 | 1,901.45 |
| 2011-09-10 | 19,760 | 5,281 | GBM | 0.766368 | 0.795904 | 408.16 | 2,223.74 |

**New derived schedule detail:** adding 90 calendar days to the six configured snapshot dates gives label ends 2010-08-30, 2010-11-30, 2011-03-01, 2011-05-30, 2011-08-30, 2011-12-09. Initial fitting snapshot is 2010-06-01 (2,703 customers, inferred directly from first origin's `n_train`). Eligible prior decisions by evaluation origin are: September 2010 → June 2010; December 2010 → June/September 2010; March 2011 → June/September 2010 (December excluded because its label end equals origin); June 2011 → June/September/December 2010 and March 2011; September 2011 → all five previous snapshots. Source/formula: `config/params.yaml:32,50-56`; `src/04_advanced_models.py::rolling_origin_schedule,eligible_rolling_training`.

| Family | Mean AP [min,max] | Mean ROC-AUC | Mean MAE [min,max], £ | Mean RMSE, £ |
|---|---|---:|---|---:|
| Baseline | 0.717942 [0.659845,0.787876] | 0.775520 | 517.79 [335.03,690.04] | 2,049.00 |
| GBM | 0.724159 [0.661687,0.789703] | 0.774078 | 514.06 [302.21,716.31] | 1,999.55 |

Aggregates are **new derived calculations**, arithmetic mean/min/max over the five origin rows for each `(model,metric)` in `rolling_origin_metrics.csv`, not pooled customer metrics or uncertainty intervals. They corroborate the rounded summaries in `docs/model_card.md:136-145`. GBM AP exceeds baseline at all five origins, MAE is lower at two of five, and ROC-AUC is lower at the first four. Variation is substantial; ranking gains do not establish uniformly better forecasting. The later rolling result differs from the main final result because its fitting set has 19,760 rather than 13,736 rows. Figure: `outputs/figures/rolling_origin_performance.png`.

## 7. Promotion economics and optimization logic

### 7.1 Implemented fixed-unit-cost accounting

Let `v=max(predicted_future_value,0)` in GBP/customer/90 days; `g` = gross-margin fraction; `d` = price-discount fraction; `b` = assumed base proportional pre-discount sales-volume growth; `m_i` = segment multiplier; `r_i=b×m_i`; `c` = GBP/contact.

\[
\begin{aligned}
\text{baseline product margin}&=vg,\\
\text{promoted revenue}&=v(1+r)(1-d),\\
\text{promoted COGS}&=v(1+r)(1-g),\\
\text{gross incremental margin}&=v\{(1+r)(g-d)-g\},\\
\text{NET incremental margin if contacted}&=v\{(1+r)(g-d)-g\}-c.
\end{aligned}
\]

This is the requested implemented formula: `net_incremental_margin = v * ((1 + r) * (g - d) - g) - contact_cost`. Unit product costs remain fixed while volume changes. `r` is proportional sales-volume growth, **not** a binary treatment-response probability. No multiply-by-repeat-probability step occurs. `v` predicts historical-behavior sales and is not identified untreated revenue. Sources: `src/05_decision_scenario.py:14-20,63-80`; `config/params.yaml:60-86`; known-value accounting test `tests/test_decision_scenario.py:134-137` (v=£1,000,r=.2,d=.1,g=.4,c=.5 → −£40.50).

Clipping prevents a negative value prediction times a negative economics coefficient from becoming a spurious positive recommendation. Nonfinite `v` raises an error; at `v=0` the contribution if contacted is `−c`. Sources: `prepare_scores,incremental_margin`; `tests/test_decision_scenario.py:122-131`.

### 7.2 Optimization, ranking and comparisons

For binary contact choices `x_i`, the implemented problem is `maximize Σx_i NET_i`, subject to `cΣx_i≤B`, with no obligation to exhaust B. Because every contact costs the same, take the descending strictly positive NET contributions up to `floor(B/c)`; this is exact for this additive, uniform-cost problem. No interaction, multi-offer, heterogeneous-contact-cost, or fairness constraint is implemented. Source: `src/05_decision_scenario.py::budget_constrained_selection`.

| Policy | Definition / ordering | Actual contacts |
|---|---|---|
| No Promotion | Empty selection; zero incremental margin, not zero total business sales | 0 |
| Non-targeted | Mean of 1,000 uniform K-subsets, no replacement **within** each campaign, seed 42; canonical ascending Customer ID order; same seed restarted per scenario gives common draws | K |
| RFM | Descending `rfm_score`, then descending `monetary`, then ascending `Customer ID`, stable sort | K, even if margin is negative |
| Proposed | `NET>0`, descending NET stable sort; after canonical ID sorting, equal margins preserve ascending ID; rank uses `method='first'`; at most K | 0…K |

Sources: `src/05_decision_scenario.py::prepare_scores,budget_constrained_selection,monte_carlo_random_policy,policy_selections,main`. The single pandas random mask in `policy_selections` is only a count/helper selection; `build_scenario_table` reports the **Monte Carlo mean**, not that one draw. Actual budget spend is `n_selected×c`. Non-targeted/RFM consume K; Proposed can abstain, so the comparison shares **maximum capacity/budget**, not equal actual spending. Current K=1,500 for all four policies. Net/contact is 0 by convention if no customers are selected (`build_scenario_table:148`).

Policy inputs are exclusively final temporal-test forecasts with the configured baseline source; neither future observed sales nor actual repeat labels determine targeting. The stored evaluation score table includes those labels for evaluation, but `incremental_margin` consumes only the forecast, RFM segment and assumptions. Provenance checks are in `src/utils.py:92-108` and `src/05_decision_scenario.py:43-60`.

## 8. Business simulation and decision results

### 8.1 Scenario inputs

| Scenario | Discount d | Base growth b | Effective growth: Champions / Loyal / At-risk / Lost | Maximum pre-contact coefficient |
|---|---:|---:|---|---:|
| Conservative | 5% | 10% | 5% / 10% / 13% / 4% | −0.0045 |
| Moderate (saved reference) | 10% | 20% | 10% / 20% / 26% / 8% | −0.0220 |
| Aggressive | 20% | 35% | 17.5% / 35% / 45.5% / 14% | −0.1090 |

All use g=40%, c=£0.50, B=£750, N=5,281, K=1,500 and multipliers Champions=.5/Loyal=1/At-risk=1.3/Lost=.4. Sources: `config/params.yaml:64-96`; `outputs/scenario_table.csv::discount_pct,response_rate_reference,gross_margin_pct,campaign_budget,campaign_capacity`. Effective rates and maximum coefficient are **derived substitutions**: `b×m`; maximum at m=1.3 of `((1+b×m)(g−d)−g)` (also documented at `docs/model_card.md:202-205`).

Because even the maximum coefficient is negative, **all nonnegative value forecasts imply negative NET if contacted** in every configured scenario. No campaign follows algebraically, not from discovering an exceptionally effective customer-ranking model. The £0 No Promotion reference is incremental, not a claim that business without promotion earns zero margin.

### 8.2 Policy totals, actual contacts/costs and signed differences

All values are **simulated GBP NET changes**, final decision 2011-09-10, baseline forecasts. Source for every row: `outputs/scenario_table.csv::(scenario,policy),customers_targeted,campaign_contact_cost,simulated_net_incremental_margin,simulated_margin_difference_vs_baseline,simulated_margin_vs_no_promotion`.

| Scenario | Policy | Contacts | Contact cost (£) | NET / difference vs No Promotion (£) | Difference vs non-targeted mean (£) |
|---|---|---:|---:|---:|---:|
| Conservative | No Promotion | 0 | 0.00 | 0.00 | +17,417.59 |
| Conservative | Non-targeted MC | 1,500 | 750.00 | −17,417.59 | 0.00 |
| Conservative | RFM | 1,500 | 750.00 | −55,333.45 | −37,915.86 |
| Conservative | Proposed | 0 | 0.00 | 0.00 | +17,417.59 |
| Moderate | No Promotion | 0 | 0.00 | 0.00 | +37,407.57 |
| Moderate | Non-targeted MC | 1,500 | 750.00 | −37,407.57 | 0.00 |
| Moderate | RFM | 1,500 | 750.00 | −118,314.35 | −80,906.77 |
| Moderate | Proposed | 0 | 0.00 | 0.00 | +37,407.57 |
| Aggressive | No Promotion | 0 | 0.00 | 0.00 | +90,677.15 |
| Aggressive | Non-targeted MC | 1,500 | 750.00 | −90,677.15 | 0.00 |
| Aggressive | RFM | 1,500 | 750.00 | −277,865.97 | −187,188.82 |
| Aggressive | Proposed | 0 | 0.00 | 0.00 | +90,677.15 |

| Scenario | Non-targeted MC mean (£) | 2.5th percentile (£) | 97.5th percentile (£) | Draws / seed / K |
|---|---:|---:|---:|---|
| Conservative | −17,417.59 | −21,707.94 | −14,023.76 | 1,000 / 42 / 1,500 |
| Moderate | −37,407.57 | −46,583.42 | −30,081.21 | 1,000 / 42 / 1,500 |
| Aggressive | −90,677.15 | −112,076.51 | −73,360.59 | 1,000 / 42 / 1,500 |

Source: same CSV `policy=non_targeted; random_mean_net_margin,random_p025_net_margin,random_p975_net_margin,random_policy_simulations,random_policy_sample_size,interval_type`; seed `config/params.yaml:4`. This is a **randomization/simulation interval across campaign selections conditional on forecasts and assumptions**, not a confidence interval for mean profit, forecast uncertainty, or a causal treatment effect.

`simulated_margin_difference_vs_baseline` uses the **non-targeted MC policy mean** as “baseline”; that word does not denote a difference from the baseline predictive model. Positive differences for Proposed mean **avoided simulated loss by abstaining**, not realized/causal savings. Negative RFM differences mean greater simulated loss. Larger high-RFM value forecasts scale up losses under negative coefficients, and Champions also have the lowest multiplier among Champions/Loyal/At-risk; high historical value alone is not sufficient to justify a discount. Source: `src/05_decision_scenario.py::build_scenario_table,policy_selections,incremental_margin`.

The reference-policy selected-contribution bootstrap is implemented and logged; with zero selected contacts it returns `(0,0,0)`. This is a degenerate conditional bootstrap, not evidence that true campaign effects have no uncertainty. Source: `src/05_decision_scenario.py:291-302,438-442`; latest primary real-data decision log `outputs/logs/05_decision_scenario.log:718-740`, specifically line 731 (`0 [0, 0]`).

## 9. Break-even, sensitivity and dashboard

### 9.1 Equations and validity domains

Setting NET=0 gives:

| Quantity | Implemented expression | Domain / interpretation |
|---|---|---|
| Required effective growth | `r*=(g+c/v)/(g−d)−1` | All inputs finite; v>0, g>d, g≤1, d≥0, c≥0. Outside domain returns NaN. With nonnegative growth and c>0, g≤d cannot yield positive NET. |
| Required base growth | `b*=r*/m` | Required growth and multiplier finite; m>0; otherwise NaN. |
| Maximum supportable discount | `d*=g−(g+c/v)/(1+r)` | Finite inputs, v>0, 0≤g≤1, r≥0, c≥0. Code intentionally restricts r≥0 even though algebra alone has a wider domain. Negative bounds retained: no nonnegative discount breaks even at that growth. |

Sources: `src/05_decision_scenario.py:174-196`; equation/domain tests `tests/test_decision_scenario.py:229-239`. Equality is zero contribution and **not eligible**: selection requires NET strictly >0. For v≤0, simulator clips value to zero (NET=−c), while break-even helpers return NaN. Quantiles omit undefined cases; `eligible_share` denominators retain the full group (`build_break_even_tables`). These are mathematical scenario conditions, not fitted response thresholds.

| Moderate, d=10%, b=20% | All customers | Valid positive-value customers | Required base growth p25 / median / p75 | Supportable discount median |
|---|---:|---:|---|---:|
| At-risk | 1,136 | 779 | 26.368042% / **26.747865%** / 28.251729% | **7.911375%** |
| Champions | 1,507 | 1,503 | 67.027460% / 67.375075% / 67.889046% | 3.539763% |
| Loyal | 1,065 | 951 | 34.077099% / 34.467919% / 35.576445% | 6.383020% |
| Lost | 1,573 | 465 | 87.513683% / 90.485473% / 99.550087% | 2.168281% |
| All | 5,281 | 3,698 | 34.024385% / 66.837584% / 67.789303% | 3.607556% |

Sources: `outputs/break_even_by_segment.csv::scenario=moderate,segment; customers,valid_break_even_customers,required_base_response_{p25,median,p75},maximum_discount_median`; all row from `outputs/break_even_summary.csv::scenario=moderate`. At-risk maximum-discount p25/p75 = 7.445893% / 8.028939% (same row `maximum_discount_p25,p75`). No customer is eligible under any of the three main scenarios. Rates in source CSVs are fractions; the table multiplies by 100 for percentage display.

### 9.2 One-dimensional tornado sensitivity

Each row varies only its stated parameter around Moderate, reselecting up to budget after the change; base selected NET is £0. Source: `outputs/tornado_sensitivity.csv::parameter,base_total_margin,low_value,low_total_margin,high_value,high_total_margin,swing`; `src/05_decision_scenario.py::tornado_sensitivity`.

| Parameter | Low value | NET at low (£) | High value | NET at high (£) | Absolute swing (£) |
|---|---:|---:|---:|---:|---:|
| Discount | 0.05 | 7,175.92 | 0.20 | 0.00 | 7,175.92 |
| Base growth | 0.10 | 0.00 | 0.35 | 3,990.80 | 3,990.80 |
| Gross margin | 0.30 | 0.00 | 0.50 | 143.77 | 143.77 |
| At-risk multiplier | 0.91 | 0.00 | 1.69 | 15.05 | 15.05 |

Figure: `outputs/figures/tornado_sensitivity.png`. Geometry uses `left=min(low−base,high−base)` and `width=abs(high−low)`, so increasing discount can reduce margin without creating negative bar widths (`tornado_bar_bounds`; `tests/test_decision_scenario.py:199-202`). Endpoint effects are not probabilistic uncertainty ranges.

### 9.3 Two-dimensional discount × growth grid

All cells hold g=.40, multipliers, c=.50 and B=750 fixed, with the same 5,281 final-test forecasts and K=1,500. Values are Proposed simulated NET versus No Promotion (£); zeros denote abstention. Source: `outputs/response_discount_surface.csv::(discount_pct,response_rate),simulated_net_incremental_margin`.

| Base growth | d=5% | d=10% | d=15% | d=20% |
|---|---:|---:|---:|---:|
| 10% | 0.00 | 0.00 | 0.00 | 0.00 |
| 20% | 7,175.92 | 0.00 | 0.00 | 0.00 |
| 30% | 20,253.23 | 1,424.27 | 0.00 | 0.00 |
| 40% | 52,483.86 | 8,738.67 | 0.00 | 0.00 |
| 50% | 86,375.02 | 18,285.43 | 962.39 | 0.00 |

| Discount | First profitable **tested** base growth | Eligible | Selected | Contact cost (£) | NET (£) |
|---|---|---:|---:|---:|---:|
| 5% | 20% | 1,600 | 1,500 | 750.00 | 7,175.92 |
| 10% | 30% | 664 | 664 | 332.00 | 1,424.27 |
| 15% | 50% | 616 | 616 | 308.00 | 962.39 |
| 20% | None through 50% | 0 | 0 | 0.00 | 0.00 |

Source: same grid row keys, `customers_eligible,customers_targeted,campaign_contact_cost`. These are discrete grid points, **not** estimated continuous minimum response thresholds or evidence to launch a promotion. Figure: `outputs/figures/response_discount_surface.png`; `src/05_decision_scenario.py::response_discount_surface,plot_response_discount_surface`.

### 9.4 Concrete dashboard behavior

| Element | Confirmed implementation |
|---|---|
| Input/provenance | Cached scores, saved scenario table, cohort table and latest baseline registry metrics; validates final-test dates/source/design. Caption says evaluation only, not a current deployment list (`dashboard/app.py:29-64`). |
| Filters | Country, RFM segment, and **saved reference-scenario** recommendation. Sandbox recommendation updates do not redefine the saved recommendation filter (`:66-81`). |
| Controls | Discount 0–40%, step 1 percentage point; assumed base growth 0–100%, step 1 point; gross margin 5–80%, step 1 point; contact budget £0–£10,000, step £50. Contact cost and multipliers remain configured (`:90-102`). |
| Scope | Reallocate contact budget within the filtered final-test subset; arithmetic only on fixed predictions, no fitting or artifact write (`:85-102`; `src/05_decision_scenario.py::evaluate_proposed_scenario`). |
| KPI 1 | Eligible customers = count of `NET>0` in filtered subset. |
| KPI 2 | Customers targeted = positive-NET selections up to subset capacity. |
| KPI 3 | Simulated NET margin = sum of selected NET contributions. |
| KPI 4 | Difference vs No Promotion = KPI 3, since reference is zero. |
| KPI 5 | Campaign contact cost = selected count × £0.50. |
| Static global display | Saved four-policy scenario comparison, random-policy error bars and table; not changed by filters/sandbox (`:116-137`). |
| Segmentation/cohorts | Filtered RFM bars, filtered predicted-sales sum/mean repeat probability, sandbox recommendation-by-segment chart; full-population cohort heatmap unaffected by filters (`:139-167`). |
| Actionable view | Selected sandbox customers sorted descending simulated NET; download CSV includes Customer ID, decision date, country, RFM segment, action, forecast, repeat probability, scenario, NET, contact cost, rank and all four sandbox inputs (`:171-192`). |

KPI definitions are implemented in `src/05_decision_scenario.py:228-253`, displayed at `dashboard/app.py:104-112`. There are no contacts in the default Moderate view. A positive exploratory target list is conditional on chosen assumptions and is not an observed response list or a deployable current-customer campaign. Static PNGs/CSVs and interactive charts are distinct deliverables (§12).

## 10. Robustness, quality and reproducibility

### 10.1 Duplicate-policy sensitivity

| Policy | Raw rows | Rows after duplicate handling | Removed exact duplicates | Final customers | Repeat prevalence | AP | ROC-AUC | MAE (£) | RMSE (£) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Retain, primary | 1,067,371 | 1,067,371 | 0 | 5,281 | 0.4336299943 | 0.7501843981 | 0.7899770637 | 395.69 | 2,016.14 |
| Drop exact duplicates | 1,067,371 | 1,033,036 | 34,335 | 5,281 | 0.4336299943 | 0.7501906165 | 0.7899710777 | 396.54 | 2,066.55 |

Source: `outputs/duplicate_robustness_summary.csv::duplicate_policy; raw_rows,rows_after_duplicate_handling,exact_duplicate_rows_removed,final_test_customers,repeat_purchase_rate,logistic_pr_auc,logistic_roc_auc,linear_mae,linear_rmse`. Both use decision 2011-09-10 and horizon 90, baseline models and original economics.

| Scenario | Drop-duplicate non-targeted MC mean (£) | Drop-duplicate RFM NET (£) | Proposed contacts / NET, both policies |
|---|---:|---:|---|
| Conservative | −16,612.70 | −52,717.21 | 0 / £0 |
| Moderate | −35,638.18 | −112,679.38 | 0 / £0 |
| Aggressive | −86,340.30 | −264,583.53 | 0 / £0 |

Source: `outputs/duplicate_robustness_business_metrics.csv::(duplicate_policy,scenario),non_targeted_monte_carlo_mean_net_margin,rfm_net_margin,proposed_contacts,proposed_net_margin,campaign_recommendation`. Retain comparison values are in §8. Both policies yield `do_not_launch_configured_campaigns`, `recommendation_changed_vs_retain=False`.

Implementation copies raw data or applies `drop_duplicates(keep='first')`, reruns the same functions in memory, and writes only two namespaced CSV outputs (`src/07_duplicate_robustness.py::apply_duplicate_policy,evaluate_duplicate_policy,write_robustness_outputs`). It does not update the main registry/prediction artifacts. This assesses sensitivity to exact-line retention, not fuzzy duplicates or whether the repeated lines are genuinely erroneous. Since all configured economics coefficients are negative, campaign abstention is structurally stable even when value forecasts change; unchanged abstention alone is not strong evidence of model robustness.

### 10.2 Historical verification evidence and scope

| Check | Latest verifiable record | What it establishes / limitation |
|---|---|---|
| Full baseline QA | `outputs/logs/06_final_qa.log:306-348`: completed 02→03→05→07; **48 passed in 9.53s**, final QA PASSED at local log time 2026-10-27 08:40:29 | Historical run, not newly rerun. Earlier 36/40/45/48-pass records are earlier stages, not conflicting latest counts. |
| QA summary | `outputs/final_qa_report.md:3-6`: 4 steps, source baseline; pytest PASSED; invariant checks PASSED; manifest generated and hash-validated | No separate numeric count of invariant assertions is recorded; do not invent one. |
| Schema check | `outputs/logs/data_validation.log:11`: 1,067,371 rows / 8 columns PASSED, 2026-10-27 08:38:17 | Raw schema/type contract, not evidence that all economic measurements are correct. |
| Temporal/mechanics tests | Feature window tests; final-label mutation; scaler mean equals fitting rows; validation-only selection; calibration chronology; rolling chronology/sample stability | Specific functions referenced in §§3,5; synthetic regression tests do not substitute for real-data predictive metrics. |
| Economics/policy checks | `tests/test_decision_scenario.py:122-258` | Clipping, contact accounting, capacity despite abstention, deterministic Monte Carlo, invalid break-even domains, surface dimensions and tornado geometry. Does not validate assumed response rates. |
| Duplicate output isolation | `tests/test_duplicate_robustness.py:74-83` sentinel preservation and disjoint names; writer `src/07_duplicate_robustness.py:134-138` | Test protects namespaced outputs. `docs/traceability_map.md:33-35` additionally records unchanged primary hashes during execution, but a full before/after checksum ledger is not present in inspected artifacts. |
| Manifest hash tests | `tests/test_decision_scenario.py:260-276`; `src/06_final_qa.py:135-146` | Synthetic input-byte hash stability/change sensitivity and actual-run manifest round-trip checks. Does not identify versions of every generated artifact. |
| Merge idempotency | `tests/test_pipeline_idempotency.py::test_merge_or_replace_columns_no_duplicate_on_rerun,test_merge_or_replace_columns_idempotent_across_multiple_calls` | Legacy merge helper avoids `_x/_y` columns; not proof that every current pipeline output is byte-identical across executions. |
| Prior compliance audit | `BA_CAPSTONE_FINAL_COMPLIANCE_FIXES_PROMPT.md:54-69`: **35 required PASS, 1 PARTIAL**, 48 tests, QA, pip check and dashboard server/AppTest passing | Historical audit summary; the one PARTIAL was lack of rendered ≤2-page brief. The PDF/script now exist, but no new numeric compliance audit was run for this pack. |
| Streamlit smoke/AppTest | Same historical summary, `:65`, says passing | Detailed assertions, screenshots, HTTP status/timing and exercised control/filter combinations are **NOT AVAILABLE** in inspected logs/tests. Do not expand this into exhaustive UI validation. Current configured baseline dashboard behavior was verified by source inspection only. |

`src/06_final_qa.py::check_invariants` checks population sizes, nonnegative history/business inputs, return-rate bounds, allowed actions, label consistency, and recomputed scenario/break-even/grid agreement; `check_submission_artifacts` checks required files, baseline/retain policy, robustness labels and temporal design. It regenerates artifacts, so it was deliberately not executed during evidence extraction. Advanced step 04 is not rerun by baseline QA; current advanced outputs have their separately recorded run (§0), while submission checks establish their presence, not freshness by retraining.

### 10.3 Reproduction recipe already documented

Python 3.11 environment, pinned `requirements.txt`, source download/checksum, then `python src/02_cleaning_features.py` → `python src/03_baseline_models.py` → `python src/05_decision_scenario.py` → `python src/07_duplicate_robustness.py`. Advanced benchmark `python src/04_advanced_models.py` follows 02/03 independently. `python src/06_final_qa.py` reruns baseline steps/tests/invariants and writes a manifest; `streamlit run dashboard/app.py` consumes outputs. These are **documented reproduction commands, not commands executed for this pack**. Source: `README.md:18-74,183-198,217-224`; actual QA step list `src/06_final_qa.py:24-35`.

Read-only extraction used `.venv\Scripts\python.exe -B -c ...` for small CSV aggregates, text-file hashes and dependency parameter inspection, avoiding project-module imports that initialize file loggers (`src/utils.py::get_logger`). No raw workbook, large intermediate table or binary model was inspected. This pack adds evidence summaries, not a new analytical run.

## 11. Critical assumptions, threats to validity and ethics

| Domain | Evidenced limitation | Report/deployment implication |
|---|---|---|
| Data/measurement | Missing IDs exclude 22.766873% of raw lines; 34,335 exact duplicates retained; returns not netted; first-observed tenure/cohort may be left-censored; first country is a simplification (§§2,4). | Population selection and value measurement may differ from commercially realized net revenue. Duplicate sensitivity cannot determine the true row semantics. |
| Value target | 90-day positive gross sales, with a heavy observed upper tail and zero median; 1,583 raw linear forecasts clipped (§2.3). | Do not call target lifetime CLV/profit; clipping ensures a nonnegative domain, not accurate calibration or reliable high-spender forecasts. |
| Temporal validity | Older UK-retailer dataset, expanding histories of different lengths, repeat customers across origins, one main final origin and five secondary origins; September cohort month partial (§§2–3,6.5). | External transfer to contemporary channels/customers is unvalidated; origins are not independent random experiments; no monotonic drift claim is supported. |
| Forecasting/calibration | Final ECE increases, GBM regression final MAE reverses, broad baseline RMSE CI; no paired challenger CI (§6). | Small gains should not be overstated; customer bootstrap omits model-selection/time-regime uncertainty. |
| Business economics | g=.40, c=.50, growth, discounts and segment multipliers are configured rather than measured; uniform contact cost and fixed unit COGS; additive customer contributions (§7). | Policy conclusions are conditional on these assumptions, with no inventory, cannibalization, customer-interaction or campaign fixed-cost model evidenced. |
| Causal identification | No randomized promotion/control; historical promotions may already influence sales; regression forecasts are historical-behavior outcomes, not untreated potential outcomes (§1). | No observed lift, causal profit/ROI or realized savings. A proposed randomized pilot in the roadmap is a future design, not completed causal evidence. |
| Fairness | Seven behavioral predictors exclude country directly, but spending/recency and segment rules can still produce group differences; current country-error tables are historical (§0). | No current protected-group impact audit or fairness guarantee. Potential disparate impact is a **future concern**, not a measured finding. |
| Privacy/governance | Customer-level IDs, predicted values and recommendation lists are present; dashboard can export selected records (§4.3,§9.4). | Access control, minimization, retention/deletion rules and human review are prospective safeguards, not demonstrated deployments. Do not reproduce individual customer records in the academic report. |

Sources beyond the section-level primary evidence: `docs/model_card.md:254-272`; `docs/production_roadmap.md::Monitoring & drift plan,Path to causal validation,Data privacy & compliance,Rollout strategy`; `phantichkinhdoanh.pdf::Mandatory Analytics Rules,Scope Guardrails`. The roadmap proposes quarterly temporal monitoring/retraining, randomized holdouts, shadow/canary rollout and access controls; it does not establish that these are operational. Its example pilot needs an independently justified eligible population: current configured `nurture` count is zero. Legal compliance is not established by the existence of the roadmap.

## 12. Figure/table inventory for the report

Filenames below were verified in the existing `outputs/figures/` directory; dynamic views were verified in `dashboard/app.py`. No figures were created or rerendered during extraction.

| Existing asset | Status | Suggested section / caption |
|---|---|---|
| `outputs/figures/temporal_calibration_curve.png` | Current static PNG | Predictive results: “Final temporal-test calibration; uncalibrated logistic/GBM and rejected sigmoid candidate.” |
| `outputs/figures/shap_summary.png` | Current static PNG | Model interpretation: “Tree SHAP for the GBM classifier, deterministic 750-customer final sample; predictive, not causal.” |
| `outputs/figures/rolling_origin_performance.png` | Current static PNG | Robustness: “Secondary rolling-origin AP and MAE with fully observed earlier labels.” |
| `outputs/figures/tornado_sensitivity.png` | Current static PNG | Business sensitivity: “One-at-a-time assumption effects on budget-constrained simulated NET, Moderate reference.” |
| `outputs/figures/response_discount_surface.png` | Current static PNG | Business sensitivity: “Discount × assumed growth grid: Proposed NET versus No Promotion; zero cells abstain.” |
| `outputs/figures/calibration_curve.png` | **Historical — exclude** | Superseded pre-remediation calibration (`outputs/RESULTS_STATUS.md:37-48`). |
| `dashboard/app.py:116-137` scenario bars/table | Existing interactive chart; no static image | Decision results: “Equal maximum capacity; Random/RFM use K while Proposed may abstain; random-policy percentile bars.” |
| `dashboard/app.py:139-155` segment/action bars | Existing interactive charts; no static image | EDA/dashboard: “RFM population counts and scenario-dependent recommendations in the filtered subset.” |
| `dashboard/app.py:159-167` cohort heatmap | Existing interactive chart; no static image | EDA: “Monthly activity by first-observed purchase cohort; final September history is partial.” |
| `dashboard/app.py:104-112,171-192` KPIs/actionable view | Existing UI; no saved screenshot | Dashboard: “Filtered scenario economics and downloadable selected-customer list, without retraining.” |
| `docs/executive_brief.pdf` / `.md` | Existing rendered/narrative brief | Managerial summary reference; ≤2-page rendering requirement is separate from full-report length. |

| Existing CSV/table | Suggested report table/use |
|---|---|
| `outputs/temporal_schedule.csv` | Half-open timeline table; expand with sample counts from §3. No standalone timeline figure exists. |
| `outputs/data_quality.csv` + dated cleaning log block | Raw/duplicate audit and non-overlapping-within-window removal table (§2). |
| `outputs/monthly_transaction_profile.csv` | Optional raw activity table with cleaning/partial-month caveats. |
| `outputs/cohort_retention.csv` | Cohort sample table or dashboard heatmap data. |
| `data/processed/customer_test_scores.csv` | Aggregate population/value/segment table (§§2,4); not full individual-record appendix. |
| `models/temporal_registry.json` | Main baseline/naive metrics and CI table (§6.1–6.2). |
| `outputs/temporal_model_comparison.csv`, `outputs/model_selection.csv` | Model comparison/adoption-gate tables (§5.2,§6.3); no standalone model-comparison figure exists. |
| `outputs/temporal_calibration_logreg.csv` | Baseline final calibration points; optional alongside richer current calibration table. |
| `outputs/calibration_metrics.csv`, `outputs/calibration_table.csv` | Brier/ECE and reliability-bin evidence (§6.4). |
| `outputs/advanced_feature_importance.csv` | SHAP ranked-attribution table (§6.4). |
| `outputs/rolling_origin_metrics.csv` | All-origin metric/sample-size table (§6.5). |
| `outputs/scenario_table.csv` | Three scenarios × four policies, contacts/cost/NET and random-policy intervals (§8). |
| `outputs/break_even_by_segment.csv`, `outputs/break_even_summary.csv` | Segment/all-customer break-even quantiles with valid denominators (§9.1). |
| `outputs/tornado_sensitivity.csv`, `outputs/response_discount_surface.csv` | One-/two-dimensional assumption sensitivity (§9.2–9.3). |
| `outputs/duplicate_robustness_summary.csv`, `outputs/duplicate_robustness_business_metrics.csv` | Retain/drop predictive and decision robustness (§10.1); no standalone duplicate-comparison figure exists. |
| `outputs/run_manifest.json`, `outputs/final_qa_report.md` | Reproducibility/provenance appendix with historical-run qualification (§0,§10). |

**Requested/useful visuals not available as saved figures:** temporal timeline, RFM/cohort static plot, model-comparison plot, scenario-comparison static plot, duplicate robustness plot, dashboard screenshot, raw missingness graphic, gross-sales distribution/Pareto chart. Existing tables/dynamic charts can support the report; producing new visuals is a later authoring task.

## 13. Recommended academic report outline

This is a section/evidence map, not report prose. No full-report page limit is assumed.

| Proposed section/subsections | Evidence-pack mapping | Existing exhibits | Executive brief versus full report |
|---|---|---|---|
| 1. Introduction / Business Problem: CRM question; prediction versus campaign decision | §§1,7–8 | Scenario policy table | Brief: question, explicit decision and principal assumptions. Full: precise unit/objective/guardrails. |
| 2. Related Context: customer analytics, temporal prediction, calibration and scenario targeting | §§0,4–5,11 | Dataset/instructor context only | **External academic literature research needed**; no verified scholarly bibliography supplied. |
| 3. Dataset & Exploratory Analysis: source; line granularity; cleaning; cohorts; value concentration | §§2,4 | Data-quality table, cohort CSV/dashboard, derived aggregate table | Brief: scale/target caveat. Full: counts, denominator warnings, all measurement definitions. |
| 4. Methodology: temporal snapshots; leakage prevention; feature construction; RFM; models | §§3–5 | Timeline and feature tables | Brief: one-line temporal design. Full: equations, preprocessing, default parameters and chronology. |
| 5. Experimental Setup & Validation: main split; refit; adoption gates; uncertainty | §§3,5,10 | Schedule, model-selection CSV, QA evidence | Full report detail; brief may state final temporal holdout and fixed baseline policy source. |
| 6. Results & Model Comparison: naive/baseline/challenger; calibration; interpretation | §6 | Metric/CI tables, calibration PNG, SHAP PNG | Brief: baseline accuracy, limited GBM improvement and regression instability. Full: exact split results, rejected calibration and uncertainty. |
| 7. Promotion Decision & Scenario Analysis: accounting; optimization; four policies; abstention | §§7–8 | Scenario CSV and dashboard policy chart | Brief priority: do not launch configured campaigns; actual spend and avoided **simulated** loss. Full: accounting derivation and comparability. |
| 8. Robustness & Sensitivity: duplicate handling; temporal variation; break-even; tornado/grid | §§6.5,9.1–9.3,10.1 | Rolling/tornado/heatmap PNGs, duplicate tables | Brief: break-even decision gate and unchanged duplicate recommendation. Full: domains, denominators, all origins and tested grid. |
| 9. Dashboard & Business Recommendations: controls/KPIs; actionable view; evidence needed for a pilot | §§9.4,11 | Existing interactive dashboard; executive brief | Brief: next decision gate. Full: filtered/global scopes and prospective monitoring/experiment plan. |
| 10. Limitations, Ethics & Conclusion: measurement, temporal, predictive, economics and causal validity | §11; unresolved items §14 | Limitations table | Both: no causal claims, historical-data limits. Full: nuanced evidence versus hypothetical concerns. |
| Reproducibility/technical appendix | §§0,3–5,10,12 | Manifest, feature/schema table, exact output pointers | Full only; keep two-page executive brief decision-focused. |

The instructor's executive brief maximum is **two pages** (`phantichkinhdoanh.pdf::Required Project Package`, printed p.2). Existing `docs/executive_brief.md` already covers recommendation, assumptions, policy comparison, sensitivity and limitations. Full methodological/metric tables belong in the academic report or appendices, not compressed into that brief.

## 14. Missing inputs, contradictions and questions for the report author

Only unresolved gaps or interpretation conflicts remaining after extraction are listed. The official requirements PDF, recorded download date, sample sizes, seven features, primary model source and main scenario parameters are already available and should not be requested again.

| Item | Evidence / why it matters | Smallest resolution or report treatment |
|---|---|---|
| Full academic-report formatting and attribution | `phantichkinhdoanh.pdf` specifies project package and ≤2-page executive brief, but not full-report length, citation style, author/team contribution details or a detailed full-report marking rubric. | Author supplies any separate full-report instructions and actual team contributions. Do not infer the report length from the brief limit. |
| Academic literature/bibliography | Reviewed current methodology docs contain dataset/instructor context but no verified scholarly literature bibliography. | Add a separately verified literature search and references during report writing; do not fabricate citations here. |
| Exact raw timestamps and complete dataset-level quality accounting | Current audit CSV has raw/duplicate totals; logs have missing IDs and overlapping per-window removals. Exact raw min/max times, sheet row counts, all-field missingness and mutually exclusive whole-workbook invalid-row totals are not preserved in inspected current outputs. | Report available counts with scope labels. If essential, authorize a later read-only raw-profile audit; do not sum snapshot logs or treat zero residual quantity removals as zero raw negative quantities. |
| Source-EDA and acquisition provenance limits | Date 2026-10-26 is recorded, but independent download receipt is absent. Monthly profile is optional source EDA, not regenerated by baseline QA; exact transaction timestamps were not rescanned. | Cite the recorded date as such and qualify monthly EDA. No need to ask for a download date already in config. |
| Detailed UI verification and isolation ledger | Historical prompt states server/AppTest passing; traceability doc states primary hashes unchanged. Detailed AppTest assertions and before/after artifact checksum ledger are absent from inspected evidence. | Cite these as historical summaries only; retrieve original audit records if a stronger claim is needed. No rerun was performed here. |
| Analytical versus release provenance | Manifest commit `f2164e8`, advanced commit `334a105`, current HEAD `bb27c85`; manifest dirty=true; current pre-extraction checkout has one pre-existing untracked PDF. | Explicitly distinguish run provenance from release/extraction state (§0); do not edit manifest or claim all artifacts were generated at current HEAD. |
| Calibration chronology wording | Docs describe strict chronology; `fit_sigmoid_calibrator` allows fit-label end **equal to** calibration origin, unlike strict `<` in rolling training. | Use exact half-open/equality semantics (§3.2), which still keep fitted outcomes available at calibration time. |
| Advanced selection versus business deployment wording | Both GBMs pass validation, but advanced prediction export always contains the GBM pair and business source remains baseline; final GBM regression worsens. | State separate challenger selection and fixed policy source. Do not imply automatic production adoption or reselect retrospectively. |
| Static report visuals and current fairness diagnostics | Several useful visuals exist only interactively; no saved raw-missingness/value-distribution/timeline/duplicate plot or dashboard screenshot. Country-error artifacts are explicitly historical. | Use existing tables/PNG assets and mark missing graphics; authorize later screenshots/new figures only if required. Do not reuse historical country tables as a current fairness audit. |
| Unmeasured economics and causal response | No observed promotion-response estimate, measured margin/cost/multiplier justification or randomized causal outcomes in current repository. | Keep assumptions explicit and results conditional. Obtain operational economics/experimental evidence only for a future deployment decision, not to manufacture positive results. |

**Reconciliations, not numerical conflicts:** current baseline/advanced metrics agree between registry, CSVs and current docs at displayed precision; tiny final-digit CSV summation differences are rounding. Main versus rolling final metrics differ because training populations differ. Old 5,041-customer descriptions and legacy metrics are superseded (`docs/data_dictionary.md:36-39`; `outputs/RESULTS_STATUS.md`). The current results-status title emphasizes Phase 1/2, but its explicit current-calibration list and the separate Phase 3 artifacts govern current advanced evidence. The final report must preserve these distinctions.

**Extraction boundary:** only this new evidence-pack Markdown file was produced by the extraction. No extraction operation modified analytical code, configuration, models, outputs, logs or existing documentation; final Git checks showed no tracked changes. The independently observed disappearance of the initially untracked PDF is recorded in §0.1 rather than silently claiming identical overall workspace state. This pack is complete as an evidence deliverable; final academic report writing remains a separate task.
