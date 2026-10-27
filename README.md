# Customer Value & Promotion Targeting for an Online Retailer

Capstone project — HUST Business Analytics 2026. Chi tiết yêu cầu: `phantichkinhdoanh.pdf`.
`PLAN.md` is the archived pre-remediation work log, not current validation evidence.

**Current evidence:** the real UCI baseline rerun evaluates **5,281 customers** at
2011-09-10 with a **43.36%** repeat-purchase rate. The current customer-level artifact
is `data/processed/customer_test_scores.csv`, explicitly allowed by `.gitignore`.
Legacy `customer_scores.csv` and `models/registry.json` are not current evidence;
see `outputs/RESULTS_STATUS.md`. No large intermediate processed tables are required
in Git.

## Business question

Khách hàng nào nên được ưu tiên cho retention/promotion để tăng expected future
margin mà không over-discount?

## Cài đặt

Python 3.11; use the pinned dependencies below. `curl` and `unzip` are required
for the download example (or extract the archive with your OS archive tool).

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python --version              # must report Python 3.11.x
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS/Linux, replace the first two commands with `python3.11 -m venv .venv`
and `source .venv/bin/activate`.

## Dữ liệu

Dataset: [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii).
Xem chi tiết nguồn/ngày tải/filter trong `docs/data_dictionary.md`.

Tải và giải nén:

```powershell
curl.exe -L -o data/raw/online_retail_ii.zip https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip
Expand-Archive -Force data/raw/online_retail_ii.zip data/raw/
Get-FileHash data/raw/online_retail_II.xlsx -Algorithm SHA256
```

Expected workbook SHA-256:
`bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980`.

## Chạy pipeline (theo thứ tự)

```bash
python src/01_eda.py                  # optional source EDA
python src/02_cleaning_features.py    # duplicate audit + temporal snapshots + cohort
python src/03_baseline_models.py      # RFM + logistic/linear temporal evaluation
python src/05_decision_scenario.py    # baseline-only equal-capacity NET scenarios
python src/07_duplicate_robustness.py # secondary retain-vs-drop duplicate sensitivity
```

Default `decision.value_prediction_source: baseline` remains the frozen business
source. Run the independent Phase 3 advanced benchmark after steps 02 and 03:

```bash
python src/04_advanced_models.py      # GBM + calibration + Tree SHAP + rolling origins
```

Phase 3 selection is validation-only and does not alter the Phase 1/2 promotion
policy. Both source files use the same prediction-column meanings; baseline remains
the primary reference and configured scenario input.

```bash
python src/06_final_qa.py             # reruns 02→03→05→07 + tests/invariants;
                                    # inserts 04 only for advanced source
```

### Corrected analytical design

- Expanding history from 2009-12-01, always strictly before the decision time.
- Train decisions: **2010-12-01, 2011-03-01**; validation: **2011-06-01**;
  final test: **2011-09-10**. Every outcome is **90 days**.
- Validation fits train only. Final test refits train+validation after their
  outcomes finish (2011-08-30 exclusive < 2011-09-10). No random customer split.
- Final outcome: [2011-09-10, 2011-12-09); excludes potentially partial 2011-12-09.
  Raw coverage is checked. Full date table: `docs/problem_statement.md`.
- Target: `future_gross_sales`, excluding returns/cancellations, not net profit/CLV.
  Predictions are clipped to zero before evaluation and simulation.
- `response_rate` is assumed proportional pre-discount sales-volume growth,
  not response probability. Fixed unit COGS gives NET margin
  `v*((1+r)*(g-d)-g) - contact_cost`. Every promotion effect is simulated/non-causal.
- Policies share K=min(floor(contact budget/contact cost), final-test population).
  Monte Carlo non-targeted campaigns and RFM contact K customers; proposed selects positive NET
  contributions up to K and may abstain. Under the unchanged three scenarios,
  proposed contacts zero (£0 spending/NET change), while Random/RFM use K and
  incur simulated losses when K>0. **No Promotion = £0** is the explicit decision
  reference, not a predictive model.

### Verified real-data baseline and business results

Source: `models/temporal_registry.json`, `outputs/data_quality.csv`, and
`outputs/scenario_table.csv`. Raw rows: **1,067,371**; exact duplicate rows beyond
first: **34,335**, retained under the documented policy.

| Final-test model | PR-AUC (average precision) | ROC-AUC | MAE | RMSE |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.750184 | 0.789977 | — | — |
| Linear Regression, clipped | — | — | £395.69 | £2,016.14 |
| Naive | 0.433630 | — | £636.34 | £3,288.87 |

Phase 3 validation selected both GBM challengers under predeclared thresholds.
On final test, GBM classification reached PR-AUC **0.769056** and ROC-AUC **0.798659**;
GBM regression reached MAE **£400.40** and RMSE **£2,167.29**, worse than the final
Linear Regression despite its validation MAE gain. Sigmoid calibration was not
retained because validation Brier improved by only 0.000096 versus the required
0.002. Five secondary origins show mean baseline/GBM PR-AUC 0.7179/0.7242 and mean
MAE £517.79/£514.06; regression leadership is not consistent. These findings do
not change the no-campaign recommendation.

Duplicate-policy sensitivity preserves the primary `retain` policy and separately
removes 34,335 exact duplicate rows. The final population/repeat rate remain
5,281/43.36%; PR-AUC is 0.750191, ROC-AUC 0.789971, MAE £396.54 and RMSE £2,066.55
under removal. Proposed still selects zero in all three scenarios, so the main
recommendation is qualitatively unchanged. Exact duplicates are not asserted to be errors.

Non-targeted now reports **1,000 uniform K-customer campaigns, seed 42**, rather
than the former single seeded draw. Each draw selects 1,500 distinct customers.
The interval is the 2.5th–97.5th percentile of random-policy outcomes conditional
on forecasts/assumptions, **not a causal confidence interval**.

| Scenario | Non-targeted MC mean NET | Random-policy interval | RFM NET | Proposed / No Promotion |
|---|---:|---|---:|---:|
| Conservative | -£17,417.59 | [-£21,707.94, -£14,023.76] | -£55,333.45 | £0 / £0 |
| Moderate | -£37,407.57 | [-£46,583.42, -£30,081.21] | -£118,314.35 | £0 / £0 |
| Aggressive | -£90,677.15 | [-£112,076.51, -£73,360.59] | -£277,865.97 | £0 / £0 |

Random/RFM spend £750 each; Proposed and No Promotion contact zero and spend £0.
All totals are simulated NET changes, not measured promotion effects.

Decision gate: among positive-value At-risk customers, median required base growth
at 10% discount is **26.75%** (versus the assumed 20%); median maximum supportable
discount at 20% base growth is **7.91%**. On the exploratory grid, positive campaigns
first appear at 5% discount/20% growth (£7,175.92), 10%/30% (£1,424.27), and
15%/50% (£962.39); none at 20% discount through 50% growth. These are tested grid
points, not an empirically established continuous frontier. Main assumptions remain unchanged.

### Expected corrected outputs

- `data/processed/customer_snapshots.csv`; final `customer_features.csv` / `customer_labels.csv`.
- `data/processed/customer_test_predictions.csv` (baseline) and optional
  `customer_test_predictions_advanced.csv`: **final temporal test only**.
- `data/processed/customer_test_scores.csv`: final-test recommendations, NET margin,
  decision date, scenario, contact cost and rank; not a current deployment list.
- `models/temporal_registry.json`: validation/final-test model metrics.
- `outputs/data_quality.csv`, `temporal_schedule.csv`, `temporal_calibration_logreg.csv`.
- `outputs/scenario_table.csv`: three scenarios × No Promotion/non-targeted MC/RFM/proposed,
  actual contacts, contact costs, before-contact margin, NET margin and policy differences.
- `outputs/tornado_sensitivity.csv` and figure: budget-constrained NET sensitivity.
- `outputs/break_even_by_segment.csv`, `break_even_summary.csv`: required-growth
  and supportable-discount quartiles; invalid/zero-value cases are excluded from
  quantiles but retained in eligibility denominators.
- `outputs/response_discount_surface.csv`, `outputs/figures/response_discount_surface.png`:
  4 discounts × 5 growth assumptions, eligibility, selection and NET margin vs No Promotion.
- `outputs/run_manifest.json`: QA-generated timestamp, Git commit/dirty flag, Python,
  raw/config/requirements SHA-256 hashes, temporal design/source, and row/customer counts.
- Phase 3 step 04: `outputs/temporal_model_comparison.csv`, `model_selection.csv`,
  `calibration_metrics.csv`, `calibration_table.csv`, `advanced_feature_importance.csv`,
  `rolling_origin_metrics.csv`, three compact figures, and `models/gbm_*.joblib`.
- Phase 4 duplicate sensitivity: `outputs/duplicate_robustness_summary.csv` and
  `duplicate_robustness_business_metrics.csv`; primary outputs continue to retain duplicates.
- `docs/executive_brief.pdf`: submission-ready brief rendered from the authoritative
  Markdown with `python scripts/render_executive_brief.py`; the command verifies the
  PDF page count does not exceed two.

Run 02/03/05 after changing the analytical configuration. Legacy generated files
are not corrected results merely because they exist. Run final QA to regenerate and
hash-validate the manifest. Unchanged raw/config bytes reproduce the same hashes;
timestamps and Git state may differ. Synthetic tests are not substitutes for UCI metrics.

`outputs/run_manifest.json` records the analytical run that generated the reported
artifacts. The final submission may be a later clean Git revision containing those
validated artifacts plus packaging or documentation updates; the manifest's recorded
run commit and dirty state therefore do not describe the release checkout state.

## Dashboard

```bash
streamlit run dashboard/app.py
```

Dashboard tương tác: 5 KPI quyết định, filter theo quốc gia/segment/
recommendation, so sánh scenario, cohort retention heatmap, và actionable
view (final-test evaluation list, downloadable CSV). It requires corrected outputs,
works with baseline alone, and displays simulated NET margin after contact cost.
The Scenario Sandbox changes discount, assumed sales-volume growth, gross margin
and contact budget in memory without retraining. Its five KPIs and downloadable
target list apply to the explicitly labeled filtered population. Global saved
comparisons and the cohort view remain unchanged. Inputs are exploratory and non-causal.
The comparison uses equal maximum capacity K, not equal actual contact counts;
Random/RFM use K even when proposed recommends no campaign.

## Serving API (mô phỏng tích hợp production)

Optional GBM demo: run step 04 to create model binaries first. API scores are
explicitly `deployment_demo_not_evaluation` and never feed the policy evaluation.
The response contains `simulated_net_incremental_margin`; campaign-wide capacity
is handled by batch step 05, not individual API calls.

```bash
python -m uvicorn src.serve:app --reload --port 8000
# hoặc: docker build -t capstone-api . && docker run -p 8000:8000 capstone-api
```

```bash
curl -X POST http://localhost:8000/score -H "Content-Type: application/json" \
  -d '{"recency_days": 30, "frequency": 5, "monetary": 500, "avg_order_value": 100, "n_distinct_products": 10, "tenure_days": 200, "return_rate": 0.1, "rfm_segment": "Loyal"}'
```

## Test

```bash
python -m pytest tests/   # window/temporal/provenance/economics/idempotency tests
python --version          # Python 3.11
pip check                 # verify installed dependency compatibility
git diff --check
```

CI (`.github/workflows/ci.yml`) tự động chạy test này mỗi lần push.

## Cấu trúc project

`src/`: pipeline/API; `config/`: assumptions; `tests/`: regression tests;
`docs/`: current methodology; `dashboard/`: final-test view; `data/`: inputs/scores;
`outputs/`: diagnostics/scenarios; `models/`: model metrics/binaries.

## Tài liệu

- `docs/problem_statement.md` — business decision, unit of analysis, target, prediction horizon
- `docs/data_dictionary.md` — nguồn dữ liệu, filter, cleaning decisions
- `docs/model_card.md` — target, features, validation, kết quả model, assumptions, risks, ethics/bias
- `docs/executive_brief.md` — khuyến nghị cho decision-maker (≤2 trang)
- `docs/traceability_map.md` — trace recommendation → data/assumption/validation
- `docs/production_roadmap.md` — những gì cần thêm để đưa vào vận hành thật dài hạn
