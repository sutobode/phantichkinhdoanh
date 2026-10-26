# Customer Value & Promotion Targeting for an Online Retailer

Capstone project — HUST Business Analytics 2026. Chi tiết yêu cầu: `phantichkinhdoanh.pdf`.
`PLAN.md` is the archived pre-remediation work log, not current validation evidence.

**Result status:** corrected UCI metrics are pending a raw-data rerun. Existing
`customer_scores.csv`, `models/registry.json`, and old generated figures/tables
are superseded; see `outputs/RESULTS_STATUS.md`. The corrected pipeline and dashboard
never fall back to the old mixed-sample scores.

## Business question

Khách hàng nào nên được ưu tiên cho retention/promotion để tăng expected future
margin mà không over-discount?

## Cài đặt

Python 3.11; use the pinned dependencies below. `curl` and `unzip` are required
for the download example (or extract the archive with your OS archive tool).

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

## Dữ liệu

Dataset: [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii).
Xem chi tiết nguồn/ngày tải/filter trong `docs/data_dictionary.md`.

Tải và giải nén:

```bash
curl -L -o data/raw/online_retail_ii.zip https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip
unzip -o data/raw/online_retail_ii.zip -d data/raw/
```

## Chạy pipeline (theo thứ tự)

```bash
python src/01_eda.py                  # optional source EDA
python src/02_cleaning_features.py    # duplicate audit + temporal snapshots + cohort
python src/03_baseline_models.py      # RFM + logistic/linear temporal evaluation
python src/05_decision_scenario.py    # baseline-only equal-capacity NET scenarios
```

Default `decision.value_prediction_source: baseline` needs no GBM. Optional comparison:

```bash
python src/04_advanced_models.py      # optional GBM + calibration/SHAP/subgroup errors
```

For an advanced-source scenario run, preselect `value_prediction_source: advanced`
in config and run 04 before 05. Do not choose source based on final-test performance.
Both source files use the same prediction-column meanings; baseline remains separate.

```bash
python src/06_final_qa.py             # reruns 02→03→05 + tests/invariants;
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
  Seeded random and RFM contact K customers; proposed selects positive NET
  contributions up to K and may abstain. Under the unchanged three scenarios,
  proposed contacts zero (£0 spending/NET change), while Random/RFM use K and
  incur simulated losses when K>0. Exact baseline totals need the UCI rerun.

### Expected corrected outputs

- `data/processed/customer_snapshots.csv`; final `customer_features.csv` / `customer_labels.csv`.
- `data/processed/customer_test_predictions.csv` (baseline) and optional
  `customer_test_predictions_advanced.csv`: **final temporal test only**.
- `data/processed/customer_test_scores.csv`: final-test recommendations, NET margin,
  decision date, scenario, contact cost and rank; not a current deployment list.
- `models/temporal_registry.json`: validation/final-test model metrics.
- `outputs/data_quality.csv`, `temporal_schedule.csv`, `temporal_calibration_logreg.csv`.
- `outputs/scenario_table.csv`: three scenarios × random/RFM/proposed policies,
  actual contacts, contact costs, before-contact margin, NET margin and policy differences.
- `outputs/tornado_sensitivity.csv` and figure: budget-constrained NET sensitivity.
- Optional step 04: `outputs/temporal_model_comparison.csv`, `temporal_*` diagnostics,
  `outputs/figures/temporal_calibration_curve.png`, and `models/gbm_*.joblib`.

Run 02/03/05 after changing the analytical configuration. Legacy generated files
are not corrected results merely because they exist. Missing workbook/dependencies
block a full rerun; synthetic tests are not substitutes for UCI metrics.

## Dashboard

```bash
streamlit run dashboard/app.py
```

Dashboard tương tác: 5 KPI quyết định, filter theo quốc gia/segment/
recommendation, so sánh scenario, cohort retention heatmap, và actionable
view (final-test evaluation list, downloadable CSV). It requires corrected outputs,
works with baseline alone, and displays simulated NET margin after contact cost.
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
