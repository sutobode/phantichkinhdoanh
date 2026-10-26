# Data Dictionary & Source Log

## Source log

- **Nguồn:** UCI Machine Learning Repository — Online Retail II
  (https://archive.ics.uci.edu/dataset/502/online+retail+ii)
- **URL tải trực tiếp:** https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip
- **Ngày tải:** 2026-10-26
- **File gốc:** `online_retail_II.xlsx`, 2 sheet: "Year 2009-2010", "Year 2010-2011"
- **Phạm vi dùng:** both sheets; expanding snapshots with 90-day outcomes.
  Exact dates: `docs/problem_statement.md` / `config/params.yaml`.
  Final outcome ends 2011-12-09 exclusive; the potentially partial last day is excluded.

## Raw columns

| Column | Type | Mô tả |
|---|---|---|
| Invoice | string | Mã hóa đơn. Bắt đầu bằng "C" = cancellation. |
| StockCode | string | Mã sản phẩm. |
| Description | string | Tên sản phẩm. |
| Quantity | int | Số lượng. Có thể âm (return/cancellation). |
| InvoiceDate | datetime | Thời điểm giao dịch. |
| Price | float | Đơn giá (GBP). |
| Customer ID | float (nullable) | Mã khách hàng, có nhiều dòng null. |
| Country | string | Quốc gia khách hàng. |

## Filters & cleaning decisions

⚠️ **Thứ tự xử lý quan trọng để chống leakage**: split theo thời gian (observation
vs outcome window) được thực hiện **trước**, sau đó mới loại cancellation/qty/price
**riêng biệt trong từng window**. Nếu clean toàn cục rồi mới split, feature
`return_rate` sẽ không tính đúng theo window. Xem `src/02_cleaning_features.py`
và `tests/test_features.py` (có test leakage riêng cho `return_rate` và cho
boundary ngày cuối mỗi window).

The real-data rerun contains **1,067,371 raw rows** and **34,335 exact duplicate
rows beyond first**, retained (`outputs/data_quality.csv`). The final temporal-test
population is **5,281 customers**, with **43.36%** repeat purchase. Former
5,041-customer/window counts are superseded, not current evidence.

1. Audit exact duplicates on the concatenated original rows, before type coercion:
   `raw.duplicated().sum()` counts repeats beyond the first occurrence.
   **Retain duplicates**: the source does not establish that identical lines are
   errors. Write count and policy to `outputs/data_quality.csv` and the step-02 log.
   The observed duplicate count is 34,335; it is not assumed to be zero.
2. Drop missing customer IDs globally (time-independent).
3. At each snapshot, split time windows before computing cancellation rate.
4. Separately remove C-prefixed invoices, Quantity <=0, and Price <=0 in each window.
5. Compute Revenue = Quantity × Price. Per-window removals are logged.

Returns are not matched to original purchases or netted from revenue. The target
is renamed `future_gross_sales` to explicitly mean **gross positive future sales**.
`return_rate` remains the cancellation-invoice share, not a net-return monetary rate.

## Joins

Không join thêm dataset ngoài (đúng Scope Guardrail: không merge dataset không
liên quan chỉ để tăng feature count).

Internal feature/label joins use Customer ID with one-to-one validation. Snapshot
keys are (Customer ID, decision_date); duplicate keys are rejected. Cohort joins
attach the first observed purchase month, not proven lifetime acquisition.

## Derived tables

- `data/processed/customer_snapshots.csv` — all four snapshots, features/labels,
  `decision_date`, `outcome_end_exclusive`, and `split` (train/validation/final_test).
- `data/processed/customer_features.csv`, `customer_labels.csv` — final-test
  snapshot feature/label exports; label is `future_gross_sales` plus `repeat_purchase`.
- `data/processed/customer_test_predictions.csv` — baseline final-test predictions only.
- `data/processed/customer_test_predictions_advanced.csv` — optional GBM equivalent.
- `data/processed/customer_test_scores.csv` — **current final-test customer-level
  evaluation deliverable**, 5,281 unique customers; explicitly allowed by `.gitignore`.
- `data/processed/customer_scores.csv` — **legacy pre-remediation artifact**, never
  consumed for corrected evaluation. See `outputs/RESULTS_STATUS.md`.

### `customer_features.csv` — mô tả cột

| Cột | Công thức | Window nguồn |
|---|---|---|
| recency_days | (as_of − ngày mua gần nhất) | observation |
| frequency | số invoice duy nhất (đã loại cancellation) | observation |
| monetary | tổng Revenue (Quantity × Price) | observation |
| avg_order_value | monetary / frequency | observation |
| n_distinct_products | số StockCode duy nhất | observation |
| tenure_days | (as_of − ngày mua đầu tiên) | observation |
| return_rate | số invoice cancel / tổng invoice (kể cả cancel) | observation |
| country | quốc gia (lấy giá trị đầu) | observation |

`as_of` = `decision_date`; every feature uses timestamps strictly before it.

### Prediction / policy columns

- `rfm_score`: quartile R+F+M sum (3–12), calculated from current snapshot history.
- `rfm_segment`: Champions >=10, Loyal >=8, At-risk >=6, otherwise Lost.
- `predicted_repeat_purchase_proba`: selected file's classifier probability.
- `predicted_future_value_raw`: unconstrained regressor output for diagnostics.
- `predicted_future_value`: max(raw prediction, 0), predicting `future_gross_sales`.
- `prediction_source`: baseline or advanced; both use the same field names in separate files.
- `training_outcome_end_exclusive`: latest label end used to fit the final model;
  must precede the test decision. `evaluation_design` identifies temporal snapshots.
- `baseline_future_revenue`: nonnegative simulator input, predicted sales under
  historical behavior (not an identified untreated counterfactual).
- `promotion_scenario`, `effective_response_rate_reference`: scenario name and
  assumed proportional pre-discount sales-volume growth after segment multiplier.
- `simulated_gross_incremental_margin`: product margin change before contact cost.
- `simulated_net_incremental_margin`: gross change minus contact cost if contacted.
- `expected_contact_cost_if_targeted`: £0.50; only selected contacts incur it.
- `final_recommendation`: nurture for positive net contribution and positive value;
  retain for positive-value Champions/Loyal with nonpositive net; otherwise no-offer.
- `target_rank`: descending positive-net-margin rank; blank when ineligible.
- `targeted_under_budget_reference_scenario`: actual proposed-policy selection.

`outputs/scenario_table.csv` has one row per scenario/policy. Policies share the
final-test population, maximum capacity K=min(floor(budget/contact cost), population),
scenario assumptions and per-contact cost. Random/RFM use K contacts; proposed uses
only positive NET contributions up to K, possibly zero. Rows report actual contacts,
actual contact cost, before-contact margin, NET margin, NET margin/contact (0 by
convention for zero contacts), and simulated difference versus the Monte Carlo
non-targeted mean. `no_promotion` explicitly has zero contacts, cost and margin.
`simulated_margin_vs_no_promotion` equals simulated NET margin because that reference
is zero. These are simulated, not observed effects.

### Decision-analysis and provenance outputs

- `random_mean_net_margin`, `random_p025_net_margin`, `random_p975_net_margin`:
  Monte Carlo summary for non-targeted rows; blank for deterministic policies.
  `random_policy_simulations`=1,000 and `random_policy_sample_size`=K=1,500.
  `interval_type` identifies a random-policy simulation interval, not causal CI.
- `outputs/break_even_by_segment.csv` / `break_even_summary.csv`: scenario/segment
  customer counts, valid positive-value count, eligibility share, p25/median/p75 of
  required base growth and maximum discount. Rates are fractions, not percentages.
  Undefined cases (`v<=0`, `g<=d` for growth, zero multiplier) return NaN, excluded
  from quantiles; eligibility shares retain the full group denominator. A negative
  supportable discount means no nonnegative discount breaks even at that growth.
- `outputs/response_discount_surface.csv`: 20 exploratory cells with discounts
  0.05/0.10/0.15/0.20 and base growth 0.10/0.20/0.30/0.40/0.50. Each records eligible
  and targeted counts, cost, NET margin and difference vs No Promotion. All use the
  final-test population and `analysis_type=scenario_based_non_causal`.
- `outputs/run_manifest.json`: generated/validated by final QA from actual raw,
  configuration and requirements file bytes (SHA-256), plus run timestamp, Git
  commit/dirty flag, Python/platform, test date, horizon, source and counts.
