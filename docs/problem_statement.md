# Problem Statement

## Business decision

Khách hàng nào nên được ưu tiên cho retention/promotion để tăng expected
future margin của nhà bán lẻ online, **mà không over-discount** (không giảm
giá lãng phí cho khách hàng vốn đã mua đều đặn)?

## Unit of analysis

1 customer snapshot (`Customer ID`, `decision_date`), with at least one valid
purchase before that decision date. A customer may recur at earlier origins.

## Target / outcome

- **Target 1 (classification)**: `repeat_purchase` — khách hàng có mua lại
  trong outcome window hay không (0/1).
- **Target 2 (regression)**: `future_gross_sales` — positive, non-cancelled
  sales revenue in the outcome window, zero if none. This is NOT net sales
  after returns, profit, or lifetime CLV.

## Prediction horizon

Fixed **90-day** horizon, expanding observation history from 2009-12-01.
Decision timestamps are midnight; features use transactions strictly before T.

| Role | Decision T | Observation (inclusive dates) | Outcome (inclusive dates) |
|---|---|---|---|
| Train | 2010-12-01 | 2009-12-01–2010-11-30 | 2010-12-01–2011-02-28 |
| Train | 2011-03-01 | 2009-12-01–2011-02-28 | 2011-03-01–2011-05-29 |
| Validation | 2011-06-01 | 2009-12-01–2011-05-31 | 2011-06-01–2011-08-29 |
| Final test | 2011-09-10 | 2009-12-01–2011-09-09 | 2011-09-10–2011-12-08 |

Train origins predict validation. Then refit the predeclared model on train plus
validation snapshots to predict final test. Training labels end before each
evaluation decision: 2011-05-30 exclusive before validation, 2011-08-30 exclusive
before final test. Final-test outcomes are never fitted or used for model choice.
The last recorded day, 2011-12-09, is potentially partial and excluded from labels.
The script checks actual raw coverage and rejects an incomplete configured horizon.
Dates are configurable in `config/params.yaml`; no random customer holdout is used.

## Policy evaluation

Only final-test predictions feed the main campaign simulation and dashboard.
Baseline: RFM + logistic regression + clipped linear future-gross-sales predictions.
GBM is an optional, separately reported comparison; select the policy source before
inspecting final-test results. No current full-population demo scores enter metrics.

Compare 1,000 seeded uniform random campaigns (Monte Carlo mean and random-policy
simulation interval), RFM-ranked, and positive-NET-margin-ranked contacts with
the same maximum capacity K=min(floor(£750/£0.50), evaluation population size).
Random and RFM use K contacts; proposed selects positive NET contributions up to K,
possibly zero. Population, scenario assumptions and per-contact cost are identical.
No Promotion is an explicit £0 decision reference with zero contacts, not a model.
The real final-test population is 5,281 customers (43.36% repeat purchase), so K=1,500.
Contact costs are deducted. With unchanged unit product costs:
`net = max(predicted_gross_sales, 0) * ((1+r)*(g-d)-g) - contact_cost`.
`r` means assumed proportional pre-discount sales-volume growth, not response
probability. Predicted sales describe historical behavior, not a no-promotion
counterfactual. Under all three configured scenarios no positive net contribution
is possible; no-contact is an admissible decision, not a reason to change assumptions.

The baseline real-data rerun gives PR-AUC 0.750184 / ROC-AUC 0.789977 and
future-gross-sales MAE £395.69 / RMSE £2,016.14. Decision extensions calculate
analytical break-even growth/discounts and a 20-cell response × discount grid;
the dashboard explores assumptions without fitting models. Full results and
non-causal policy comparisons are in `docs/model_card.md` and `docs/executive_brief.md`.
The primary analysis retains exact transaction duplicates; a separately named
drop-duplicate sensitivity uses the same dates/models and does not overwrite it.

## Vì sao KHÔNG dùng causal uplift

Dataset UCI Online Retail II là dữ liệu giao dịch lịch sử, **không có
randomized treatment/control** cho khuyến mãi. Do đó mọi phân tích impact của
khuyến mãi trong project này là **scenario-based estimate** dựa trên giả định
tường minh (discount %, response rate), không phải causal effect đã được đo
lường — tuân thủ đúng Scope Guardrail của đề bài.

## Deliverables (xem PLAN.md để biết chi tiết từng tuần)

1. Reproducible pipeline: `src/01_eda.py` → `src/06_final_qa.py`
2. `docs/data_dictionary.md`
3. `docs/executive_brief.md`
4. `dashboard/app.py` (Streamlit)
5. `docs/model_card.md`
