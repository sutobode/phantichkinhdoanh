# Problem Statement

## Business decision

Khách hàng nào nên được ưu tiên cho retention/promotion để tăng expected
future margin của nhà bán lẻ online, **mà không over-discount** (không giảm
giá lãng phí cho khách hàng vốn đã mua đều đặn)?

## Unit of analysis

1 khách hàng (`Customer ID`), giới hạn ở những khách hàng có ít nhất 1 giao
dịch hợp lệ trong observation window.

## Target / outcome

- **Target 1 (classification)**: `repeat_purchase` — khách hàng có mua lại
  trong outcome window hay không (0/1).
- **Target 2 (regression)**: `future_value` — tổng doanh thu khách hàng tạo
  ra trong outcome window.

## Prediction horizon

- Observation window: 2009-12-01 → 2011-06-30 (19 tháng) — dùng để tính mọi
  feature.
- Outcome window: 2011-07-01 → 2011-12-09 (~5.3 tháng) — dùng để tạo nhãn.
- Lý do chọn mốc này: xem `outputs/logs/01_eda.log` và
  `outputs/monthly_transaction_profile.csv` — dataset có 25 tháng dữ liệu,
  tỷ lệ repeat-purchase ổn định ở mốc 180 ngày (58.1%), nên outcome window
  ~5.3 tháng đủ để có label có ý nghĩa mà vẫn giữ đủ lịch sử cho observation
  window.

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
