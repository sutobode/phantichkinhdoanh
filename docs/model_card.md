# Model / Analysis Card

## Business decision

Khách hàng nào nên được ưu tiên retention/promotion để tăng expected future
margin **mà không over-discount**? Output cuối cùng là `final_recommendation`
(retain / nurture / no-offer) cho từng khách hàng, dùng để quyết định ai nhận
khuyến mãi.

## Unit of analysis, target, prediction horizon

- **Unit of analysis**: 1 khách hàng (`Customer ID`), có giao dịch hợp lệ
  trong observation window.
- **Target 1 (classification)**: `repeat_purchase` — khách hàng có mua lại
  trong outcome window hay không (binary).
- **Target 2 (regression)**: `future_value` — tổng doanh thu khách hàng tạo ra
  trong outcome window (bao gồm 0 nếu không mua).
- **Prediction horizon**: observation window = 2009-12-01 → 2011-06-30 (19
  tháng, dùng để tính feature); outcome window = 2011-07-01 → 2011-12-09
  (~5.3 tháng, dùng để tạo nhãn).

## Features

7 feature, tất cả tính **chỉ từ observation window** (xem
`docs/data_dictionary.md` để biết công thức chi tiết):
`recency_days, frequency, monetary, avg_order_value, n_distinct_products,
tenure_days, return_rate`.

## Models & kết quả (chạy thật trên Online Retail II, seed=42)

| Model | Target | Metric | Giá trị (95% bootstrap CI) |
|---|---|---|---|
| Naive baseline | repeat_purchase | PR-AUC | 0.502 |
| Naive baseline | future_value | MAE | 1,115.6 |
| Logistic Regression | repeat_purchase | PR-AUC | 0.815 [0.785, 0.847] |
| Logistic Regression | repeat_purchase | ROC-AUC | 0.804 [0.778, 0.830] |
| Linear Regression | future_value | MAE | 582.2 [487.9, 696.1] |
| **Gradient Boosting** | repeat_purchase | PR-AUC | **0.820 [0.787, 0.851]** |
| **Gradient Boosting** | repeat_purchase | ROC-AUC | **0.806 [0.780, 0.832]** |
| **Gradient Boosting** | future_value | MAE | **531.2 [432.8, 679.7]** |

GBM được chọn làm model chính thức (nhỉnh hơn baseline, dùng cho
`customer_scores.csv` và toàn bộ decision layer).

### Rolling-origin validation (ổn định qua thời gian)

Lặp lại train/eval tại 3 cutoff khác nhau (2011-01-31, 2011-03-31,
2011-06-30), mỗi cutoff build lại dataset từ đầu, outcome window 90 ngày:

| Model | PR-AUC mean | PR-AUC std |
|---|---|---|
| Logistic Regression | 0.707 | 0.021 |
| Gradient Boosting | 0.714 | 0.035 |

Std thấp (~0.02-0.035) cho thấy model ổn định qua các thời điểm khác nhau,
không phải kết quả "may mắn" ở 1 lần chia.

### SHAP feature importance (GBM classifier)

`recency_days` > `monetary` > `frequency` > `n_distinct_products` >
`tenure_days` > `avg_order_value` > `return_rate`. Recency là yếu tố dự báo
mạnh nhất cho repeat-purchase — khớp trực giác RFM.

## Validation design

- **Temporal holdout** ở tầng nhãn: feature chỉ dùng observation window,
  label chỉ dùng outcome window (sau observation_end) — không leak.
- **Train/test split** để đánh giá model: random split GIỮA CÁC KHÁCH HÀNG
  (không phải random split theo thời gian giao dịch) — vì mỗi khách hàng đã
  có feature/label được giới hạn thời gian đúng đắn từ trước, random split
  cross-sectional này không gây leakage (xem comment trong
  `src/03_baseline_models.py`).
- **Rolling-origin validation** bổ sung (3 cutoff) để kiểm tra ổn định qua
  thời gian — đúng tinh thần "validation design phải khớp với
  data-generating process".
- Mọi feature/label pipeline đã được unit test cho leakage (xem
  `tests/test_features.py`, 6 test) và cho logic quyết định (xem
  `tests/test_decision_scenario.py`, 7 test).

## Assumptions (tường minh)

1. `gross_margin_pct = 0.40` — giả định biên lợi nhuận gộp, KHÔNG có trong
   dataset, tham khảo mức phổ biến ngành e-commerce bán lẻ.
2. `segment_response_multiplier` — hệ số phản hồi khuyến mãi theo RFM
   segment, dựa trên lý thuyết uplift modeling (Sure Things / Persuadables /
   Lost Causes), **KHÔNG ước lượng từ dữ liệu** vì dataset không có
   treatment/control thật.
3. `discount_response_scenarios` (3 kịch bản: conservative/moderate/
   aggressive) — giả định kinh doanh, có sensitivity analysis đi kèm (xem
   `outputs/tornado_sensitivity.csv`).
4. `contact_cost_per_customer`, `budget_total` — giả định chi phí liên hệ cho
   budget-constrained optimization.

## Risks & limitations

- **Không có causal identification**: dataset không có randomized
  promotion treatment/control (Scope Guardrail của đề). Mọi con số
  "incremental margin" là **scenario-based estimate**, KHÔNG phải causal
  uplift đã được kiểm chứng. Xem `docs/production_roadmap.md` — "Path to
  causal validation" để biết cách kiểm chứng bằng A/B test thật trước khi
  scale.
- **Outcome window ngắn** (~5.3 tháng) so với vòng đời khách hàng thực tế —
  `future_value` có thể underestimate giá trị dài hạn của khách hàng mới.
- **Imbalance dữ liệu theo quốc gia**: 94% giao dịch từ UK. Model ít được
  "học" từ khách hàng ngoài UK.
- **RFM segmentation dùng quantile trên toàn bộ population** (bao gồm cả
  test set) — không ảnh hưởng đến các metric dự báo đã báo cáo (RFM segment
  không phải input của model), nhưng cần lưu ý nếu dùng segment này để suy
  luận thống kê khác.

## Ethics / Bias check (chạy thật, không chỉ liệt kê rủi ro)

Kiểm tra mean absolute error của GBM classifier trên test set, nhóm theo
quốc gia (quốc gia có <20 khách hàng trong test set được gộp vào "Other" để
tránh kết luận sai do mẫu nhỏ):

| country_group | n | mean_abs_error | error vs overall |
|---|---|---|---|
| United Kingdom | 939 | 0.353 | +0.6% |
| Other (small country, n<20) | 70 | 0.322 | -8.4% |

Không phát hiện nhóm nào lệch >30% so với sai số trung bình chung → **không
có bằng chứng model bias nghiêm trọng theo quốc gia** ở mức gộp nhóm này.
Lưu ý: do 94% dữ liệu từ UK, các quốc gia nhỏ bị gộp chung nên không thể kết
luận chi tiết cho TỪNG quốc gia riêng lẻ (đây là limitation, không phải kết
luận "không có bias" tuyệt đối). Script: `src/04_advanced_models.py`, output:
`outputs/ethics_bias_check_by_country.csv`.

Đồng thời, recommendation phân bổ rõ theo segment (xem dashboard): Champions
→ chủ yếu "retain" (1459/1460), Lost → chủ yếu "no-offer" (1506/1508),
At-risk/Loyal → chủ yếu "nurture" — phân bổ này phản ánh ĐÚNG giả định thiết
kế (persuadables được ưu tiên), không phải artifact ngẫu nhiên.

## Reproducibility

- Seed cố định (`config/params.yaml: seed: 42`) cho mọi bước có random
  (train/test split, bootstrap, GBM).
- Toàn bộ tham số (window, discount scenarios, budget...) tập trung ở
  `config/params.yaml`, không hardcode trong code.
- 15 unit test (`pytest tests/`) cover: no-leakage (6), decision-logic
  correctness (7), pipeline idempotency (2).
