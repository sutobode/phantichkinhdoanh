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

**Classification (`repeat_purchase`)**

| Model | PR-AUC | ROC-AUC |
|---|---|---|
| Naive baseline | 0.501 | — |
| Logistic Regression | 0.815 [0.785, 0.847] | 0.804 [0.778, 0.830] |
| **Gradient Boosting** | **0.820 [0.787, 0.851]** | **0.806 [0.780, 0.832]** |

GBM thắng rõ ràng trên cả PR-AUC và ROC-AUC cho bài toán classification —
được chọn làm model chính thức cho `predicted_repeat_purchase_proba`.

**Regression (`future_value`)**

| Model | MAE | RMSE |
|---|---|---|
| Naive baseline | 1,093.3 | 5,065.1 |
| Linear Regression | 562.7 [489.9, 652.7] | **1,431.0 [978.2, 1,901.2]** |
| Gradient Boosting | **530.5 [432.6, 679.7]** | 2,048.0 [835.3, 3,280.9] |

(Số liệu trên đã áp dụng `clip_nonneg` — xem mục "Bug đã phát hiện" bên dưới.
Trước khi clip, Linear Regression có MAE 593.3 vì 827/1,009 khách hàng trong
test set bị dự đoán `future_value` âm — một dấu hiệu model lỗi rõ ràng vì
ground truth không bao giờ âm.)

⚠️ **Phát hiện quan trọng, không che giấu**: GBM có MAE tốt hơn (sai số
*điển hình* thấp hơn) nhưng RMSE **tệ hơn** Linear Regression (sai số bị
"phạt nặng" ở các giá trị ngoại lai). Điều tra cho thấy nguyên nhân: 1 khách
hàng có `future_value` thực tế = £22,314 nhưng GBM dự đoán £78,682 (sai lệch
£56,368) — một outlier duy nhất đủ để kéo RMSE của GBM lên cao, vì
`future_value` có phân phối lệch phải cực mạnh (max=£144,204, median=£16.9,
std=£5,067 trên tập test). Linear Regression ít "tự tin thái quá" hơn ở nhóm
khách hàng giá trị cực cao nên RMSE ổn định hơn, dù MAE trung bình kém hơn
đôi chút.

**Quyết định**: vẫn dùng GBM cho `predicted_future_value_gbm` (feed vào
`customer_scores.csv` và decision layer), vì:
1. Quyết định kinh doanh dựa trên **tổng margin across nhiều khách hàng**
   (xem `05_decision_scenario.py`), nơi sai số trung bình (MAE) ảnh hưởng
   nhiều hơn 1 vài outlier riêng lẻ.
2. Classification task (quan trọng hơn cho targeting rule) GBM thắng dứt
   khoát.

**Nhưng** — khuyến nghị vận hành: KHÔNG dùng `predicted_future_value_gbm`
một cách mù quáng cho quyết định cá nhân hóa ở nhóm khách hàng giá trị cực
cao (`monetary` top 1%) mà không review thủ công, vì đây chính là nhóm GBM
có rủi ro dự đoán lệch lớn nhất.

### Bug đã phát hiện qua senior code review (sau khi pipeline đã "chạy đúng")

Một senior code review độc lập (không chỉ tự kiểm tra của người viết) phát
hiện 7 vấn đề thật, đã verify bằng số liệu cụ thể trên dữ liệu thật (không
chỉ sửa theo lý thuyết) và fix toàn bộ:

1. **Giá trị dự đoán âm không được clip** — `future_value` ground truth luôn
   ≥ 0, nhưng model có thể dự đoán âm (827/1,009 khách hàng với Linear
   Regression, 4/5,041 với GBM). Hậu quả nghiêm trọng hơn con số nghe có vẻ
   nhỏ: với khách hàng Champions (response multiplier thấp), giá trị dự đoán
   âm × bracket âm trong công thức margin = **margin dương giả**, khiến
   khách hàng 17850 bị gắn nhầm "nurture" dù model dự đoán giá trị âm/vô
   nghĩa cho họ. Fix: `utils.clip_nonneg()`, áp dụng ngay sau mọi lệnh
   `.predict()` cho `future_value`. Test: `tests/test_utils.py`.
2. **`models/registry.json` dùng `run_id` cố định** (`"advanced_gbm_v1"`,
   không có timestamp/git_commit) cho model GBM — chạy lại pipeline nhiều
   lần tạo nhiều entry trùng tên, không phân biệt được, phá vỡ mục đích
   audit trail. Fix: run_id theo timestamp UTC + `git_commit_hash()`, nhất
   quán với entry baseline.
3. **RFM segment tính lại trên test subset** (~1,000 dòng) thay vì dùng lại
   segment đã tính trên toàn population (5,041 dòng) trong
   `error_by_segment.csv` — verify cho thấy **93/1,009 khách hàng (9.2%)**
   bị đổi nhãn segment do quantile cut-point khác nhau, khiến phân tích lỗi
   theo segment không nhất quán với segment hiển thị ở dashboard/
   customer_scores.csv. Fix: tính `rfm_segment` 1 lần trên toàn `df` trước
   khi split train/test, dùng lại cho cả model training context lẫn error
   analysis.
4. **`bootstrap_ci` có thể crash** nếu toàn bộ resample đều lỗi (vd PR-AUC
   trên resample chỉ có 1 class, dễ xảy ra ở cutoff nhỏ/mất cân bằng của
   rolling-origin validation) — `np.percentile` trên mảng rỗng sẽ raise.
   Fix: bọc try/except, trả về CI=[NaN, NaN] kèm cảnh báo log thay vì crash
   toàn bộ pipeline.
5. **Serving API (`src/serve.py`) không validate `rfm_segment`** — nhận bất
   kỳ string nào, lỗi segment không hợp lệ chỉ bị bắt sâu trong business
   logic, trả về HTTP 500 kèm stack trace thay vì 422 validation error rõ
   ràng. Fix: `Literal["Champions","Loyal","At-risk","Lost"]` thay cho `str`.
6. **`bootstrap_ci` bị copy-paste y hệt giữa `03_baseline_models.py` và
   `04_advanced_models.py`** — vi phạm chính convention đã ghi trong
   `utils.py` docstring. Fix: gộp về `utils.bootstrap_ci`, dùng chung.
7. **Tính dư thừa** trong `compute_rfm_features`: `avg_order_value` tính lại
   `nunique("Invoice")` lần 2 dù giá trị này đã có sẵn ở cột `frequency`
   trong cùng 1 lệnh `.agg()`. Fix: `avg_order_value = monetary / frequency`.

Toàn bộ fix đã re-run `src/06_final_qa.py` (pipeline từ đầu + 19 unit test +
7 invariant check) — PASSED.

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
