# Temporal Design & RFM Diagnostics (Appendix)

Mọi con số được tính lại từ code và dữ liệu bởi `src/11_assumption_robustness.py`
(→ `outputs/assumption_robustness/temporal_counts.csv`, `rfm_*.csv`, `recency_check.csv`, `at_risk_rfm_patterns.csv`).
Không thay đổi thiết kế hay nhãn nào.

## 1. Thiết kế chính (đã xác minh)

Đơn vị là **snapshot khách hàng–ngày quyết định** (một khách có thể xuất hiện ở nhiều ngày), không phải khách duy nhất.

| Vai trò | Ngày quyết định T | Outcome [T, T+90) | Snapshot |
|---|---|---|---|
| Train 1 | 2010-12-01 | → 2011-03-01 | 4.266 |
| Train 2 | 2011-03-01 | → 2011-05-30 | 4.537 |
| Validation | 2011-06-01 | → 2011-08-30 | 4.933 |
| Final test | 2011-09-10 | → 2011-12-09 | 5.281 |

- Fit cho validation = train 1 + train 2 = **8.803**; fit cuối = train + validation = **13.736**.
- Ranh giới: feature dùng giao dịch `t < T`. `build_snapshot` đặt `observation_end = T − 1 ngày`, rồi `split_windows` cộng 1 ngày và so sánh `<`. Outcome dùng `[T, T+90)`.
  Guard trong `snapshot_schedule` và `evaluate_temporal_models` (`src/02_cleaning_features.py`, `src/03_baseline_models.py`) từ chối nếu outcome của dữ liệu fit kết thúc `>=` ngày đánh giá.
- `recency_days = T − ngày mua hợp lệ gần nhất (< T)`: khớp **5.281/5.281** khách final test (`recency_check.csv`).

## 2. Rolling-origin (phụ)

`config/params.yaml › advanced_modeling.rolling_origin_dates` có **6 ngày**: 2010-06-01, 2010-09-01, 2010-12-01, 2011-03-01, 2011-06-01, 2011-09-10.
Ngày đầu chỉ dùng để train, nên có **5 origin đánh giá**. Tiêu chí train (`eligible_rolling_training`, `src/04_advanced_models.py`):
`decision_date < origin` **và** `outcome_end_exclusive < origin` (**`<` nghiêm ngặt**).

| Origin | Snapshot train được dùng | n_train | n_eval | Nếu dùng `<=` |
|---|---|---|---|---|
| 2010-09-01 | 06-01 | 2.703 | 3.321 | 2.703 |
| 2010-12-01 | 06-01, 09-01 | 6.024 | 4.266 | 6.024 |
| 2011-03-01 | 06-01, 09-01 | **6.024** | 4.537 | **10.290** |
| 2011-06-01 | 06-01, 09-01, 12-01, 03-01 | 14.827 | 4.933 | 14.827 |
| 2011-09-10 | cả 5 ngày trước | 19.760 | 5.281 | 19.760 |

- **Vì sao 2010-12-01 và 2011-03-01 có cùng n_train:** snapshot 2010-12-01 có outcome kết thúc (exclusive) đúng **2011-03-01** = origin.
  Điều kiện `<` loại nó, dù về mặt khái niệm nhãn đã quan sát đủ (cửa sổ nửa mở). Đây là lựa chọn **bảo thủ**. Với `<=`, origin 2011-03-01 sẽ có 10.290 snapshot.
- **Vì sao 19.760 ≠ 13.736:** fit cuối của thiết kế chính chỉ dùng 3 snapshot (2010-12-01, 2011-03-01, 2011-06-01), còn rolling-origin 2011-09-10 thêm 2010-06-01 (2.703) và 2010-09-01 (3.321):
  13.736 + 6.024 = 19.760. Hai model khác nhau, nên metric khác nhau (báo cáo đã nêu ở 8.3).

## 3. Lựa chọn horizon và ngày quyết định

90 ngày và bốn ngày quyết định là **lựa chọn thiết kế**, không phải tối ưu đã chứng minh: horizon đủ ngắn để có nhiều snapshot đầy đủ trong 25 tháng dữ liệu
và đủ dài cho chu kỳ mua bán sỉ. Final test 2011-09-10 được chọn để [T, T+90) kết thúc 2011-12-09 (exclusive), loại ngày cuối không đầy đủ.
Hạn chế: 90 ngày chưa phải giá trị vòng đời; final test rơi vào mùa cao điểm Q4 (tỷ lệ mua lại 43,4% so với 32,9% ở validation), nên AP không so sánh trực tiếp giữa các kỳ.

## 4. RFM — cách tính và điểm cần lưu ý

`rfm_score` (`src/03_baseline_models.py`): mỗi chiều `pd.qcut(x.rank(method="first"), 4)`. R đảo ngược (gần đây hơn → điểm cao hơn).
Tổng 3–12: Champions ≥ 10, Loyal 8–9, At-risk 6–7, Lost ≤ 5. Tái tính độc lập khớp 100% với hàm gốc.

Ranh giới thực tế ở final test (`rfm_quartile_boundaries.csv`):

| Điểm | Recency (ngày) | Frequency (hóa đơn) | Monetary (£) |
|---|---|---|---|
| 1 | 326–647 | 1–1 | 2,90–323,86 |
| 2 | 163–326 | 1–3 | 324,19–804,02 |
| 3 | 49–163 | 3–6 | 804,88–2.118,75 |
| 4 | 0–49 | 6–309 | 2.118,91–484.615,10 |

- **Hòa (ties):** vì phải chia đều 4 nhóm theo hạng, các giá trị bằng nhau bị tách sang hai điểm khác nhau.
  Frequency có 3 giá trị bị tách (1, 3, 6), ảnh hưởng **2.392 khách**. Đáng chú ý: khách chỉ mua 1 lần có thể nhận F=1 hoặc F=2 tùy thứ tự dòng (Customer ID tăng dần).
  Recency có 3 giá trị bị tách (45 khách). Monetary không có.
- **Ảnh hưởng:** nếu dùng hạng trung bình (`method="average"`) thay vì `first`, **193/5.281 khách (3,7%)** đổi segment (`rfm_tie_effect.csv`).
  Nhãn được giữ nguyên; nên nêu đây là hạn chế (ranh giới không phải ngưỡng giá trị cố định, phá hòa theo thứ tự dòng).

## 5. "At-risk" là proxy điểm số, không phải nhãn churn

At-risk = tổng điểm 6–7, gộp các tổ hợp R/F/M rất khác nhau (`at_risk_rfm_patterns.csv`):

| R/F/M | Khách | Tỷ lệ mua lại | Diễn giải |
|---|---|---|---|
| 2/2/2 | 196 | 28,6% | Trung bình mọi chiều |
| 3/2/2 | 142 | 41,5% | Khá gần đây |
| 4/1/1 | 69 | 50,7% | **Khách mới mua gần đây** — không "có nguy cơ" |
| 4/1/2 | 40 | 55,0% | Khách mới, giá trị trung bình |
| 1/2/3 | 71 | 18,3% | Lâu không mua, từng chi khá — gần nghĩa "at-risk" nhất |

Tên "At-risk" vì vậy chỉ là quy ước. Segment này không được kiểm chứng độc lập như nhãn churn, và multiplier 1,3 gán cho nó là giả định
(xem `docs/assumption_register.md`). Báo cáo nên gọi là "nhóm điểm RFM trung bình" hoặc ghi chú rõ điều này.
