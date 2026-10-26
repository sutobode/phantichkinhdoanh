# Traceability Map

Mỗi khuyến nghị chính trong `executive_brief.md` trace ngược về data →
assumption → analysis → validation cụ thể, đáp ứng yêu cầu "final
recommendation must be traceable from data, assumptions, analysis, and
validation" của đề.

---

## Khuyến nghị 1: "Tặng khuyến mãi tràn lan có thể lỗ margin (-£63,000 ở kịch
bản aggressive)"

- **Data**: `data/processed/customer_scores.csv` cột `predicted_future_value_gbm`
  (5,041 khách hàng).
- **Analysis**: `src/05_decision_scenario.py::build_scenario_table()`, công
  thức `incremental_margin_i = v_i * gross_margin_pct * [(1+eff_r)(1-d) - 1]`
  áp dụng cho TẤT CẢ khách hàng (blanket policy).
- **Assumption**: `config/params.yaml::decision` — `gross_margin_pct=0.40`,
  `discount_response_scenarios[aggressive]={d:0.20, r:0.35}`,
  `segment_response_multiplier`.
- **Validation**: không áp dụng trực tiếp (đây là phép tính kế toán dựa trên
  v_i đã được validate — xem Khuyến nghị 3), nhưng công thức margin được unit
  test (`tests/test_decision_scenario.py::test_incremental_margin_*`, 2 test).
- **Output**: `outputs/scenario_table.csv`, hàng `aggressive`, cột
  `total_margin_non_targeted_blanket = -63,005`.

## Khuyến nghị 2: "Targeting đúng mang lại +£28,977 ở cùng kịch bản"

- **Data**: như trên.
- **Analysis**: cùng hàm `build_scenario_table()`, cột
  `total_margin_targeted_positive_only` = sum(incremental_margin_i) với
  incremental_margin_i > 0.
- **Assumption**: như trên + quy tắc "chỉ target nếu margin > 0" (không có
  thêm giả định mới).
- **Validation**: `tests/test_decision_scenario.py::test_budget_constrained_selection_*`
  xác nhận logic lọc margin>0 đúng.
- **Output**: `outputs/scenario_table.csv`, hàng `aggressive`, cột
  `total_margin_targeted_positive_only = 28,976.69`.

## Khuyến nghị 3: "Dùng model GBM (PR-AUC 0.82) để dự báo predicted_future_value/
proba — nền tảng cho mọi tính toán margin ở trên"

- **Data**: `data/processed/customer_features.csv` (features từ observation
  window 2009-12-01→2011-06-30) + `customer_labels.csv` (labels từ outcome
  window 2011-07-01→2011-12-09).
- **Analysis**: `src/04_advanced_models.py` — GradientBoostingClassifier/
  Regressor, train trên 80% khách hàng (random theo customer), eval trên 20%
  còn lại.
- **Assumption**: validation design = temporal holdout ở tầng nhãn + random
  split cross-sectional cho train/test (xem comment trong
  `src/03_baseline_models.py`, lý do tại sao random split ở đây KHÔNG gây
  leakage).
- **Validation**:
  - Bootstrap CI: PR-AUC 0.820 [0.787, 0.851] (1000 lần resample).
  - Rolling-origin validation: 3 cutoff khác nhau, std=0.035 (ổn định).
  - 6 unit test chống leakage trong `tests/test_features.py`.
- **Output**: `models/gbm_classifier_v1.joblib`, `models/gbm_regressor_v1.joblib`,
  `models/registry.json` (run_id `advanced_gbm_v1`).

## Khuyến nghị 4: "Champions → retain (không discount), Lost → no-offer,
At-risk/Loyal → nurture"

- **Data**: `customer_scores.csv` cột `rfm_segment` (quantile-based trên
  recency/frequency/monetary, `src/03_baseline_models.py::rfm_segment()`) +
  `incremental_margin_reference_scenario`.
- **Analysis**: `src/05_decision_scenario.py::assign_recommendation()` — quy
  tắc: margin>0 → nurture; margin≤0 & segment∈{Champions,Loyal} & value>0 →
  retain; còn lại → no-offer.
- **Assumption**: `segment_response_multiplier` (Champions=0.5 "sure thing",
  At-risk=1.3 "persuadable", Lost=0.4 "lost cause") — dựa trên lý thuyết
  uplift modeling, KHÔNG ước lượng từ dữ liệu thật (ghi rõ trong
  `config/params.yaml` và `docs/model_card.md`).
- **Validation**: `tests/test_decision_scenario.py::test_assign_recommendation_logic`.
- **Output**: `customer_scores.csv` cột `final_recommendation`; phân phối xác
  nhận trong `docs/model_card.md` (Champions 1459/1460 → retain, Lost
  1506/1508 → no-offer).

## Khuyến nghị 5: "Chính sách moderate, ngân sách £750 → nhắm 1,500 khách
hàng, +£22,787"

- **Data**: `incremental_margin_reference_scenario` (reference = "moderate").
- **Analysis**: `src/05_decision_scenario.py::budget_constrained_selection()`
  — greedy top-N theo margin giảm dần, N = floor(budget/cost) = floor(750/0.5)
  = 1500. Tối ưu CHÍNH XÁC (không phải xấp xỉ) vì cost đồng nhất mỗi khách
  hàng (0/1 knapsack với cost bằng nhau → greedy-by-value là lời giải tối
  ưu).
- **Assumption**: `contact_cost_per_customer=0.50`, `budget_total=750.0`.
- **Validation**: `tests/test_decision_scenario.py::test_budget_constrained_selection_respects_cap_and_positivity`.
- **Output**: `outputs/scenario_table.csv` hàng `moderate`, cột
  `total_margin_budget_constrained=22,787`, `n_customers_budget_constrained=1500`.

## Khuyến nghị 6: "Kết quả nhạy nhất với discount_pct và response_rate"

- **Data/Analysis**: `src/05_decision_scenario.py::tornado_sensitivity()` —
  vary từng tham số quanh base case (moderate scenario), đo swing lên
  `total_margin_targeted_positive_only`.
- **Output**: `outputs/tornado_sensitivity.csv` — `response_rate` swing=143,516
  (lớn nhất), `discount_pct` swing=104,602 (thứ 2), `at_risk_response_multiplier`
  swing=13,750, `gross_margin_pct` swing=13,017.
