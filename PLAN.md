# Plan: Customer Value & Promotion Targeting for an Online Retailer

Nguồn yêu cầu: `phantichkinhdoanh.pdf` (HUST Business Analytics Capstone 2026)

## Cấu trúc thư mục đề xuất

```
phantichkinhdoanh/
├── data/
│   ├── raw/                          # online_retail_II.xlsx
│   └── processed/                    # customer_features/labels/scores.csv
├── src/
│   ├── utils.py                      # config/logging/path helpers dùng chung
│   ├── data_validation.py            # pandera schema check
│   ├── 01_eda.py
│   ├── 02_cleaning_features.py
│   ├── 03_baseline_models.py
│   ├── 04_advanced_models.py         # GBM + SHAP + rolling-origin + ethics check
│   ├── 05_decision_scenario.py       # scenario/budget/recommendation
│   ├── 06_final_qa.py                # rerun pipeline + pytest + invariant checks
│   └── serve.py                      # FastAPI serving
├── outputs/                           # figures/, logs/, mọi bảng kết quả trung gian
├── config/params.yaml
├── tests/
│   ├── test_features.py              # 6 test chống leakage
│   ├── test_decision_scenario.py     # 7 test logic tài chính
│   └── test_pipeline_idempotency.py  # 2 test chống bug merge trùng cột
├── dashboard/app.py                  # Streamlit
├── docs/
│   ├── problem_statement.md
│   ├── data_dictionary.md
│   ├── executive_brief.md
│   ├── model_card.md
│   ├── traceability_map.md
│   └── production_roadmap.md
├── slides/outline.md
├── models/                           # gbm_*.joblib + registry.json
├── .github/workflows/ci.yml
├── Dockerfile
├── README.md
├── requirements.txt
└── PLAN.md
```

Trạng thái thực tế đã khớp 100% với cấu trúc này (đã verify bằng `find`).

---

## Định hướng điểm cao (Differentiators)

Rule tối thiểu trong đề chỉ là "ngưỡng pass". Để đạt điểm cao, cần vượt qua baseline ở các trục sau — các mục này được cài vào từng tuần bên dưới (đánh dấu **[Điểm cao]**):

1. **Rigor của validation**: không chỉ 1 temporal holdout mà dùng **rolling-origin validation** (nhiều điểm cắt thời gian khác nhau) để chứng minh model ổn định qua thời gian — đúng tinh thần "validation design phải khớp với data-generating process".
2. **So sánh nhiều baseline, không chỉ 1**: naive baseline (trung bình/tỷ lệ chung) → RFM rule-of-thumb → logistic/linear regression → (stretch) gradient boosting. Việc này cho thấy hiểu "baseline required" đúng nghĩa, không phải chỉ làm cho có.
3. **Định lượng uncertainty**: bootstrap confidence interval cho metrics (PR-AUC, MAE) và cho expected incremental margin — tránh báo cáo một con số point-estimate "chắc nịch" không có sai số, đây là lỗi phổ biến bị trừ điểm.
4. **Stretch nên làm thật, không bỏ qua**: Gradient boosting + SHAP, budget-constrained targeting optimization, customer-level recommendation layer (retain/nurture/no-offer). Làm tốt stretch là điểm khác biệt rõ nhất với các nhóm chỉ làm baseline.
5. **Traceability tường minh**: đề yêu cầu "final recommendation must be traceable from data, assumptions, analysis, and validation" — nên có hẳn 1 tài liệu `traceability_map.md` chỉ rõ từng khuyến nghị bắt nguồn từ đâu, thay vì để ngầm hiểu trong code.
6. **Ethics/bias check thực chất**: không chỉ liệt kê "có rủi ro bias" trong model card, mà thực sự kiểm tra performance/targeting rate có lệch bất thường theo quốc gia hay nhóm giá trị khách hàng không, và ghi lại kết quả.
7. **Kỹ thuật phần mềm sạch**: code modular trong `src/` (hàm tái sử dụng, không copy-paste giữa script), tham số hóa qua `config/params.yaml` (không hardcode ngày/threshold), unit test cho logic RFM và leakage-check, pin version thư viện trong `requirements.txt`, set seed.
8. **Dashboard thực sự "decision-relevant"**: có filter tương tác (theo segment/quốc gia) và 1 actionable view xuất ra được danh sách khách hàng ưu tiên kèm expected value — không chỉ là biểu đồ tĩnh.
9. **Executive brief có roadmap vận hành**: ngoài recommendation, nên có đoạn ngắn "cách retailer triển khai thực tế hàng tháng" + KPI theo dõi sau triển khai — thể hiện tư duy business thật, không chỉ tư duy model.

---

## Mở rộng Production-Ready (vượt chuẩn học thuật)

⚠️ Lưu ý ưu tiên: đây là phần **bonus sau khi đã hoàn thành đầy đủ 6 tuần + differentiators ở trên**. Không đánh đổi rigor của validation/baseline lấy việc dựng hạ tầng — thầy chấm theo rubric phân tích, không chấm theo DevOps. Chia 2 nhóm:

### (A) Nên code thật — effort thấp, giá trị cao, không lạc đề

- [x] **Experiment tracking nhẹ**: `models/registry.json` — mỗi lần train ghi run_id, git commit, metrics (baseline + GBM + rolling-origin)
- [x] **Data validation layer**: `src/data_validation.py` (pandera schema check), tích hợp vào đầu `02_cleaning_features.py::main()`
- [x] **Model registry tối giản**: `models/gbm_classifier_v1.joblib`, `models/gbm_regressor_v1.joblib`
- [x] **Serving API**: `src/serve.py` (FastAPI, endpoint `POST /score`) — đã test thật qua curl, xác nhận logic retain/nurture/no-offer đúng qua API
- [x] **Containerize**: `Dockerfile`
- [x] **CI tối giản**: `.github/workflows/ci.yml`
- [x] **Logging có cấu trúc**: `src/utils.py::get_logger`, toàn bộ script dùng logging (file UTF-8 + console)

### (B) Chỉ cần thiết kế/tài liệu hóa — `docs/production_roadmap.md`

- [x] Monitoring & drift plan
- [x] Retraining trigger & cadence
- [x] Path to causal validation (A/B test design)
- [x] Feature store / train-serve consistency
- [x] Scaling note
- [x] Data privacy/compliance
- [x] Rollout strategy (shadow → canary → full)

---

## TUẦN 1 — Business Framing & Data

- [x] Tạo cấu trúc thư mục ở trên
- [x] Tải dataset UCI Online Retail II → `data/raw/` (1,067,371 dòng, 2009-12-01 → 2011-12-09)
- [x] Viết problem statement (`docs/problem_statement.md`)
- [x] Chia timeline dataset: observation 2009-12-01→2011-06-30, outcome 2011-07-01→2011-12-09
- [x] EDA sơ bộ (`src/01_eda.py`) — chạy thành công, output `outputs/monthly_transaction_profile.csv` + log
- [x] `docs/data_dictionary.md` hoàn thiện (raw columns + derived features)
- [x] **[Điểm cao]** `config/params.yaml` — mọi tham số tập trung, không hardcode
- [x] **[Điểm cao]** EDA seasonality + repeat-purchase rate theo horizon (30/90/180 ngày) — dùng để chọn outcome window

**Checkpoint:** Problem statement + data readiness review

## TUẦN 2 — Cleaning, Features, Baseline

- [x] `src/02_cleaning_features.py` — clean, RFM, cohort, feature, labels. ⚠️ Đã phát hiện & fix 2 bug thật trong quá trình làm: (1) leakage tiềm ẩn ở `return_rate` nếu clean trước khi split window — fix bằng cách split theo thời gian TRƯỚC, clean riêng từng window SAU; (2) off-by-one tại boundary ngày cuối mỗi window (loại nhầm 625 dòng giao dịch thật ngày 2011-12-09) — fix bằng exclusive-end-of-day logic
- [x] **Cohort view**: `outputs/cohort_retention.csv`, hiển thị trong dashboard (heatmap)
- [x] Feature: recency, frequency, monetary, avg_order_value, n_distinct_products, tenure_days, **return_rate**
- [x] Labels: repeat_purchase, future_value — chỉ từ outcome window
- [x] Explicit leakage check — 6 unit test tự động (`tests/test_features.py`), bao gồm test khóa 2 bug đã fix ở trên
- [x] Validation design: temporal holdout ở tầng nhãn (ghi rõ lý do random split cross-sectional không gây leakage trong comment code)
- [x] `src/03_baseline_models.py` — RFM segmentation (Champions/Loyal/At-risk/Lost), Logistic Regression (PR-AUC 0.815), Linear Regression (MAE 593.3, RMSE 1,435.4)
- [x] `docs/data_dictionary.md` hoàn thiện với số liệu cleaning thật theo từng window
- [x] **`data/processed/customer_scores.csv`** — deliverable riêng biệt, cập nhật qua từng tuần (03→04→05)
- [x] **[Điểm cao]** Naive baseline: PR-AUC 0.502, MAE 1,115.6 — xác nhận model thật sự học được (logistic 0.815 >> 0.502)
- [x] **[Điểm cao]** `tests/test_features.py` — 6 test tự động chống leakage

**Checkpoint:** Reproducible baseline

## TUẦN 3 — Advanced Model & Diagnostics

- [x] Đánh giá baseline: PR-AUC 0.815 [0.785,0.847], ROC-AUC 0.804 [0.778,0.830], MAE/RMSE 593.3/1,435.4 — **đã bổ sung RMSE** (đề yêu cầu rõ "MAE/RMSE", ban đầu chỉ làm MAE, phát hiện thiếu khi tự-audit lại)
- [x] Error analysis theo RFM segment (`outputs/error_by_segment.csv`) + theo quốc gia (`outputs/ethics_bias_check_by_country.csv`)
- [x] `src/04_advanced_models.py`: GBM classifier (PR-AUC 0.820, thắng rõ) + regressor (MAE 531.2 tốt hơn, nhưng **RMSE 2,048.1 kém hơn** Linear Regression do 1 outlier — điều tra nguyên nhân cụ thể, trình bày trung thực trong `docs/model_card.md` thay vì chỉ báo cáo metric có lợi)
- [x] **[Điểm cao]** Rolling-origin validation: 3 cutoff (2011-01-31/03-31/06-30), GBM PR-AUC mean=0.714 std=0.035 — ổn định qua thời gian, `outputs/rolling_origin_validation.csv`
- [x] **[Điểm cao]** Bootstrap CI cho mọi metric chính (1000 resample), `config/params.yaml::n_bootstrap`
- [x] **[Điểm cao]** Calibration curve logistic vs GBM: `outputs/calibration_logreg.csv`, `calibration_gbm.csv`, chart `outputs/figures/calibration_curve.png`

**Checkpoint:** Model comparison + diagnostics

## TUẦN 4 — Decision Logic & Business Impact

- [x] `src/05_decision_scenario.py` — 3 scenario (conservative/moderate/aggressive), công thức incremental margin dựa trên segment_response_multiplier (lý thuyết uplift modeling), ghi rõ scenario-based không phải causal
- [x] Kết quả nổi bật: blanket policy ở kịch bản aggressive **lỗ -£62,890**; targeted policy cùng kịch bản **+£28,971** (minh họa đúng "tăng margin không over-discount")
- [x] **[Điểm cao]** Budget-constrained optimization: greedy top-N (tối ưu CHÍNH XÁC vì cost đồng nhất), budget £750 → 1,500 khách hàng, +£22,787 (kịch bản moderate)
- [x] **[Điểm cao]** Customer-level recommendation: nurture (2,072) / retain (1,459) / no-offer (1,510), gắn lý thuyết uplift (sure things/persuadables/lost causes)
- [x] **[Điểm cao]** Bootstrap CI cho total incremental margin: 26,033 [25,200, 26,908]
- [x] **Senior code review** (độc lập, sau khi "xong"): phát hiện 7 bug/vấn đề thật (giá trị dự đoán âm gây margin dương giả, registry run_id trùng, RFM segment tính lại trên test-subset gây sai nhãn 93/1009 khách hàng, bootstrap_ci có thể crash, serve.py thiếu validate input, code trùng lặp, tính toán dư thừa) — đã fix toàn bộ + viết test khóa lại, xem `docs/model_card.md` mục "Bug đã phát hiện qua senior code review"
- [x] **[Điểm cao]** Tornado sensitivity: response_rate (swing 143,516) và discount_pct (104,602) là 2 yếu tố nhạy nhất, chart `outputs/figures/tornado_sensitivity.png`
- [x] 7 unit test cho logic tài chính (`tests/test_decision_scenario.py`) — công thức margin, budget optimization, recommendation logic

**Checkpoint:** Actionable decision logic

## TUẦN 5 — Robustness, Dashboard, Draft Brief

- [x] Sensitivity analysis — tornado chart (xem Tuần 4)
- [x] Dashboard Streamlit (`dashboard/app.py`): 5 KPI, scenario comparison chart, segment/recommendation chart, cohort heatmap, actionable view — **đã test bằng browser thật (Playwright), xác nhận render đúng không lỗi**, bao gồm fix 1 bug thật (thiếu cột `country` trong customer_scores.csv)
- [x] `docs/executive_brief.md` (≤2 trang) — số liệu đối chiếu khớp 100% với `outputs/scenario_table.csv`
- [x] **[Điểm cao]** Dashboard có filter (country/segment/recommendation) + actionable view tải CSV được
- [x] **[Điểm cao]** Ethics/bias check thực chất: error GBM theo quốc gia (UK vs Other, không nhóm nào lệch >30%), `outputs/ethics_bias_check_by_country.csv`
- [x] **[Điểm cao]** Executive brief có "Đề xuất triển khai" (pilot A/B test, vận hành hàng tháng, KPI theo dõi, ngưỡng cảnh báo)

**Checkpoint:** Draft dashboard + recommendations

## TUẦN 6 — Final QA & Submission

- [x] `src/06_final_qa.py`: chạy lại pipeline 02→05 từ đầu + pytest + 7 invariant check — **PASSED**, kết quả y hệt lần chạy gốc (5,041 khách hàng, PR-AUC 0.820, margin scenario khớp executive brief)
- [x] `README.md` đầy đủ: cài đặt, chạy pipeline, dashboard, serving API, Docker, test, cấu trúc, tài liệu
- [x] `model_card.md` hoàn thiện: kết quả model thật, validation design, assumptions, risks, ethics/bias (kết quả thật), limitations
- [x] Executive brief đối chiếu đủ 4 nội dung + số liệu verify khớp 100% với output thật
- [x] `customer_scores.csv` phiên bản cuối (18 cột, không trùng lặp, đã qua idempotency test)
- [x] `slides/outline.md` — khung demo
- [x] **[Điểm cao]** `docs/traceability_map.md` — 6 khuyến nghị chính, mỗi cái trace đủ data→assumption→analysis→validation→output
- [x] **[Điểm cao]** 15 unit test pass (`pytest tests/`), kết quả đưa vào README
- [x] **[Điểm cao]** Rà overclaim: grep "causal uplift|guaranteed|proven|ROI" — không có vi phạm

**Checkpoint:** Final submission package

---

## Mandatory Rules — checklist xuyên suốt

- [x] Unit of analysis, target, prediction horizon, business decision — `docs/problem_statement.md`
- [x] Validation = temporal holdout ở tầng nhãn (+ rolling-origin bổ sung)
- [x] Không dùng thông tin ngoài observation window — 6 automated leakage test, bao gồm 2 bug thật đã fix
- [x] Baseline hoàn thành trước stretch (naive→logistic/linear→GBM, đúng thứ tự)
- [x] Không claim causal uplift/ROI — ghi rõ scenario-based ở mọi nơi, verify bằng grep
- [x] Business impact đo bằng decision rule + budget constraint cụ thể (không chỉ accuracy)
- [x] Không merge dataset ngoài
- [x] **[Điểm cao]** Mọi metric chính có bootstrap CI (PR-AUC, ROC-AUC, MAE, incremental margin)
- [x] **[Điểm cao]** Traceability map đầy đủ 6 khuyến nghị chính

## Tổng kết trạng thái: **TẤT CẢ các mục đã hoàn thành** (bao gồm cả core requirements, differentiators, và production extensions). Đã chạy `src/06_final_qa.py` xác nhận PASSED — project sẵn sàng nộp bài.
