# Production Roadmap

Project hiện tại là capstone 6 tuần, scope giới hạn theo yêu cầu học thuật.
Tài liệu này ghi lại những gì CẦN THÊM nếu đưa giải pháp vào vận hành thật
dài hạn — chứng minh đã nghĩ tới production dù ngoài phạm vi chấm điểm.

## 1. Monitoring & drift plan

- **Feature drift**: theo dõi Population Stability Index (PSI) giữa phân
  phối feature lúc train (observation window hiện tại) và phân phối feature
  của batch khách hàng mới mỗi tháng. PSI > 0.2 trên feature bất kỳ → cảnh
  báo review.
- **Performance decay**: theo dõi PR-AUC/MAE trên nhãn thực tế (có độ trễ —
  chỉ biết sau khi outcome window mới kết thúc) mỗi quý, so với baseline đã rerun
  trong `models/temporal_registry.json`: PR-AUC 0.750184, MAE £395.69 trên 5,281
  khách hàng final test. Giảm >10% tương đối → trigger retrain sớm.

## 2. Retraining trigger & cadence

- Retrain định kỳ **mỗi quý**: chỉ dùng snapshot trước đó có đủ outcome 90 ngày
  đã kết thúc trước decision mới; giữ feature history nhất quán với thiết kế.
- Retrain sớm nếu: (a) drift vượt ngưỡng ở mục 1, hoặc (b) có thay đổi lớn
  trong catalog sản phẩm/thị trường (vd mở rộng sang quốc gia mới).

## 3. Path to causal validation (quan trọng nhất)

Hiện tại mọi con số "incremental margin" là **scenario-based estimate**
(Scope Guardrail của đề — dataset không có randomized treatment/control).
Trước khi scale chính sách targeting ra toàn bộ customer base thật, cần:

1. Chọn ngẫu nhiên 1 nhóm holdout (10-20% khách hàng thuộc nhóm "nurture")
   KHÔNG nhận khuyến mãi, dù model đề xuất targeting.
2. Chạy chính sách targeting cho phần còn lại trong 1-2 chu kỳ mua hàng.
3. So sánh actual repeat-purchase rate & revenue giữa nhóm treatment và
   holdout → ước lượng **causal uplift thật** thay vì giả định
   `segment_response_multiplier`.
4. Dùng kết quả A/B test để hiệu chỉnh giả định tăng sales volume tương đối
   (`response_rate`, không phải response probability) trong `config/params.yaml`.

## 4. Feature store / train-serve consistency

Nếu tích hợp vào hệ thống CRM thật (serving real-time), công thức tính
feature (`src/02_cleaning_features.py::compute_rfm_features`) phải được dùng
**giống hệt** giữa lúc train và lúc serve — tránh train/serve skew. Khuyến
nghị: đóng gói logic tính feature thành 1 package riêng (không phải script),
import chung bởi cả batch training pipeline và serving API
(`src/serve.py` hiện đã tái sử dụng `FEATURE_COLS` + logic tương tự, nhưng ở
mức thật cần 1 feature store/service tập trung).

## 5. Scaling note

Dataset giao dịch được xử lý bằng pandas; số customer snapshot và thời gian chạy
thiết kế mới cần đo sau raw-data rerun. Nếu áp dụng cho customer base thật
lớn hơn (hàng triệu khách hàng, hàng trăm triệu giao dịch), cần:

- Chuyển ETL/feature engineering sang Spark/dask, hoặc chạy trực tiếp trên
  data warehouse (BigQuery/Snowflake SQL thay vì pandas).
- Batch scoring định kỳ (vd hàng đêm) thay vì tính lại toàn bộ feature mỗi
  lần gọi API.

## 6. Data privacy & compliance

- `customer_test_scores.csv` chứa predicted value + recommendation theo từng
  khách hàng — cần access control (chỉ team marketing/CRM được xem), không
  public.
- Nếu triển khai thật ở EU, cần tuân thủ GDPR: quyền được xóa dữ liệu
  (right to erasure) — khi khách hàng yêu cầu xóa, phải loại khỏi cả
  `customer_features.csv`, `customer_test_scores.csv`, và mọi model đã train trên
  dữ liệu của họ (hoặc retrain không có họ).

## 7. Rollout strategy

1. **Shadow mode** (1 tháng): chạy model + decision rule song song, KHÔNG
   áp dụng quyết định thật, so sánh với policy hiện tại.
2. **Canary** (1 tháng): áp dụng cho 10-20% khách hàng nhóm "nurture", theo
   dõi KPI thực tế so với dự báo.
3. **Full rollout**: nếu canary đạt kỳ vọng, mở rộng toàn bộ.
4. **Rollback plan**: nếu margin thực tế thấp hơn baseline (không targeting)
   sau canary, dừng ngay và quay lại policy cũ, review lại assumption.

## 8. Những gì ĐÃ code thật trong project này (không chỉ thiết kế)

- Experiment tracking: `models/temporal_registry.json` (ghi mỗi lần train: metric, git
  commit, timestamp).
- Model registry: `models/gbm_classifier_v1.joblib`, `gbm_regressor_v1.joblib`.
- Data validation layer: `src/data_validation.py` (pandera schema check trước
  khi chạy pipeline).
- Serving API: `src/serve.py` (FastAPI, endpoint `/score`).
- Containerize: `Dockerfile`.
- CI: `.github/workflows/ci.yml` (chạy pytest mỗi lần push).
- Logging có cấu trúc: `src/utils.py::get_logger` (ghi `outputs/logs/*.log`).
