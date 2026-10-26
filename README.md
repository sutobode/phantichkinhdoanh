# Customer Value & Promotion Targeting for an Online Retailer

Capstone project — HUST Business Analytics 2026. Chi tiết yêu cầu: `phantichkinhdoanh.pdf`.
Lộ trình đầy đủ: `PLAN.md`.

## Business question

Khách hàng nào nên được ưu tiên cho retention/promotion để tăng expected future
margin mà không over-discount?

## Cài đặt

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
python src/01_eda.py                  # EDA, xác định observation/outcome window
python src/02_cleaning_features.py    # clean + RFM + cohort + feature engineering
python src/03_baseline_models.py      # RFM segmentation + baseline models
python src/04_advanced_models.py      # (stretch) gradient boosting + SHAP
python src/05_decision_scenario.py    # promotion-targeting rule + scenario simulation
python src/06_final_qa.py             # reproducibility check cuối cùng
```

Kết quả trung gian (hình, bảng, log) ghi vào `outputs/`. Model đã train lưu
trong `models/`. Bảng điểm khách hàng cuối cùng: `data/processed/customer_scores.csv`.

`src/06_final_qa.py` tự động chạy lại toàn bộ bước 02→05 từ đầu + pytest +
kiểm tra invariant (không cột trùng, không giá trị âm vô lý...) — chạy script
này trước khi nộp bài để xác nhận project reproducible.

## Dashboard

```bash
streamlit run dashboard/app.py
```

Dashboard tương tác: 5 KPI quyết định, filter theo quốc gia/segment/
recommendation, so sánh scenario, cohort retention heatmap, và actionable
view (danh sách khách hàng ưu tiên target theo ngân sách, tải CSV được).

## Serving API (mô phỏng tích hợp production)

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
pytest tests/        # 15 unit test: no-leakage (6), decision-logic (7), pipeline idempotency (2)
```

CI (`.github/workflows/ci.yml`) tự động chạy test này mỗi lần push.

## Cấu trúc project

Xem `PLAN.md` phần "Cấu trúc thư mục đề xuất".

## Tài liệu

- `docs/problem_statement.md` — business decision, unit of analysis, target, prediction horizon
- `docs/data_dictionary.md` — nguồn dữ liệu, filter, cleaning decisions
- `docs/model_card.md` — target, features, validation, kết quả model, assumptions, risks, ethics/bias
- `docs/executive_brief.md` — khuyến nghị cho decision-maker (≤2 trang)
- `docs/traceability_map.md` — trace recommendation → data/assumption/validation
- `docs/production_roadmap.md` — những gì cần thêm để đưa vào vận hành thật dài hạn
