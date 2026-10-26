# Slide / Demo Outline

1. **Business question** — Khách hàng nào nên ưu tiên retention/promotion để
   tăng margin mà không over-discount? (đọc từ `docs/problem_statement.md`)
2. **Data & scope** — UCI Online Retail II, 5,041 khách hàng active,
   observation 19 tháng / outcome 5.3 tháng (`docs/data_dictionary.md`)
3. **Pipeline & validation** — RFM + cohort → baseline (naive/logistic/
   linear) → Gradient Boosting (PR-AUC 0.820) → rolling-origin validation
   (3 cutoff, std thấp) → SHAP (`docs/model_card.md`)
4. **Core insight (demo chính)** — Mở `outputs/figures/tornado_sensitivity.png`
   và scenario chart trong dashboard: blanket policy lỗ **-£63,000** ở kịch
   bản aggressive, targeted policy đạt **+£28,977** cùng kịch bản
5. **Live demo** — `streamlit run dashboard/app.py`: filter theo segment,
   xem actionable view (danh sách khách hàng ưu tiên), tải CSV
6. **Recommendation** — 3 nhóm hành động (retain/nurture/no-offer) +
   implementation roadmap (`docs/executive_brief.md`)
7. **Limitations & next step** — scenario-based (không causal), đề xuất A/B
   test thật (`docs/production_roadmap.md` mục 3)
8. **Engineering rigor** (nếu được hỏi) — 15 unit test, idempotent pipeline,
   CI, serving API, Docker (`README.md`)

Tài liệu nguồn cho từng slide đã liệt kê kèm theo — không cần tạo nội dung
mới, chỉ tổng hợp lại từ các file đã có.
