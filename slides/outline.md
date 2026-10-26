# Slide / Demo Outline

1. **Business question** — Khách hàng nào nên ưu tiên retention/promotion để
   tăng margin mà không over-discount? (đọc từ `docs/problem_statement.md`)
2. **Data & scope** — UCI Online Retail II, expanding customer snapshots,
   90-day gross-positive-sales outcomes; counts pending rerun (`docs/data_dictionary.md`)
3. **Pipeline & validation** — RFM + cohort → baseline (naive/logistic/
   linear) → optional Gradient Boosting/SHAP. Train decisions 2010-12-01/2011-03-01,
   validation 2011-06-01, final test 2011-09-10 (`docs/model_card.md`).
4. **Core insight** — Fixed unit costs and contact expenses make all three current
   scenarios unprofitable for every nonnegative value forecast: no-contact policy.
   Same maximum K: Random/RFM use K; proposed may abstain. All impacts simulated, not observed.
   Only show regenerated artifacts; see `outputs/RESULTS_STATUS.md`.
5. **Live demo** — `streamlit run dashboard/app.py`: filter theo segment,
   xem actionable view (danh sách khách hàng ưu tiên), tải CSV
6. **Recommendation** — 3 nhóm hành động (retain/nurture/no-offer) +
   implementation roadmap (`docs/executive_brief.md`)
7. **Limitations & next step** — scenario-based (không causal), đề xuất A/B
   test thật (`docs/production_roadmap.md` mục 3)
8. **Engineering rigor** (nếu được hỏi) — temporal/economics regression tests,
   CI, serving API, Docker (`README.md`)

Tài liệu nguồn cho từng slide đã liệt kê kèm theo — không cần tạo nội dung
mới, chỉ tổng hợp lại từ các file đã có.
