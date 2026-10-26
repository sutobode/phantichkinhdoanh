# Data Dictionary & Source Log

## Source log

- **Nguồn:** UCI Machine Learning Repository — Online Retail II
  (https://archive.ics.uci.edu/dataset/502/online+retail+ii)
- **URL tải trực tiếp:** https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip
- **Ngày tải:** 2026-10-26
- **File gốc:** `online_retail_II.xlsx`, 2 sheet: "Year 2009-2010", "Year 2010-2011"
- **Phạm vi dùng:** toàn bộ dataset (hai năm), chia observation/outcome window —
  giá trị cụ thể chốt sau khi chạy `src/01_eda.py` (xem `config/params.yaml`).

## Raw columns

| Column | Type | Mô tả |
|---|---|---|
| Invoice | string | Mã hóa đơn. Bắt đầu bằng "C" = cancellation. |
| StockCode | string | Mã sản phẩm. |
| Description | string | Tên sản phẩm. |
| Quantity | int | Số lượng. Có thể âm (return/cancellation). |
| InvoiceDate | datetime | Thời điểm giao dịch. |
| Price | float | Đơn giá (GBP). |
| Customer ID | float (nullable) | Mã khách hàng, có nhiều dòng null. |
| Country | string | Quốc gia khách hàng. |

## Filters & cleaning decisions

⚠️ **Thứ tự xử lý quan trọng để chống leakage**: split theo thời gian (observation
vs outcome window) được thực hiện **trước**, sau đó mới loại cancellation/qty/price
**riêng biệt trong từng window**. Nếu clean toàn cục rồi mới split, feature
`return_rate` sẽ không tính đúng theo window. Xem `src/02_cleaning_features.py`
và `tests/test_features.py` (có test leakage riêng cho `return_rate` và cho
boundary ngày cuối mỗi window).

Raw: 1,067,371 dòng (2 sheet gộp lại).

| Bước | Điều kiện loại | Phạm vi | Số dòng loại |
|---|---|---|---|
| 1 | Customer ID null | Toàn bộ dataset (time-independent) | 243,007 |
| 2 | Invoice bắt đầu bằng "C" | Observation window | 14,080 |
| 3 | Price <= 0 | Observation window | 44 |
| 4 | Invoice bắt đầu bằng "C" | Outcome window | 4,664 |
| 5 | Price <= 0 | Outcome window | 27 |
| **Tổng** | | | **261,822 / 1,067,371 (24.53%)** |

Quantity <= 0 không loại thêm dòng nào ở cả 2 window (đã trùng với cancellation,
Online Retail II không có quantity âm ngoài cancellation).

Observation window raw = 594,423 dòng → 580,299 dòng sạch.
Outcome window raw = 229,941 dòng → 225,250 dòng sạch.
(594,423 + 229,941 = 824,364 = đúng bằng số dòng sau khi loại Customer ID null —
xác nhận split không làm rơi/trùng dòng nào ở biên ngày.)

Observation window (2009-12-01 → 2011-06-30, trọn ngày) chứa **5,041 khách hàng**
dùng để tính feature.

## Joins

Không join thêm dataset ngoài (đúng Scope Guardrail: không merge dataset không
liên quan chỉ để tăng feature count).

## Derived tables

- `data/processed/customer_features.csv` — feature RFM + hành vi mỗi khách hàng, tính **chỉ từ** observation window
- `data/processed/customer_labels.csv` — nhãn repeat_purchase/future_value, tính **chỉ từ** outcome window
- `data/processed/customer_scores.csv` — deliverable cuối: feature + prediction + recommendation

### `customer_features.csv` — mô tả cột

| Cột | Công thức | Window nguồn |
|---|---|---|
| recency_days | (as_of − ngày mua gần nhất) | observation |
| frequency | số invoice duy nhất (đã loại cancellation) | observation |
| monetary | tổng Revenue (Quantity × Price) | observation |
| avg_order_value | monetary / frequency | observation |
| n_distinct_products | số StockCode duy nhất | observation |
| tenure_days | (as_of − ngày mua đầu tiên) | observation |
| return_rate | số invoice cancel / tổng invoice (kể cả cancel) | observation |
| country | quốc gia (lấy giá trị đầu) | observation |

`as_of` = ngày ngay sau `observation_end` (đảm bảo recency/tenure luôn ≥ 0,
xem comment trong `src/02_cleaning_features.py::main`).
