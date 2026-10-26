# Executive Brief — Customer Retention & Promotion Targeting

**Để:** Marketing / CRM decision-maker | **Dataset:** 2 năm giao dịch, 5,041 khách hàng active

## Câu hỏi kinh doanh

Chúng ta nên tặng khuyến mãi cho ai để **tăng lợi nhuận kỳ vọng**, thay vì
giảm giá tràn lan và làm mất margin ở những khách hàng đã sẵn sàng mua?

## Phát hiện chính

Gửi khuyến mãi **cho TẤT CẢ khách hàng** (chính sách hiện tại/mặc định ở
nhiều nơi) có thể **phá hủy lợi nhuận** nếu mức giảm giá lớn: mô phỏng cho
thấy ở kịch bản giảm giá 20%, chính sách "tặng cho tất cả" lỗ **-£63,000**
margin kỳ vọng — vì nhiều khách hàng (nhóm "Champions") vốn đã mua đều đặn dù
không có khuyến mãi, nên giảm giá cho họ chỉ làm mất margin, không tạo thêm
doanh thu.

**Nhắm mục tiêu đúng khách hàng** đảo ngược hoàn toàn kết quả: cùng kịch bản
giảm giá 20%, chỉ gửi khuyến mãi cho nhóm khách hàng phù hợp mang lại
**+£28,977** margin kỳ vọng — chênh lệch **~£92,000** so với tặng tràn lan.

## Khuyến nghị

Phân loại khách hàng thành 3 nhóm hành động, dựa trên mô hình dự báo (Gradient
Boosting, độ chính xác PR-AUC 0.82, đã kiểm định ổn định qua nhiều mốc thời
gian):

| Nhóm | Số khách hàng | Hành động | Lý do |
|---|---|---|---|
| **Nurture** (At-risk, Loyal) | 2,073 | Gửi khuyến mãi | Nhóm "dễ bị tác động" — khuyến mãi tạo doanh thu tăng thêm đủ bù chi phí giảm giá |
| **Retain** (Champions) | 1,459 | Chăm sóc KHÔNG dùng khuyến mãi (loyalty, ưu tiên dịch vụ) | Đã mua đều đặn, giảm giá chỉ lãng phí margin |
| **No-offer** (Lost, giá trị thấp) | 1,509 | Không đầu tư | Khả năng phản hồi thấp, chi phí không đáng |

**Chính sách đề xuất (kịch bản "moderate": giảm giá 10%, ngân sách liên hệ
£750):** nhắm mục tiêu tối đa **1,500 khách hàng** nhóm Nurture có margin kỳ
vọng cao nhất → **+£22,787** margin kỳ vọng (so với +£8,830 nếu tặng tràn
lan cùng mức giảm giá).

## Giả định quan trọng (cần lưu ý)

- Các con số margin là **ước lượng theo kịch bản**, dựa trên giả định về tỷ
  lệ phản hồi (response rate) và biên lợi nhuận — **không phải** kết quả đã
  được kiểm chứng bằng thử nghiệm thực tế (dataset hiện tại không có nhóm đối
  chứng/A-B test).
- Tỷ lệ phản hồi giả định **khác nhau theo nhóm khách hàng**: nhóm "At-risk/
  Loyal" được giả định phản hồi tốt hơn khuyến mãi (dựa trên lý thuyết hành
  vi khách hàng), nhóm "Champions" được giả định ít bị tác động vì đã trung
  thành sẵn.
- Đã kiểm tra độ nhạy (sensitivity): kết quả nhạy nhất với **mức giảm giá**
  và **tỷ lệ phản hồi thực tế** — đây là 2 yếu tố nên verify sớm nhất bằng
  thử nghiệm thực tế.
- Model dự báo giá trị khách hàng **kém chính xác hơn ở nhóm khách hàng chi
  tiêu cực cao** (vd top 1% "whale customer") — nên review thủ công trước
  khi áp dụng khuyến nghị cho nhóm này, không nên tự động hóa hoàn toàn.

## Đề xuất triển khai

1. **Tháng 1-2**: Chạy pilot A/B test thật (có nhóm đối chứng) trên nhóm
   "Nurture" để đo tỷ lệ phản hồi thực tế, thay giả định bằng số liệu đo
   được.
2. **Vận hành hàng tháng**: cập nhật `customer_scores.csv` mỗi tháng với dữ
   liệu giao dịch mới, refresh danh sách nhắm mục tiêu.
3. **KPI theo dõi sau triển khai**: tỷ lệ phản hồi thực tế theo segment, chi
   phí khuyến mãi/doanh thu tăng thêm, margin thực tế so với dự báo.
4. **Ngưỡng cảnh báo**: nếu response rate thực tế thấp hơn >30% so với giả
   định, dừng và review lại model/giả định (không tiếp tục chạy chính sách
   dựa trên giả định sai).

## Rủi ro chính

Nếu giả định response rate sai (vd Champions phản hồi tốt hơn dự kiến), chi
phí cơ hội là bỏ lỡ margin từ nhóm đó — nhưng rủi ro NGƯỢC LẠI (tặng tràn lan
và lỗ margin như minh họa ở trên) lớn hơn nhiều. Khuyến nghị bắt đầu thận
trọng (kịch bản "conservative" hoặc "moderate") và mở rộng dần sau khi có dữ
liệu A/B test thật.
