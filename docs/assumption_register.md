# Assumption Register & Business-Assumption Robustness

Phạm vi: các giả định kinh tế của decision layer (`config/params.yaml › decision`, `src/05_decision_scenario.py`).
Đây là **phân tích phụ (secondary)**: không đổi config, công thức, temporal split, target hay kết quả chính.
Số liệu: `outputs/assumption_robustness.csv`, `outputs/assumption_robustness/*.csv`
(sinh bởi `src/11_assumption_robustness.py`; test: `tests/test_assumption_robustness.py`).

Công thức (giữ nguyên): `ΔM_i = v_i·[(1 + b·m_z)(g − d) − g] − c`, với `v_i` = dự báo doanh số gộp 90 ngày (Linear, clip ≥ 0).
Khuyến mãi có lãi trên mỗi £ chỉ khi `b·m_z > d/(g − d)` (ngưỡng tăng sản lượng hòa vốn).

## 1. Bảng giả định

Phân loại nguồn: **DO** = quan sát từ dataset · **LIT** = tài liệu bên ngoài · **PA** = giả định lập kế hoạch · **DC** = lựa chọn thiết kế.

| Giả định | Giá trị hiện tại | Ý nghĩa / đơn vị | Nguồn | Căn cứ | Dải nhạy hợp lý | Hạn chế |
|---|---|---|---|---|---|---|
| Gross margin `g` | 40% | Lợi nhuận gộp trên £1 doanh số, đồng nhất mọi SKU | **PA** | Không có dữ liệu giá vốn. Retailer bán quà tặng (giftware), nhiều khách là **nhà bán sỉ** [1] — biên bán sỉ thường thấp hơn bán lẻ. Không tìm được benchmark có thẩm quyền cho đúng ngành (nguồn thứ cấp cho "general retail" ~33% — chưa kiểm chứng) | 30–50% (đã test) | Biên thật khác nhau theo SKU/khách; kết luận nhạy với `g` (§3) |
| Mức giảm giá `d` | 5% / 10% / 20% | % giảm trên giá bán | **DC** | Ba mức minh họa nhẹ–vừa–sâu; không phải mức tối ưu | 5–20% (đã test thêm 15%) | Không mô hình hóa ngưỡng giá tâm lý, loại ưu đãi (free shipping, bundle) |
| Tăng sản lượng `b` | 10% / 20% / 35% | % tăng doanh số **trước giảm giá** của khách được liên hệ (KHÔNG phải tỷ lệ redemption) | **PA** | Không ước lượng được từ dataset (không có treatment/control). Meta-analysis cho độ co giãn giá cấp thương hiệu: trung bình −1,76 [2] và −2,62 [3] ⇒ tăng sản lượng ngụ ý ở d=10%: 20% và 32% (§4) | 10–50% (đã test) | Phụ thuộc d (độ co giãn) — mô hình cấu hình coi d và b độc lập; bump khuyến mãi có phần chuyển thương hiệu/dồn mua [4] |
| Multiplier segment `m_z` | Champions 0,5 · Loyal 1,0 · At-risk 1,3 · Lost 0,4 | Hệ số nhân `b` theo RFM segment | **PA** | Lập luận định tính (sure things / persuadables / lost causes). Dataset chỉ xác nhận **thứ tự tỷ lệ mua lại** (Champions 76,5% > Loyal 46,6% > At-risk 33,0% > Lost 16,9%), KHÔNG xác nhận độ nhạy với khuyến mãi | 4 bộ: configured / flat / compressed / steep (đã test) | RFM không nhận diện loại phản ứng điều trị; "At-risk" là proxy điểm số (xem `docs/temporal_rfm_diagnostics.md`) |
| Chi phí liên hệ `c` | £0,50 / khách | Chi phí biên mỗi lần liên hệ | **PA** | Nằm trong dải thư trực tiếp UK (~50p–£1/thư, nguồn nhà cung cấp [5]); cao hơn nhiều so với email (~20p cho email prospecting gồm chi phí dữ liệu, nguồn thứ cấp [6]) | £0,05–£1 | Nguồn chất lượng thấp; với `c` nhỏ, ngưỡng v tối thiểu `c/α` giảm nhưng **dấu của α không đổi** |
| Ngân sách | £750 ⇒ K = 1.500 | Trần chi phí liên hệ | **DC** | Ràng buộc minh họa | — | Không mô hình hóa chi phí giảm giá trong ngân sách (đã nằm trong ΔM) |
| Giá vốn đơn vị không đổi | — | Giảm giá trừ thẳng vào biên | **DC** | Đúng kinh tế học cho giảm giá bán | — | Bỏ qua chiết khấu nhà cung cấp khi tăng sản lượng |
| Đồng nhất sản phẩm/khách | — | Một `g`, một `d` cho mọi khách | **DC** | Đơn giản hóa | — | Bỏ qua cơ cấu sản phẩm, khách bán sỉ vs lẻ |
| Không có treatment/control | — | Mọi tác động khuyến mãi là kịch bản | **DO** | Dataset không ghi nhận khuyến mãi | — | Không thể kiểm chứng `b`, `m_z` từ dữ liệu hiện có |

**Mô phỏng hiện tại chứng minh được gì / không được gì.** Mô phỏng chứng minh rằng *nếu* các giả định đúng thì
chính sách tối ưu là không triển khai, và chỉ ra chính xác ngưỡng mà giả định phải vượt để kết luận đảo chiều. Mô phỏng
**không** chứng minh được rằng khách hàng của retailer này phản ứng với khuyến mãi theo mức `b` hay `m_z` nào:
đó là giả định lập kế hoạch, không phải ước lượng thực nghiệm. Bằng chứng ngoài duy nhất có thể dùng (độ co giãn giá từ
meta-analysis) là cấp thương hiệu trong hàng tiêu dùng nhanh, nên chỉ dùng để định vị độ lớn hợp lý, không để hiệu chỉnh.

## 2. Nguồn tham khảo

1. UCI Machine Learning Repository, *Online Retail II* (2019) — "The company mainly sells unique all-occasion gift-ware. Many customers of the company are wholesalers." https://archive.ics.uci.edu/dataset/502/online+retail+ii · Liên quan: Chen, Sain & Guo (2012), *J. Database Marketing & Customer Strategy Management* 19(3):197–208, doi:10.1057/dbm.2012.17.
2. Tellis, G. J. (1988). The price elasticity of selective demand: A meta-analysis of econometric models of sales. *Journal of Marketing Research* 25(4):331–341. Trung bình −1,76 (367 ước lượng). Xác nhận qua bản tóm tắt MSI: https://www.msi.org/working-papers/the-price-elasticity-of-selective-demand-a-metaanalysis-of-sales-response-models/ — *chưa đọc toàn văn*.
3. Bijmolt, T. H. A., van Heerde, H. J., & Pieters, R. G. M. (2005). New empirical generalizations on the determinants of price elasticity. *Journal of Marketing Research* 42(2):141–156. Trung bình −2,62 (1.851 ước lượng, 81 nghiên cứu). https://repository.tilburguniversity.edu/items/813834f9-4626-45f8-ad0a-2920a545e089
4. van Heerde, H. J., Gupta, S., & Wittink, D. R. (2003). Is 75% of the sales promotion bump due to brand switching? No, only 33% is. *Journal of Marketing Research* 40(4):481–491. Working paper: https://ideas.repec.org/p/ysm/somwrk/ysm222.html
5. Oppizi (blog nhà cung cấp, không đề ngày), "How much does a direct mail campaign cost in the UK?" — 50p–£1/thư. https://oppizi.com/uk/en/resources/blog/direct-mail/how-much-should-a-direct-mail-campaign-cost-in-the-u-k/ — **chất lượng thấp**.
6. Digital Doughnut (2019), "Unlock the potential of combined print and digital" (trích Wilmington/DMA) — ~20p/email prospecting. https://www.digitaldoughnut.com/articles/2019/february/unlock-the-potential-of-combined-print-and-digital — **chất lượng thấp, cũ**.

**Còn chờ (pending):** benchmark gross margin có thẩm quyền cho bán sỉ/bán lẻ quà tặng UK (ví dụ bảng margins-by-sector của Damodaran — chưa trích vì chưa xác minh được số liệu); benchmark chi phí liên hệ từ DMA/Royal Mail/JICMAIL.

## 3. Lưới giả định khai báo trước (Task C)

Lưới: `g` ∈ {30, 40, 50%} × `d` ∈ {5, 10, 15, 20%} × `b` ∈ {10, 20, 35, 50%} × 4 bộ multiplier = **192 tổ hợp**
(cộng 48 dòng elasticity minh họa). Bộ multiplier: configured (0,5/1/1,3/0,4), flat (1/1/1/1), compressed (0,75/1/1,15/0,7),
steep (0,25/1/1,6/0,2). Không chọn tổ hợp theo lợi nhuận.

- Tổ hợp tham chiếu (g=40%, d=10%, b=20%, configured): **không triển khai** (0 khách); random K = −£37.579; RFM = −£118.314.
- **70/192 tổ hợp (36%) dẫn tới triển khai.** Tỷ lệ triển khai theo d: 5% → 81%; 10% → 46%; 15% → 17%; 20% → 2%. Theo b: 10% → 8%; 20% → 27%; 35% → 48%; 50% → 63%. Theo g: 30% → 23%; 40% → 36%; 50% → 50%.
- Trong **mọi** tổ hợp: proposed ≥ random K và proposed ≥ RFM top-K (vì được phép không chọn ai). Random K lỗ ở 82%, RFM top-K lỗ ở 84% tổ hợp.

Mức tăng sản lượng nhỏ nhất trong lưới để triển khai (`break_even_regions.csv`; trống = không triển khai với b ≤ 50%):

| g | multiplier | d=5% | d=10% | d=15% | d=20% |
|---|---|---|---|---|---|
| 30% | configured | 20% | 50% | — | — |
| 40% | configured | 20% | 35% | 50% | — |
| 40% | flat | 20% | 35% | — | — |
| 50% | configured | 10% | 20% | 35% | — |
| 50% | steep | 10% | 20% | 35% | 50% |

**Diễn giải.** Ở chính 3 kịch bản cấu hình, kết luận "không triển khai" đúng trong 42/48 tổ hợp (g × bộ multiplier).
6 ngoại lệ đều chỉ kích hoạt segment At-risk, với lãi nhỏ (£22–£2.567):
- conservative + steep (At-risk 1,6) ở g = 40%: 466 khách, £320;
- conservative ở g = 50% với configured / compressed / steep;
- moderate ở g = 50% với configured / steep.

Aggressive không bao giờ có lãi. Ngoài 3 kịch bản cấu hình, kết luận chỉ đảo chiều khi đồng thời giảm giá nhẹ (5–10%)
**và** tăng sản lượng ≥ 20–35%, hoặc `g` = 50%. Giảm giá 20% gần như không bao
giờ có lãi trong lưới. Vì vậy câu hỏi kinh doanh then chốt không phải "model nào tốt hơn" mà là
**"một ưu đãi 5–10% có làm khách mua thêm ≥ 20–35% không?"** — chỉ thử nghiệm thật trả lời được.

## 4. Độ co giãn giá — minh họa (không phải ước lượng nhân quả)

Với cầu co giãn hằng số ε: `b = (1 − d)^ε − 1`. Ngưỡng |ε| để có lãi (multiplier = 1): `|ε*| = ln(g/(g−d)) / −ln(1−d)`.

| g | d=5% | d=10% | d=15% | d=20% |
|---|---|---|---|---|
| 30% | 3,55 | 3,85 | 4,27 | 4,92 |
| 40% | **2,60** | 2,73 | 2,89 | 3,11 |
| 50% | 2,05 | 2,12 | 2,20 | 2,29 |

Với g = 40%, độ co giãn trung bình −1,76 [2] **không đạt** ngưỡng ở mọi mức giảm giá; −2,62 [3] chỉ vừa chạm ngưỡng ở d=5%
(14,4% so với cần 14,3%). Hơn nữa, các con số này là độ co giãn **cấp thương hiệu**: một phần đáng kể của bump khuyến mãi
là chuyển thương hiệu/dồn mua [4], vốn không phải doanh số tăng thêm của nhà bán lẻ. ⇒ Bằng chứng ngoài **ủng hộ** kết luận
thận trọng (abstain) hơn là bác bỏ nó. Với multiplier configured và ε = −2,62, chính sách chọn 662–693 khách và lãi
£1.268–£2.148 (d = 5–15%, g = 40%) — nhỏ và hoàn toàn phụ thuộc multiplier At-risk/Loyal.

## 5. Monte Carlo random policy (Task D)

`monte_carlo_convergence.csv`: 5 seed khai báo trước (11, 22, 33, 44, 55) × {500, 1.000, 5.000} lần rút, so với kỳ vọng giải tích
`E = (K/N)·Σ net_i` (đúng với chọn đều K không hoàn lại; có unit test so với liệt kê toàn bộ tổ hợp).

| Kịch bản | Giải tích | Sai số tương đối lớn nhất: 500 / 1.000 / 5.000 lần rút |
|---|---|---|
| moderate | −£37.579 | 0,97% / 0,75% / 0,34% |
| illustrative light (5%, 20%) | −£5.817 | 1,39% / 0,96% / 0,47% |

Giá trị báo cáo (−£37.407,57; 1.000 lần rút, seed 42) lệch 0,46% so với giải tích — trong biên Monte Carlo. Có thể báo cáo giá trị
giải tích như con số chính xác và giữ Monte Carlo cho khoảng biến thiên. Khoảng này chỉ phản ánh biến thiên giữa các danh sách ngẫu
nhiên, **không** phải bất định nhân quả.

## 6. Feasibility `future_net_sales_proxy` (Task E — chỉ khảo sát)

`net_proxy_feasibility.csv`, final test [2011-09-10, 2011-12-09):

- 2.657 dòng cancellation của khách final test, trị giá £101.864 = **3,6%** doanh số gộp cùng kỳ (£2,87 triệu).
- **0%** hóa đơn hủy có số trùng hóa đơn mua (bỏ chữ "C") ⇒ **không ghép được theo hóa đơn**; chỉ ghép xấp xỉ theo (khách, StockCode, ngày mua gần nhất trước đó).
- Theo ghép xấp xỉ: 65,6% giá trị hủy gắn với lần mua **trong** kỳ outcome; **23,9%** gắn với lần mua **trước** decision date
  (trừ vào nhãn tương lai sẽ phạt hành vi quá khứ); 10,4% không tìm thấy lần mua tương ứng. Độ trễ trung vị 14 ngày.

**Khuyến nghị:** không thay `future_gross_sales`. Nếu cần, chỉ dùng proxy dạng
`net = gross − cancellation ghép được trong cùng kỳ`, báo cáo như phân tích độ nhạy. Kiểm tra trước (`outputs/robustness/ba_gross_vs_net.csv`)
cho thấy danh sách chọn trùng 97–99% và 3 kịch bản cấu hình vẫn chọn 0 khách. Không có dữ liệu giá vốn nên không thể gọi là "lợi nhuận thật".

## 7. Câu hỏi nhóm cần sẵn sàng bảo vệ

1. Vì sao g = 40%? → Giả định. Với 3 kịch bản cấu hình, kết luận giữ ở 42/48 tổ hợp; ngoại lệ cần g = 50% hoặc multiplier At-risk ≥ 1,6 (§3).
2. Multiplier lấy từ đâu? → Giả định định tính; dataset chỉ xác nhận thứ tự tỷ lệ mua lại. Champions/Lost không bao giờ ảnh hưởng quyết định; At-risk/Loyal mới là then chốt.
3. Tăng 20% sản lượng có hợp lý không? → Theo meta-analysis cấp thương hiệu, d=10% ngụ ý 20–32%; cấp cửa hàng có thể thấp hơn [4]. Vẫn dưới ngưỡng 33% cần thiết.
4. Sao không ước lượng uplift từ dữ liệu? → Không có treatment/control; mọi ước lượng quan sát sẽ bị nhiễu (Scope Guardrail).
5. Cần dữ liệu gì? → Thử nghiệm ngẫu nhiên có nhóm đối chứng; cỡ mẫu theo `outputs/analysis_extensions/ab_test_sample_size.csv` (ví dụ d=10%: ~390–620 khách/nhóm với metric winsorize p99); giá vốn theo SKU.
