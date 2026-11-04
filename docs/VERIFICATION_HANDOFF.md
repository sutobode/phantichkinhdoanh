# Verification & Handoff — BA Capstone (Customer Value & Promotion Targeting)

Tài liệu này dành cho thành viên nhóm và cho **một phiên Claude trên máy khác** cần
kiểm tra lại kết quả mà không có ngữ cảnh cuộc trò chuyện trước. Mọi lệnh đều chạy
từ thư mục gốc repo.

Cập nhật: 2026-11-04 · Nhánh: `hung` · Commit gốc: `5dbc99514a0eb131733d518dc7ef1870c0b9848b`

---

## 0. Tóm tắt trạng thái

| Hạng mục | Trạng thái |
|---|---|
| `BA-report.pdf` (22 trang) tái tạo được từ raw data | **78/78 con số khớp** (Windows Py3.11.16 và Linux server Py3.11.18) |
| Pipeline nhánh `hung` (`06_final_qa.py`) | **PASSED, 48/48 test** khi chạy trong git checkout |
| Metrics model so với bản đã commit | Trùng tới từng chữ số (chỉ cột thời gian chạy khác) |
| Kết luận chính của báo cáo ("không launch 3 chiến dịch") | **Đúng**, nhưng là hệ quả *số học* của giả định (xem §3.1) |
| Nhánh `master` | **Lỗi công thức margin** + tài liệu mâu thuẫn với báo cáo — KHÔNG dùng để nộp |

---

## 1. Bối cảnh

- Đề bài: `phantichkinhdoanh.pdf` (HUST BA Capstone 2026), đề số 1.
- Báo cáo nộp: `BA-report.pdf` (tác giả: Nguyen Quoc Hung, Bui Trung Kien, Nguyen Minh Thuong).
- **Code tạo ra báo cáo nằm ở nhánh `hung`**, không phải `master`:

| | `master` | `hung` (khớp báo cáo) |
|---|---|---|
| Thiết kế | Cutoff 2011-06-30, outcome ~5,3 tháng, random split | Snapshot T, horizon 90 ngày, train/val/test theo thời gian |
| Nguồn dự báo cho quyết định | GBM | Linear Regression (chọn trước) |
| Công thức margin | `v·g·[(1+r)(1−d) − 1]` ❌ (ngầm giả định giá vốn giảm theo giá bán) | `v·[(1+r)(g−d) − g] − c` ✅ |
| Kết luận | Target 2.072 khách, +£22.787 | Không launch 3 chiến dịch cấu hình |

---

## 2. Đã kiểm chứng những gì

### 2.1 Script kiểm chứng
| Script | Làm gì | Output |
|---|---|---|
| `src/06_final_qa.py` (có sẵn) | Chạy lại 02→03→05→07 + pytest + invariant + manifest | `outputs/final_qa_report.md`, `outputs/run_manifest.json` |
| `src/04_advanced_models.py` (có sẵn) | GBM, calibration, SHAP, rolling-origin | `outputs/temporal_model_comparison.csv`, ... |
| `src/08_verify_ba_report.py` (**mới**) | Dựng lại độc lập phương pháp báo cáo từ raw data, so **78 con số** với báo cáo; thêm robustness gross/net + Monte Carlo | `outputs/ba_report_verification.csv`, `outputs/robustness/ba_*.csv` |
| `src/09_scenario_extensions.py` (**mới**) | Kịch bản neo ngưỡng hòa vốn, replay trên doanh số thật, ablation nguồn dự báo | `outputs/scenario_extensions/*.csv` |
| `scripts/remote_ba.py` (**mới**) | SSH helper chạy trên server trong `/home/darkweb/BA` | — |

### 2.2 Kết quả chính (giá trị kỳ vọng khi chạy lại)

| Mục báo cáo | Giá trị | Dung sai |
|---|---|---|
| Raw rows / thiếu Customer ID / trùng | 1.067.371 / 243.007 / 34.335 | chính xác |
| SHA-256 workbook | `bcbe73b3…2df2e980` | chính xác |
| Khách final test / repeat rate | 5.281 / 43,36% | chính xác |
| AP Logistic / GBM (final test) | 0,7502 / 0,7691 | ±0,002 |
| MAE Linear / GBM (final test) | £395,69 / £400,40 | ±£1 |
| Max α Conservative / Moderate / Aggressive | −0,0045 / −0,0220 / −0,1090 | chính xác |
| RFM-based 1.500 khách (Moderate) | −£118.314 | ±1% |
| Random 1.500 khách (Moderate) | −£37.408 | ±1% (phụ thuộc RNG) |
| Proposed (cả 3 kịch bản) | 0 khách | chính xác |
| Sensitivity d=5%, b=20% | 1.500 khách, £7.175,92 | ±£1 |
| Retain / Nurture (Moderate) | 2.454 / 0 | chính xác |
| SHAP top-2 (GBM classifier) | frequency 0,641 > recency 0,573 | ±0,01 |
| AP / ROC GBM (validation) | 0,7189 / 0,8150 | ±0,0002 (Linux ra 0,7188 / 0,8149) |

### 2.2b Tái lập chéo nền tảng (đã chạy 2026-11-04)

| Môi trường | 06 Final QA | 08 (78 con số) | So với output đã commit |
|---|---|---|---|
| Windows 11, Py 3.11.16 | PASSED, 48/48 | 78/78 | Trùng từng byte, trừ cột thời gian chạy |
| Ubuntu 24.04 server, Py 3.11.18 | PASSED, 48/48 | 78/78 | Lệch số thực ≤1e-12 (Linear, scenario, break-even); phần GBM lệch ≤0,6% tương đối (SHAP, ECE, Brier, guardrail, số khách mỗi bin calibration). Quyết định chọn model, thứ tự SHAP, mọi con số làm tròn trong báo cáo (trừ GBM validation ±0,0001) giữ nguyên |

`09_scenario_extensions.py` cho kết quả giống nhau giữa hai máy (chênh ≤1e-9).
`run_manifest.json`: `config_sha256` và `requirements_sha256` khác nhau giữa
Windows (CRLF) và Linux (LF) dù nội dung giống hệt — xem §3 mục 5.

### 2.3 Robustness trả lời 2 phản biện (phương pháp báo cáo)

**Phản biện 1 — "dùng gross thay vì net"** (`outputs/robustness/ba_gross_vs_net.csv`):
- 3 kịch bản cấu hình: gross, net, và gross×(1−return share) **đều chọn 0 khách** (α<0 ⇒ ΔM<0 với mọi v≥0).
- Kịch bản có lãi: danh sách chọn theo net trùng **97–99%** với gross; margin thấp hơn 9–16%.
- Cancellation chiếm 5,8–7,0% giá trị gross; dòng không phải sản phẩm (POST, M, …) 1,7%.

**Phản biện 2 — "hệ số tự giả định"** (`outputs/robustness/ba_monte_carlo.csv`, 5.000 lần rút, multipliers ×U(0.5,1.5)):

| | linked (growth bám discount) | independent (stress) |
|---|---|---|
| P(policy chọn 0 khách) | 83% | 71% |
| P(random 1.500 khách lỗ) | 100% | 97% |
| P(proposed ≥ random) | 100% | 100% |
| Median margin khi có chọn | £1.354 | £4.572 |

### 2.4 Kịch bản mở rộng & vai trò của model (`outputs/scenario_extensions/`)

Kịch bản minh họa neo ngay trên ngưỡng hòa vốn của At-risk (trùng các điểm Section 8.2):

| Kịch bản (d, b) | Proposed (dự báo) | Random | RFM | Proposed trên doanh số thật | % oracle |
|---|---|---|---|---|---|
| light (5%, 20%) | +£7.176 | −£5.789 | −£25.942 | +£11.469 | 79% |
| standard (10%, 35%) | +£3.991 | −£22.457 | −£80.526 | +£5.795 | 71% |
| deep (15%, 50%) | +£962 | −£44.108 | −£147.705 | +£1.158 | 55% |

**Ablation nguồn dự báo** (cùng rule, chỉ đổi v; chấm bằng doanh số thật, % oracle):

| Kịch bản | Không model (v hằng số) | Monetary lịch sử | **Linear (báo cáo)** | GBM |
|---|---|---|---|---|
| light | 64% | 65% | **79%** | 78% |
| standard | 81% | 79% | **71%** | 80% |
| deep | 82% | 73% | **55%** | 67% |

⚠️ **Diễn giải trung thực**: phần lớn lợi thế của Proposed so với Random/RFM đến từ
**quy tắc segment (giả định multiplier)**, không phải từ model. Model chỉ tạo giá trị
rõ khi ngân sách thực sự ràng buộc (light: 2.201 khách đủ điều kiện > K=1.500). Ở
standard/deep, Linear **kém hơn cả không-model** vì 30% dự báo bị âm rồi clip về 0
(những khách này thực tế có mua nhưng bị loại).

---

## 3. Phát hiện cần xử lý

1. **Kết luận chính là hệ quả số học của giả định.** α = (1+b·m)(g−d) − g < 0 cho cả
   12 tổ hợp kịch bản × segment ⇒ chọn 0 khách bất kể dữ liệu/model. Ngưỡng cần:
   b·m > d/(g−d) = 14% / 33% / 100% cho d = 5% / 10% / 20%. Ba kịch bản giữ nguyên
   số từ `master` (thiết kế cho công thức sai) nên đều nằm dưới ngưỡng.
2. **Model hồi quy giá trị yếu** cho dữ liệu zero-inflated (56,6% khách có doanh số
   tương lai = 0; 30% dự báo Linear âm). Ablation §2.4 cho thấy hệ quả.
3. **Calibration xấu đi ở final test** (ECE 0,12–0,13) — ảnh hưởng nhỏ vì xác suất không
   vào công thức quyết định; nên nói rõ.
4. **Repo**: `master` mâu thuẫn báo cáo; `outputs/shap_feature_importance.csv` trên `hung`
   là file cũ từ `master` (ghi recency đứng đầu — sai so với code hiện tại);
   `outputs/run_manifest.json` đã commit sinh từ commit cũ `f2164e8`.
5. **CRLF**: `core.autocrlf=true` trên Windows làm `config_sha256` trong manifest khác
   giữa Windows và Linux (nội dung giống hệt). Đề xuất thêm `.gitattributes`
   (`* text=auto eol=lf`).

---

## 4. Việc cần làm để chốt kết luận

| # | Việc | Ai | Mức | Điều kiện "xong" |
|---|---|---|---|---|
| 1 | Commit `src/08_verify_ba_report.py`, `src/09_scenario_extensions.py`, `scripts/remote_ba.py`, tài liệu này lên `hung` và push | Kien | Bắt buộc | Máy khác `git pull` là có đủ script |
| 2 | Báo cáo Section 7.3: thêm bảng §2.4 (kịch bản minh họa + replay) làm 7.3b | Thuong | Cao | Có ≥1 bảng mà Proposed ≠ 0 và so được Random/RFM |
| 3 | Báo cáo Section 7.3c/8: thêm ablation và **diễn giải trung thực** (segment rule tạo phần lớn giá trị; model giúp khi ngân sách ràng buộc) | Kien | Cao | Không còn claim "model tạo ra lợi thế" chưa có số |
| 4 | Abstract/Conclusion: đổi thành quy tắc có điều kiện — "chỉ launch khi growth kỳ vọng > d/(g−d)/m_z; khi đó target theo rule" | Cả nhóm | Cao | Câu kết luận nêu ngưỡng hòa vốn |
| 5 | Section 8: thêm 8.5 gross/net + Monte Carlo (§2.3) | Hung | Trung bình | Bảng §2.3 có trong báo cáo |
| 6 | Table 7: thêm cột "growth cần đạt" 14%/33%/100% | Thuong | Thấp | — |
| 7 | Dọn repo: xóa `outputs/shap_feature_importance.csv`; commit manifest mới; chọn `hung` làm nhánh nộp (merge vào `master` hoặc ghi rõ trong README); thêm `.gitattributes` | Kien | Trung bình | `git status` sạch sau `06_final_qa.py` (trừ registry/manifest/timing) |
| 8 | (Tuỳ chọn) Mô hình hai phần P(mua)×E[sales∣mua] làm nguồn so sánh trong ablation | Kien | Thấp | Thêm 1 dòng vào ablation; KHÔNG đổi nguồn chính |
| 9 | Chạy lại `06` → `04` → `08` → `09` sau mọi thay đổi; xác nhận §2.2 vẫn đúng | Người sửa cuối | Bắt buộc | 78/78 khớp, 48 test pass |

---

## 5. Tái lập trên máy khác (Windows hoặc Linux)

### 5.1 Chuẩn bị
```bash
git clone <repo-url> phantichkinhdoanh && cd phantichkinhdoanh
git switch hung                      # BẮT BUỘC: code của báo cáo ở nhánh này
# Nếu việc #1 (§4) chưa làm: copy src/08_*.py, src/09_*.py, scripts/remote_ba.py,
# BA-report.pdf từ server /home/darkweb/BA/phantichkinhdoanh/ (xem §6)
```
Python **3.11** bắt buộc (một test kiểm tra `python_version.startswith("3.11.")`).
```bash
uv venv .venv --python 3.11          # hoặc: py -3.11 -m venv .venv
uv pip install --python .venv/bin/python -r requirements.txt   # Windows: .venv/Scripts/python.exe
```

### 5.2 Dữ liệu
```bash
curl -L --retry 5 -o data/raw/online_retail_ii.zip https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip
unzip -tq data/raw/online_retail_ii.zip && unzip -o data/raw/online_retail_ii.zip -d data/raw/
sha256sum data/raw/online_retail_II.xlsx   # phải = bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980
```
Server UCI hay ngắt kết nối và **không hỗ trợ resume** (`curl -C -` lỗi 33) — nếu zip
hỏng thì xóa và tải lại từ đầu.

### 5.3 Chạy (thứ tự bắt buộc)
```bash
export PYTHONIOENCODING=utf-8        # Windows console
python src/06_final_qa.py            # ~3 phút; kỳ vọng "48 passed" + "FINAL QA PASSED"
python src/04_advanced_models.py     # ~1,5 phút
python src/08_verify_ba_report.py    # ~1,5 phút; kỳ vọng "78/78 khớp"
python src/09_scenario_extensions.py # vài giây; cần output của 03 và 04
```

### 5.4 Checklist chấp nhận
- [ ] `outputs/final_qa_report.md` ghi `pytest: PASSED`, `Invariant checks: PASSED`.
- [ ] `outputs/ba_report_verification.csv`: cột `match` toàn `True` (78 dòng).
- [ ] Windows: `git diff outputs/temporal_model_comparison.csv` chỉ khác cột `fit_seconds`, `inference_ms_per_1000`.
      Linux: ~22 file output hiện `M` — chạy `python <csvdiff>` (xem §5.5) để xác nhận lệch chỉ ở mức ≤1e-12 (phần Linear/scenario) và ≤0,6% (phần GBM).
- [ ] `outputs/scenario_extensions/prediction_source_ablation.csv` khớp bảng §2.4 (±1 điểm %).

### 5.5 Bẫy đã gặp
| Hiện tượng | Nguyên nhân | Xử lý |
|---|---|---|
| `test_manifest_hashes_stable…` FAIL: `git rev-parse HEAD` exit 128 | Chạy ngoài git repo (vd giải nén từ `git archive`) | Chạy trong `git clone`/checkout thật |
| Mọi file hiện `M` trong git status trên Linux | File copy từ Windows có CRLF | `git checkout -- .` |
| Log 03 có dòng MAE 396,54 thay vì 395,69 | Dòng đó do `07_duplicate_robustness` gọi lại hàm của 03 trên dữ liệu đã bỏ trùng | Không phải lỗi — kết quả chính vẫn 395,69 |
| Log 03 có dòng PR-AUC = 1.0, MAE ~999.939 | pytest ghi dữ liệu synthetic vào cùng file log | Lọc dòng có `n_evaluation': 5281` |
| `08` lỗi `clip_nonneg` / `future_value` KeyError | Chạy bản 08 cũ viết cho `master` | Dùng bản trên nhánh `hung` (đã hỗ trợ `future_gross_sales`) |
| `outputs/shap_feature_importance.csv` ghi recency đứng đầu | File cũ từ `master`, code hiện tại không ghi file này | SHAP đúng nằm trong log 04 / `advanced_feature_importance.csv` |
| Linux: nhiều file output hiện `M` sau khi chạy | Sai khác dấu phẩy động giữa nền tảng (GBM nhạy nhất) | So từng cột số giữa `git show HEAD:<file>` và file mới; script mẫu ở server: `/home/darkweb/BA/.tmp/csvdiff.py` (chạy từ thư mục repo) |
| Manifest khác `config_sha256` giữa máy | CRLF (Windows `core.autocrlf=true`) vs LF | Không phải thay đổi nội dung; sửa gốc bằng `.gitattributes` (việc #7) |

---

## 6. Chạy trên server dùng chung

- Server: thông tin trong biến môi trường **`MY_SERVER_INFOR`** dạng `ip/username/password`
  (không bao giờ ghi mật khẩu ra file/log/commit).
- **Phạm vi được phép: chỉ `/home/darkweb/BA`.** Không đụng service khác (server đang chạy
  clickhouse, nfcapd, vector… trong `~/idslab`), không cài gì ra ngoài thư mục này, không dùng sudo.
- `scripts/remote_ba.py` tự đặt `HOME`, cache (`XDG_*`, `UV_*`, `PIP_CACHE_DIR`,
  `MPLCONFIGDIR`, `NUMBA_CACHE_DIR`) và `TMPDIR` vào trong `/home/darkweb/BA`, và từ chối
  upload/download ngoài thư mục đó.

```bash
pip install paramiko
python scripts/remote_ba.py run "ls -la"                      # cwd = /home/darkweb/BA
python scripts/remote_ba.py run "cd phantichkinhdoanh && git log --oneline -1"
python scripts/remote_ba.py run "cd phantichkinhdoanh && .venv/bin/python src/08_verify_ba_report.py > outputs/logs/check_08.txt 2>&1; grep 'khớp)' outputs/logs/check_08.txt"
python scripts/remote_ba.py get phantichkinhdoanh/outputs/ba_report_verification.csv ./ba_report_verification_server.csv
```

Bố cục trên server:
```
/home/darkweb/BA/
├── .uv/            # uv binary + Python 3.11 (managed) + cache
├── .home/ .tmp/    # HOME/TMPDIR giả cho mọi lệnh
├── hung.bundle     # git bundle nhánh hung (nguồn của .git trên server)
└── phantichkinhdoanh/   # git repo (nhánh hung) + .venv + data/raw + outputs
```
Kiểm tra không ghi ra ngoài:
`find /home/darkweb -maxdepth 4 -newer /home/darkweb/BA/.home -not -path '/home/darkweb/BA*'`
— các file trong `~/idslab` là của service đang chạy, không phải của pipeline.

---

## 7. Prompt mẫu cho Claude ở máy khác

> Repo `phantichkinhdoanh`, nhánh `hung`. Đọc `docs/VERIFICATION_HANDOFF.md` trước.
> Tạo venv Python 3.11, tải dữ liệu theo §5.2 (kiểm tra SHA-256), chạy §5.3 theo đúng
> thứ tự, rồi đối chiếu checklist §5.4 và bảng §2.2/§2.4. Báo cáo mọi sai lệch vượt dung
> sai kèm số liệu. Không sửa `config/params.yaml` hay pipeline 02→07. Nếu chạy trên server
> dùng `scripts/remote_ba.py` và CHỈ làm việc trong `/home/darkweb/BA` (§6).
