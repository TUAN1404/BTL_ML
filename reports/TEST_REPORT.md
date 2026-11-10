# BÁO CÁO KIỂM THỬ ĐỘC LẬP TOÀN DIỆN DỮ LIỆU VÀ PIPELINE
**Đồ án Học máy: Dự đoán giá bất động sản (King County & Indian Urban Housing)**  
*Đơn vị thực hiện: Agent Kiểm thử Độc lập (QA / Test Automation Lead)*  
*Thời điểm nghiệm thu: 09/11/2026*  
*Mã đối tượng kiểm tra: `prepare_data.py`, `d1/`, `d2/`, `check_data.py`, `data/processed/`*  

---

## 1. BẢNG TÓM TẮT TOÀN BỘ KẾT QUẢ KIỂM TRA

| Mã kiểm tra | Nội dung kiểm tra | Kết quả thực nghiệm | Đánh giá mức độ |
| :---: | :--- | :---: | :---: |
| **TC-REP-01** | Tính tái lập dữ liệu: SHA-256 hash của 4 file processed đồng nhất tuyệt đối qua 3 lần chạy lại từ đầu | **PASS** | Nghiêm ngặt (Tuyệt đối) |
| **TC-RAW-01** | Bảo toàn dữ liệu gốc: Các file trong `data/raw/` không bị sửa đổi, xóa hay ghi đè | **PASS** | Nghiêm ngặt |
| **TC-CLN-01** | Unit test hàm `clean_d1`: Loại bỏ 1 dòng `bedrooms == 33`, trích xuất `sale_year`, `sale_month`, `house_age`, `renovated`, giữ cột `id` | **PASS** | Nghiêm ngặt |
| **TC-CLN-02** | Unit test hàm `clean_d2`: Loại bỏ 33 dòng `Super_Area_SqFt < 100`, loại bỏ `Property_ID`, kiểm tra logic tầng và diện tích | **PASS** | Nghiêm ngặt |
| **TC-CLN-03** | Tính bất biến (Immutability): DataFrame đầu vào của `clean_d1` và `clean_d2` không bị sửa đổi (dùng `.copy()`) | **PASS** | Chuẩn mực |
| **TC-SPL-01** | Phân chia Bộ 1 (King County): `GroupShuffleSplit` theo `id`, 0 ID nào giao nhau giữa Train và Test | **PASS** | Chuẩn mực |
| **TC-SPL-02** | Phân chia Bộ 2 (Indian Urban): Xác nhận `test.csv` gốc thiếu target; `d2_train` và `d2_test` được chia chuẩn từ `train.csv` (80/20) | **PASS** | Chuẩn mực |
| **TC-SPL-03** | Tỉ lệ phân chia Train / Test: Bộ 1 đạt 19.99% test (trong biên 18-22%), Bộ 2 đạt đúng 20.00% test | **PASS** | Chuẩn mực |
| **TC-SPL-04** | Kiểm tra trùng lặp dòng chéo giữa Train và Test: Bộ 2 trùng 0 dòng; Bộ 1 chỉ trùng duy nhất 1 dòng đặc trưng (do 2 căn nhà song sinh có parcel ID khác nhau) | **PASS** (Kèm Cảnh báo) | Nhẹ (Miền dữ liệu) |
| **TC-SPL-05** | Nhất quán phân phối biến mục tiêu Train vs Test: Sai lệch mean, median, std đều < 3% trên cả 2 bộ dữ liệu | **PASS** | Chuẩn mực |
| **TC-INT-01** | Không còn giá trị khuyết thiếu: Tỉ lệ Missing = 0% trên cả 4 file processed | **PASS** | Chuẩn mực |
| **TC-INT-02** | Nhất quán schema: Tên cột, thứ tự cột và kiểu dữ liệu (dtypes) khớp 100% giữa Train và Test | **PASS** | Chuẩn mực |
| **TC-INT-03** | Loại bỏ hoàn toàn cột ID khỏi dữ liệu huấn luyện (`id`, `Property_ID`) | **PASS** | Chuẩn mực |
| **TC-INT-04** | Hợp lệ biến mục tiêu: Target luôn dương (> 0), phương sai lớn, không phải hằng số | **PASS** | Chuẩn mực |
| **TC-INT-05** | Quy ước Target: Chưa bị biến đổi log trong file processed (Bộ 1 max = \$7.7M, Bộ 2 max = 4,000 Lakhs) | **PASS** | Chuẩn mực |
| **TC-INT-06** | Quy ước Đặc trưng: Chưa bị scale chuẩn hóa trước (mean != 0, std != 1) | **PASS** | Chuẩn mực |
| **TC-INT-07** | Miền giá trị hợp lý Bộ 1: Phát hiện 13 dòng `bedrooms == 0`, 10 dòng `bathrooms == 0`, 12 dòng `house_age == -1` | **PASS** (Kèm Cảnh báo) | Cảnh báo miền giá trị |
| **TC-INT-08** | Miền giá trị hợp lý Bộ 2: Đạt chuẩn 100% (BHK, Bathrooms, diện tích, số tầng, tuổi nhà, cờ nhị phân) | **PASS** | Tuyệt đối |
| **TC-CFG-01** | Đầy đủ và nhất quán cấu hình: Khớp 100% các cột giữa `config_d1.py`, `config_d2.py` và file dữ liệu train | **PASS** | Chuẩn mực |
| **TC-CFG-02** | Không chồng lấn nhóm đặc trưng: Giao giữa `{TARGET}`, `CAT`, `ORD`, `NUM` rỗng | **PASS** | Chuẩn mực |
| **TC-CFG-03** | Không bỏ sót cột: 22/22 cột Bộ 1 và 17/17 cột Bộ 2 đều được phân nhóm rõ ràng | **PASS** | Chuẩn mực |
| **TC-CFG-04** | Đúng kiểu dữ liệu đặc trưng: NUM là kiểu số, CAT/ORD là chuỗi hoặc mã phân loại | **PASS** | Chuẩn mực |
| **TC-CFG-05** | Tính chính xác của biến thứ bậc ORD Bộ 2: `Locality_Type` và `Furnishing_Status` khớp đủ giá trị unique và thứ tự tăng dần logic | **PASS** | Chuẩn mực |
| **TC-CFG-06** | Độ phủ danh mục `zipcode` (Bộ 1): Đạt 70/70 mã ở cả train và test; 0 mã lạ ở test | **PASS** | Chuẩn mực |
| **TC-CFG-07** | Độ phủ danh mục `City` và `Property_Type` (Bộ 2): 10/10 thành phố và 4/4 loại hình BĐS; 0 nhãn lạ ở test | **PASS** | Chuẩn mực |
| **TC-LEA-01** | Rà soát rò rỉ dữ liệu (Data Leakage): Không có fit/transform trước pipeline, không get_dummies trước split, không target encoding, không lọc outlier thống kê trên toàn cục | **PASS** | Không phát hiện rò rỉ |
| **TC-SMK-01** | Smoke Test Bộ 1 (King County Ridge): Chạy thông suốt không lỗi, $R^2_{\text{test}} = 0.8719 \gg R^2_{\text{Dummy}} = -0.0639$, $\text{RMSLE} = 0.1855$ | **PASS** | Chuẩn mực |
| **TC-SMK-02** | Smoke Test Bộ 2 (Indian Urban Ridge): Chạy thông suốt không lỗi, $R^2_{\text{test}} = 0.7261 \gg R^2_{\text{Dummy}} = -0.0915$, $\text{RMSLE} = 0.3154$ | **PASS** | Chuẩn mực |
| **TC-SMK-03** | Độ bền vững với danh mục chưa từng thấy (Unseen Categories): Xử lý an toàn nhãn mới không crash mô hình | **PASS** | Bền vững |
| **TC-FLT-01** | Kiểm thử đột biến (Fault Injection): Đánh giá khả năng bắt lỗi của hệ thống kiểm tra đối với 6 ca tiêm lỗi | **PASS** | Đã phát hiện 3 lỗ hổng của check_data.py |
| **TC-UNK-01** | Đánh giá nhãn thực tế trên file `data/raw/test.csv` (Indian Urban) | **KHÔNG KIỂM TRA ĐƯỢC** | Thiếu nhãn từ ban tổ chức Kaggle |

---

## 2. DANH SÁCH LỖI VÀ ĐIỂM YẾU HỆ THỐNG THEO MỨC ĐỘ

Sau quá trình kiểm thử độc lập và tiêm lỗi đột biến, nhóm kiểm thử đã xác định được các lỗi và điểm yếu kỹ thuật trong mã nguồn dự án:

### Mức độ: TRUNG BÌNH (Medium)

#### 1. Lỗi crash `KeyError` trong hàm in bảng thống kê của `check_data.py`
- **Mô tả:** Trong script kiểm tra nhanh [check_data.py](file:///d:/Desktop/PTIT/N%C4%83m%204/MH/BTL/check_data.py), hàm `print_min_max_table()` lấy danh sách cột số từ `df_train` rồi truy cập trực tiếp vào `df_test[col]`. Nếu `df_train` chứa bất kỳ cột số nào mà `df_test` không có (ví dụ: trường hợp vô tình để sót cột `id` hoặc một feature mới ở tập train), chương trình bị văng lỗi crash `KeyError` không kiểm soát thay vì hoàn tất báo cáo kiểm định.
- **Bằng chứng thực nghiệm:**
  - File: [check_data.py#L38-L42](file:///d:/Desktop/PTIT/N%C4%83m%204/MH/BTL/check_data.py#L38-L42)
  - Traceback ghi nhận trong ca kiểm thử tiêm lỗi `test_case_3` và `test_case_5`:
    `KeyError: 'id'` / `KeyError: 'price_per_sqft'` tại dòng 40: `e_min = df_test[col].min()`.
- **Ảnh hưởng đến mô hình:** Không làm sai lệch trọng số mô hình (do đây là script kiểm tra phụ trợ), nhưng làm gián đoạn quy trình CI/CD tự động kiểm định dữ liệu.
- **Đề xuất khắc phục:** Trong [check_data.py](file:///d:/Desktop/PTIT/N%C4%83m%204/MH/BTL/check_data.py), nếu `not cols_match` thì dừng chương trình trả về `False` ngay; hoặc trong `print_min_max_table()` chỉ duyệt qua danh sách các cột chung:
  ```python
  common_num_cols = [c for c in num_cols if c in df_test.columns]
  ```

#### 2. Lỗ hổng bỏ sót kiểm tra trùng lặp dòng thực tế trong `check_data.py`
- **Mô tả:** [check_data.py](file:///d:/Desktop/PTIT/N%C4%83m%204/MH/BTL/check_data.py) (dòng 112-126) kiểm tra phân chia dữ liệu bằng cách đọc lại file raw `kc_house_data.csv` rồi chạy lại `GroupShuffleSplit`. Script **hoàn toàn không kiểm tra xem 2 file đã lưu trên đĩa (`d1_train.csv` và `d1_test.csv`) có bị trùng lặp dòng hay không**. Đối với Bộ 2, script không có bất kỳ bước kiểm tra trùng lặp nào. Khi tiêm lỗi chép 50 dòng từ test sang train, `check_data.py` vẫn báo `[PASS] 100%`.
- **Bằng chứng thực nghiệm:**
  - File: [check_data.py#L112-L128](file:///d:/Desktop/PTIT/N%C4%83m%204/MH/BTL/check_data.py#L112-L128).
  - Kết quả Fault Injection `test_case_4`: `check_data.py bắt được: False`.
- **Ảnh hưởng đến mô hình:** Nếu có lỗi xảy ra trong quá trình ghi file dẫn đến rò rỉ dữ liệu giữa train và test, `check_data.py` sẽ không thể cảnh báo cho người huấn luyện mô hình.
- **Đề xuất khắc phục:** Bổ sung kiểm tra giao dữ liệu trực tiếp vào `check_data.py`:
  ```python
  dups = pd.merge(df_train, df_test, how="inner")
  overlap_ok = len(dups) == 0 # len(dups) <= 1 cho Bộ 1
  all_pass &= print_status("Không có dòng trùng lặp chéo giữa Train và Test", overlap_ok, f"Số dòng trùng: {len(dups)}")
  ```

---

### Mức độ: NHẸ (Low)

#### 3. Lỗ hổng bỏ sót kiểm tra thang đo biến mục tiêu trong `check_data.py`
- **Mô tả:** [check_data.py#L92-L94](file:///d:/Desktop/PTIT/N%C4%83m%204/MH/BTL/check_data.py#L92-L94) chỉ kiểm tra `df_train["price"].min() > 0`. Khi biến mục tiêu bị biến đổi log trước (`np.log1p`), giá trị min vẫn $> 0$ ($11.23 > 0$), khiến `check_data.py` vẫn kết luận `[PASS]` mà không phát hiện được thang đo đã bị biến đổi.
- **Bằng chứng thực nghiệm:**
  - File: [check_data.py#L89-L99](file:///d:/Desktop/PTIT/N%C4%83m%204/MH/BTL/check_data.py#L89-L99).
  - Kết quả Fault Injection `test_case_6`: `check_data.py bắt được: False`.
- **Ảnh hưởng đến mô hình:** Nếu nhãn bị log trước rồi đưa vào pipeline tiếp tục log lần hai, mô hình sẽ bị biến dạng phân phối nghiêm trọng.
- **Đề xuất khắc phục:** Bổ sung điều kiện kiểm tra thang đo tối đa:
  ```python
  target_scale_ok = (df_train["price"].max() > 1000) and (df_test["price"].max() > 1000)
  all_pass &= print_status("Biến mục tiêu ở thang đo gốc (chưa bị log)", target_scale_ok)
  ```

#### 4. Cảnh báo chia cho 0 (`RuntimeWarning`) trong `clean_d2.py` khi nhận DataFrame rỗng
- **Mô tả:** Khi kiểm thử trường hợp biên với DataFrame rỗng (`initial_rows == 0`), hàm `clean_d2` phát sinh 2 cảnh báo chia cho 0: `RuntimeWarning: invalid value encountered in scalar divide` tại dòng 56 và dòng 77 khi tính tỉ lệ phần trăm vi phạm logic.
- **Bằng chứng thực nghiệm:**
  - File: [d2/clean_d2.py#L56](file:///d:/Desktop/PTIT/N%C4%83m%204/MH/BTL/d2/clean_d2.py#L56) và [d2/clean_d2.py#L77](file:///d:/Desktop/PTIT/N%C4%83m%204/MH/BTL/d2/clean_d2.py#L77).
- **Ảnh hưởng đến mô hình:** Không ảnh hưởng đến dữ liệu thật (dữ liệu thật có $\approx 80,000$ dòng), chỉ gây cảnh báo log ở ca kiểm thử biên.
- **Đề xuất khắc phục:** Thêm kiểm tra `if initial_rows == 0: return cleaned_df` ngay đầu hàm.

---

## 3. CÁC CẢNH BÁO CẦN NGƯỜI QUYẾT ĐỊNH (DOMAIN DECISIONS)

Đây là các hiện tượng đặc thù phản ánh bản chất của tập dữ liệu thực tế, không phải lỗi kỹ thuật tiền xử lý. Nhóm nghiên cứu cần thống nhất cách tiếp cận trước khi huấn luyện mô hình chính thức:

| Hiện tượng | Số lượng ghi nhận | Bản chất miền thực tế | Đề xuất lựa chọn xử lý |
| :--- | :---: | :--- | :--- |
| **`house_age == -1`** (Bộ 1) | **11 dòng** ở Train<br>**1 dòng** ở Test | Hợp đồng mua bán nhà hình thành trong tương lai (Pre-construction sales): giao dịch ký năm 2014 (`sale_year = 2014`) nhưng công trình hoàn công năm 2015 (`yr_built = 2015`). | **Lựa chọn 1 (Khuyến nghị):** Dùng `np.maximum(0, df["house_age"])` đưa về 0 (nhà mới xây).<br>**Lựa chọn 2:** Giữ nguyên giá trị -1 làm cờ phân loại nhà dự án.<br>**Lựa chọn 3:** Loại bỏ 12 dòng này. |
| **`bedrooms == 0`** (Bộ 1) | **11 dòng** ở Train<br>**2 dòng** ở Test | Căn hộ studio không phân chia phòng ngủ riêng biệt hoặc đất phụ trợ có nhà kho nhỏ. Giá bán từ \$280,000 đến \$1,295,650. | **Khuyến nghị:** Giữ nguyên giá trị 0 vì là bất động sản có thực và có giá trị giao dịch cao. |
| **`bathrooms == 0`** (Bộ 1) | **8 dòng** ở Train<br>**2 dòng** ở Test | Nhà thô, đất nền hoặc bất động sản thương mại/kho phụ trợ chưa hoàn thiện tiện ích vệ sinh. | **Khuyến nghị:** Giữ nguyên giá trị 0. |
| **1 dòng trùng đặc trưng giữa Train và Test** (Bộ 1) | **1 dòng** duy nhất | Hai căn nhà phố liền kề song sinh (Twin Townhouses) tại Seattle, xây cùng năm, chung kiến trúc và thông số, nhưng có mã định danh thửa đất khác nhau (`parcel id = 7899800860` và `7899800864`). | **Khuyến nghị:** Giữ nguyên. GroupShuffleSplit đã tách đúng 100% theo ID thửa đất; việc trùng thông số vật lý là bình thường đối với các dự án nhà liền kề. |
| **Đa cộng tuyến cao** (`sqft_living`, `Super_Area_SqFt`) | Bộ 1: $sqft\_living = sqft\_above + sqft\_basement$<br>Bộ 2: $\text{corr}(Super, Carpet) > 0.95$ | Mối quan hệ toán học nội tại giữa các đặc trưng diện tích. | Thành viên huấn luyện mô hình tuyến tính (OLS) cần dùng Ridge/Lasso điều quy L1/L2 hoặc bỏ bớt 1 cột để tránh suy biến ma trận hiệp phương sai. |

---

## 4. CÁC MỤC KHÔNG KIỂM TRA ĐƯỢC VÀ LÝ DO

1. **Nhãn thực tế của tập `data/raw/test.csv` (Bộ 2 - Indian Urban Housing):**
   - **Lý do:** Đây là tập kiểm thử kín (Private Test Set) của cuộc thi Kaggle (*Indian Urban House Price Prediction*). File gốc hoàn toàn không có cột nhãn `Price_INR_Lakhs`.
   - **Biện pháp đã xử lý của nhóm:** Người 1 đã đưa ra quyết định đúng đắn khi cô lập file `test.csv` gốc này và chủ động chia tập `train.csv` (79,967 dòng sau làm sạch) thành hai tập độc lập `d2_train.csv` (63,973 dòng) và `d2_test.csv` (15,994 dòng) theo tỉ lệ 80/20 với `random_state=42`. Toàn bộ quá trình kiểm thử đánh giá mô hình được nghiệm thu trên tập `d2_test.csv` tự tạo này.

---

## 5. TỔNG KẾT KẾT QUẢ THỰC THI PYTEST TOÀN DỰ ÁN

Lệnh thực thi kiểm định tổng thể: `pytest -v`
```text
============================= test session starts =============================
platform win32 -- Python 3.13.7, pytest-9.2.0, pluggy-1.7.0
rootdir: D:\Desktop\PTIT\Năm 4\MH\BTL
collected 95 items

tests/test_audit_independent.py ......................                   [ 23%]
tests/test_clean.py ................                                     [ 40%]
tests/test_config.py ............                                        [ 52%]
tests/test_data_integrity.py ........                                    [ 61%]
tests/test_data_pipeline.py .................                            [ 78%]
tests/test_fault_injection.py ......                                     [ 85%]
tests/test_reproducibility.py .                                          [ 86%]
tests/test_smoke.py ....                                                 [ 90%]
tests/test_split.py .........                                            [100%]

======================= 95 passed, 2 warnings in 55.18s =======================
```
- Tổng số ca kiểm thử tự động: **95/95 ca PASSED (Tỉ lệ đạt 100%)**.
- Thời gian thực thi toàn bộ: **55.18 giây**.

---

## 6. KẾT LUẬN NGHIỆM THU CUỐI CÙNG

> **KẾT LUẬN:** Dữ liệu hoàn toàn **CÓ THỂ BÀN GIAO NGAY** cho các thành viên trong nhóm để triển khai xây dựng và huấn luyện mô hình học máy (Người 2: Cây quyết định / Rừng ngẫu nhiên; Người 3: Boosting / Mạng nơ-ron), do dữ liệu trong `data/processed/` đạt 100% tính toàn vẹn, hoàn toàn không bị rò rỉ (0% data leakage), có khả năng tái lập tuyệt đối (hash SHA-256 đồng nhất qua mọi lần chạy), và các khiếm khuyết được phát hiện chỉ giới hạn ở script kiểm tra phụ trợ `check_data.py` chứ không ảnh hưởng đến chất lượng dữ liệu bàn giao.

