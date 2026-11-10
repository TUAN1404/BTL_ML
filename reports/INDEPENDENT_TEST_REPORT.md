# BÁO CÁO KIỂM THỬ ĐỘC LẬP (INDEPENDENT AUDIT REPORT)
**Dự án:** Đồ án môn Học máy (PTIT) — Dự đoán giá bất động sản  
**Bộ dữ liệu kiểm định:** 
1. King County House Sales (`kc_house_data.csv`)
2. Indian Urban Housing (`train.csv`, `test.csv`)  
**Công cụ kiểm thử tự động:** `pytest` (22 test cases tại `tests/test_audit_independent.py`)  
**Thời gian thực hiện kiểm định:** 2026-11-09  
**Người thực hiện:** Agent Kiểm thử Độc lập (Independent QA Agent)

---

## 1. TỔNG QUAN KẾT QUẢ KIỂM THỬ

* **Tổng số ca kiểm thử tự động (Test Cases):** **22 tests**
* **Kết quả thực thi:** **22 PASSED (100% ĐẠT)** trong 30.40 giây.
* **Số lỗi nghiêm trọng (Làm sai kết quả mô hình):** **0**
* **Số điểm cần lưu ý (Mức Trung bình / Nhẹ):** **2** (liên quan đến `house_age` âm và tương tác CLI `input()`).

---

## 2. BẢNG ĐỐI CHIẾU QUY ƯỚC DỰ ÁN & BẰNG CHỨNG THỰC TẾ

| Hạng mục kiểm tra | Quy ước bắt buộc | Kết quả kiểm tra | Bằng chứng thực nghiệm (File, Số liệu, Dòng code) |
|---|---|:---:|---|
| **1. Tính toàn vẹn dữ liệu gốc** | Không sửa đổi bất kỳ file nào trong `data/raw/` | **PASS** | File `kc_house_data.csv`: (21613, 21); `train.csv`: (80000, 18); `test.csv`: (20000, 17). Không bị thay đổi byte hay số dòng. |
| **2. Tỉ lệ phân chia Train / Test** | 80% Train / 20% Test, cố định `random_state=42` | **PASS** | `d1_train`: 17,291 dòng (80.01%), `d1_test`: 4,321 dòng (19.99%).<br>`d2_train`: 63,973 dòng (80.00%), `d2_test`: 15,994 dòng (20.00%). |
| **3. Cô lập ID Bộ 1 (Chống rò rỉ)** | Dùng `GroupShuffleSplit` theo `id`, 0 ID bị lặp giữa train và test | **PASS** | `prepare_data.py` dòng 48–56: `set(d1_train_id).intersection(d1_test_id) == set()` (0 ID trùng). |
| **4. Xử lý ngoại lai King County** | Chỉ loại bỏ duy nhất 1 dòng ngoại lai lỗi gõ phím `bedrooms == 33` | **PASS** | `d1/clean_d1.py` dòng 33–39: Loại bỏ đúng bản ghi index 15870 (`id = 2402100895`, 33 phòng/1620 sqft). Dữ liệu giảm từ 21,613 xuống 21,612 dòng. |
| **5. Xử lý ngoại lai Indian Urban** | Loại bỏ 33 dòng diện tích cực nhỏ `< 100 sqft` | **PASS** | `d2/clean_d2.py` dòng 42–49: Loại bỏ 33 dòng có `Super_Area_SqFt < 100` (nhỏ nhất 16.02 sqft). Dữ liệu giảm từ 80,000 xuống 79,967 dòng. |
| **6. Loại bỏ cột định danh** | Không để cột ID trong tập train và test sau cùng | **PASS** | `d1_train`, `d1_test` không có cột `id`; `d2_train`, `d2_test` không có cột `Property_ID`. |
| **7. Trạng thái tiền xử lý (Leakage Check)** | CHƯA làm One-Hot, Scale, Impute, Log1p | **PASS** | Các cột `zipcode`, `City`, `Property_Type`, `Furnishing_Status` giữ nguyên dạng số nguyên hoặc chuỗi gốc (`object`); `price` và `Price_INR_Lakhs` chưa bị log1p. |
| **8. Bảo toàn biến mục tiêu** | Target giữ nguyên gốc, không có giá trị $\le 0$ | **PASS** | `d1` target `price`: min \$75,000, max \$7,700,000.<br>`d2` target `Price_INR_Lakhs`: min 2.77 Lakhs, max 3,334.92 Lakhs. |
| **9. Tính bất biến của hàm Clean** | `clean_d1(df)` và `clean_d2(df)` không sửa df đầu vào | **PASS** | Cả 2 hàm đều gọi `df.copy()`. Đã kiểm định bằng `pd.testing.assert_frame_equal`. |
| **10. Tính đồng bộ Cấu hình (Config)** | `config_d1.py` và `config_d2.py` khớp 100% với file CSV | **PASS** | Cột trong `d1` (22 cột) và `d2` (17 cột) khớp hoàn toàn với `TARGET + CAT + NUM + ORD`. |

---

## 3. DANH SÁCH CÁC PHÁT HIỆN & PHÂN LOẠI MỨC ĐỘ RỦI RO

### Vấn đề 1: Tồn tại 12 bản ghi có `house_age = -1` trong Bộ 1
* **Mức độ rủi ro:** **TRUNG BÌNH (MEDIUM)**
* **Vị trí code:** `d1/clean_d1.py`, dòng 47–57
* **Bằng chứng:** Trong dữ liệu King County có 12 căn nhà bán vào năm 2014 nhưng năm xây dựng ghi 2015 (nhà bán trước khi hoàn thiện / pre-construction). Khi tính `house_age = sale_year - yr_built`, kết quả ra `-1`.
* **Đánh giá ảnh hưởng:**
  - `clean_d1.py` tuân thủ đúng yêu cầu chỉ cảnh báo chứ không tự ý sửa dữ liệu thô.
  - Đối với các mô hình cây (Random Forest, XGBoost), giá trị `-1` không gây lỗi tính toán mà được coi như một giá trị rời rạc riêng biệt.
  - Tuy nhiên, đối với các mô hình tuyến tính (Linear, Ridge, Lasso) hoặc khi lấy $\ln$, tuổi âm không có ý nghĩa vật lý.
* **Khuyến nghị cho Pipeline:** Người phụ trách Pipeline (Người 3) cần thêm bước `house_age = np.maximum(0, house_age)` trong transformer của Pipeline.

---

### Vấn đề 2: Lệnh `input()` tương tác trong `prepare_data.py`
* **Mức độ rủi ro:** **NHẸ (LOW)**
* **Vị trí code:** `prepare_data.py`, dòng 99
* **Bằng chứng:**
  ```python
  if has_target_in_test:
      choice = input("Bạn muốn dùng test.csv chính thức hay tự chia 80/20 từ train.csv?...")
  ```
* **Đánh giá ảnh hưởng:**
  - Hiện tại `test.csv` của Kaggle không có cột giá (`has_target_in_test == False`), nên nhánh code này không bị kích hoạt và script chạy tự động trơn tru.
  - Tuy nhiên, nếu sau này có file test có nhãn, lệnh `input()` có thể làm treo các tiến trình chạy tự động (CI/CD hoặc script batch).
* **Khuyến nghị:** Nên ưu tiên truyền cờ (flag argument như `--use-official-test`) thay vì `input()` tương tác qua console.

---

## 4. KẾT LUẬN CỦA AGENT KIỂM THỬ ĐỘC LẬP

1. **Chất lượng dữ liệu và mã nguồn:** Hệ thống chuẩn bị dữ liệu (`prepare_data.py`, `clean_d1.py`, `clean_d2.py`, `check_data.py`, `config_d1.py`, `config_d2.py`) đạt tiêu chuẩn kỹ thuật cao, tuân thủ nghiêm ngặt nguyên tắc chống rò rỉ dữ liệu (No Data Leakage).
2. **Độ sẵn sàng:** 4 file dữ liệu `data/processed/d1_train.csv`, `d1_test.csv`, `d2_train.csv`, `d2_test.csv` hoàn toàn đạt chuẩn và **sẵn sàng 100% để cả 3 thành viên trong nhóm nạp vào Pipeline huấn luyện mô hình**.

