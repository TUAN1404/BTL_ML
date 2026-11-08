# Hướng dẫn & Tài liệu Dữ liệu (DATA_README)

Tài liệu mô tả 4 tập dữ liệu đã làm sạch sơ bộ tại thư mục `data/processed/`, dùng chung cho toàn bộ các thành viên trong nhóm làm đồ án môn Học máy (PTIT).

---

## 1. Danh sách 4 file dữ liệu (`data/processed/`)

| File | Số dòng | Số cột | Cột mục tiêu (Target) | Ý nghĩa |
|---|---:|---:|:---:|---|
| [`d1_train.csv`](file:///d:/Desktop/PTIT/Năm%204/MH/BTL/data/processed/d1_train.csv) | **17,291** | **22** | `price` (USD) | Tập huấn luyện & Cross-Validation Bộ 1 (King County) |
| [`d1_test.csv`](file:///d:/Desktop/PTIT/Năm%204/MH/BTL/data/processed/d1_test.csv)   | **4,321**  | **22** | `price` (USD) | Tập kiểm thử đánh giá cuối cùng Bộ 1 |
| [`d2_train.csv`](file:///d:/Desktop/PTIT/Năm%204/MH/BTL/data/processed/d2_train.csv) | **63,973** | **17** | `Price_INR_Lakhs` | Tập huấn luyện & Cross-Validation Bộ 2 (Indian Urban) |
| [`d2_test.csv`](file:///d:/Desktop/PTIT/Năm%204/MH/BTL/data/processed/d2_test.csv)   | **15,994** | **17** | `Price_INR_Lakhs` | Tập kiểm thử đánh giá cuối cùng Bộ 2 |

---

## 2. Các bước ĐÃ LÀM và CHƯA LÀM

### ĐÃ LÀM (Làm sạch logic & trích xuất đặc trưng cơ bản)
* **Bộ 1 (King County):**
  - Loại bỏ cột định danh `id`.
  - Loại bỏ đúng **1 dòng ngoại lai** lỗi gõ phím: `bedrooms == 33` nhưng diện tích chỉ $1,620\text{ sqft}$ (`id = 15870`).
  - Parse chuỗi thời gian `date` thành `sale_year` và `sale_month`.
  - Tính đặc trưng `house_age = sale_year - yr_built` (có 12 căn bán trên giấy nên `house_age = -1`).
  - Tạo cờ nhị phân `renovated = (yr_renovated > 0)` và bỏ `yr_renovated`, `date`.
  - *Tổng số dòng loại bỏ:* **1 dòng** ($21,613 \rightarrow 21,612$).
* **Bộ 2 (Indian Urban):**
  - Loại bỏ cột định danh `Property_ID`.
  - Loại bỏ đúng **33 dòng ngoại lai** diện tích cực nhỏ: `Super_Area_SqFt < 100` (nhỏ nhất là $16.02\text{ sqft} \approx 1.5\text{ m}^2$).
  - Kiểm tra logic số tầng (`Floor_Number <= Total_Floors`) và diện tích lọt lòng (`Carpet <= Super_Area`): $0$ vi phạm.
  - *Tổng số dòng loại bỏ:* **33 dòng** ($80,000 \rightarrow 79,967$).

### CHƯA LÀM (Bắt buộc để trong Scikit-Learn Pipeline)
Để tránh rò rỉ dữ liệu (**Data Leakage**), các bước sau **chưa được thực hiện** trên các file CSV này mà phải nằm trong Pipeline và chỉ được `fit` trên tập train:
1. **Chưa One-Hot / Ordinal Encoding:** Biến danh mục vẫn ở dạng chuỗi gốc hoặc mã vùng (ví dụ `City`, `Property_Type`, `Locality_Type`, `zipcode`).
2. **Chưa Feature Scaling:** Chưa áp dụng `StandardScaler` hay `MinMaxScaler`.
3. **Chưa biến đổi Target:** `price` và `Price_INR_Lakhs` được giữ nguyên giá trị thực tế (chưa lấy `np.log1p`).
4. **Chưa lọc Outlier thống kê:** Chưa dùng IQR hay Z-score để xóa dữ liệu.

---

## 3. Cách thức phân chia Train / Test

* Cố định tham số ngẫu nhiên: `random_state = 42`, tỉ lệ phân chia **80% Train / 20% Test**.
* **Bộ 1:** Chia theo nhóm bằng `GroupShuffleSplit` dựa trên cột `id`. Điều này đảm bảo những ngôi nhà được bán lại nhiều lần trong giai đoạn 2014–2015 sẽ nằm trọn vẹn trong một tập, **$0$ mã `id` nào bị trùng lặp giữa Train và Test**.
* **Bộ 2:** Chia ngẫu nhiên bằng `train_test_split` từ tập 80k dòng của `train.csv` (vì file `test.csv` từ Kaggle không có cột giá).

---

## 4. Cách sử dụng & Chạy lại

### Cách chạy tái lập toàn bộ dữ liệu:
Chỉ cần chạy lệnh sau tại thư mục gốc, toàn bộ 4 file sẽ được tạo lại với kết quả hoàn toàn đồng nhất:
```bash
python prepare_data.py
```
Để kiểm tra lại tính toàn vẹn (missing, shape, min/max):
```bash
python check_data.py
```

### Cách nạp dữ liệu vào code mô hình (Ví dụ):
```python
import pandas as pd
from d1.config_d1 import CAT, NUM, ORD, TARGET

# Đọc dữ liệu Bộ 1
train_df = pd.read_csv("data/processed/d1_train.csv")
test_df = pd.read_csv("data/processed/d1_test.csv")

X_train, y_train = train_df.drop(columns=[TARGET]), train_df[TARGET]
X_test, y_test = test_df.drop(columns=[TARGET]), test_df[TARGET]
```

---

## 5. Lưu ý quan trọng cho người huấn luyện mô hình

1. **Vấn đề Đa cộng tuyến (Multicollinearity) cho mô hình tuyến tính:**
   - **Bộ 1:** Tồn tại đẳng thức đại số hoàn hảo:
     $$\text{sqft\_above} + \text{sqft\_basement} = \text{sqft\_living}$$
     Nếu dùng hồi quy OLS thuần túy mà không bỏ bớt 1 biến, ma trận sẽ bị suy biến. Khuyến nghị dùng **Ridge/Lasso** hoặc loại bớt một cột diện tích.
   - **Bộ 2:** Hai biến `Super_Area_SqFt` và `Carpet_Area_SqFt` có hệ số tương quan tuyến tính rất cao ($r > 0.95$). Người phụ trách Linear Regression cần lưu ý kiểm tra VIF.

2. **Quy tắc sử dụng tập Test:**
   - **Tập Test chỉ được nạp đúng 1 lần duy nhất** ở bước cuối cùng để ghi nhận điểm số trong bảng so sánh báo cáo.
   - Mọi quyết định chọn mô hình, tinh chỉnh siêu tham số (Hyperparameter Tuning) phải được thực hiện thông qua **5-Fold Cross-Validation trên tập Train**.

