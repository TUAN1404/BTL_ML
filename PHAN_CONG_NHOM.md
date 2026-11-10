# Phân công nhiệm vụ nhóm: Dự đoán giá bất động sản (House Price Prediction)

Môn Học máy (Machine Learning) - PTIT. Nhóm 3 người, 2 bộ dữ liệu (King County & Indian Urban Housing), 6 thuật toán hồi quy.

---

## 1. Cơ chế phối hợp nhóm (Mô hình Dữ liệu Sạch & Chia Tập Chuẩn - ĐÃ HOÀN TẤT)

Nhóm thống nhất áp dụng **Mô hình Dữ liệu Sạch Tập trung**:

* **Người 1 (A - Data & Baseline Lead):** Đã hoàn tất toàn bộ phân tích khám phá (EDA), tiền xử lý, làm sạch và chia tập Train/Test chuẩn hóa, bàn giao thành phẩm gồm:
  * **4 file dữ liệu sẵn sàng cho huấn luyện và kiểm thử (`data/processed/`):**
    * `d1_train.csv` (17,291 dòng × 22 cột) & `d1_test.csv` (4,321 dòng × 22 cột): Chia theo nhóm `id` bằng `GroupShuffleSplit` (tỉ lệ 80/20, `random_state=42`), triệt tiêu 100% rò rỉ ID nhà.
    * `d2_train.csv` (63,973 dòng × 17 cột) & `d2_test.csv` (15,994 dòng × 17 cột): Chia ngẫu nhiên 80/20 từ dữ liệu đã lọc bỏ căn hộ ngoại lai `< 100 sqft`.
    * Cùng 2 file toàn bộ đã làm sạch: `d1_clean.csv` (21,612 dòng) và `d2_clean.csv` (79,967 dòng).
  * **2 file cấu hình đặc trưng chuẩn:** `d1/config_d1.py` và `d2/config_d2.py` phân loại sẵn các nhóm biến `TARGET`, `NUM`, `CAT`, `ORD` để đưa trực tiếp vào `ColumnTransformer`.
  * **Hệ thống kiểm thử tự động toàn diện:** 95 ca test tự động trong thư mục `tests/` cùng báo cáo kiểm định chất lượng `reports/TEST_REPORT.md` chứng minh dữ liệu đạt 100% độ toàn vẹn, 0% rò rỉ (no data leakage) và tái lập tuyệt đối (reproducible).
* **Người 2 (B) & Người 3 (C):** **KHÔNG cần phải tải dữ liệu thô hay viết lại code tiền xử lý**. Chỉ cần đọc trực tiếp 4 file `d1_train.csv`, `d1_test.csv`, `d2_train.csv`, `d2_test.csv` và import các danh mục đặc trưng từ `config_d1.py`, `config_d2.py` để tập trung toàn lực vào huấn luyện mô hình, trực quan hóa và tinh chỉnh siêu tham số.
* **Người 1** phụ trách thêm 2 mô hình hồi quy Baseline: **Linear Regression** và **Ridge / Lasso Regression**.

---

## 2. Bảng phân công tổng quát

| Thành viên | Vai trò chính | Thuật toán phụ trách | Nguồn dữ liệu sử dụng | Mảng việc chung phụ trách | Phần viết báo cáo |
|---|---|---|---|---|---|
| **Người 1 (A)** | **Data & Baseline Lead** | Linear Regression, Ridge / Lasso | Tự xử lý từ raw ra 4 file Train/Test và 2 file cấu hình | Cung cấp dữ liệu processed, `config_d1.py`, `config_d2.py`, script `prepare_data.py`, `check_data.py`, bộ test `tests/` | **Chương 1, 2 & 3:** Tổng quan đề tài, EDA, toàn bộ quá trình tiền xử lý, kiểm định chất lượng dữ liệu, mô hình Linear, Ridge/Lasso |
| **Người 2 (B)** | **Tree Models & Visualization** | Decision Tree, Random Forest | Đọc trực tiếp từ `data/processed/` (`d1_train/test`, `d2_train/test`) | Viết mô-đun vẽ đồ thị (`common/plots.py`): Predicted vs Actual, Residual Plots, Overfitting Curves, Feature Importance | **Chương 4:** Cơ chế DT & RF, phân tích Overfitting của cây đơn, sức mạnh giảm phương sai của Random Forest |
| **Người 3 (C)** | **Advanced Models & Pipeline Lead** | XGBoost / LightGBM, MLP (Neural Net) | Đọc trực tiếp từ `data/processed/` (`d1_train/test`, `d2_train/test`) | Dựng khung pipeline chung (`common/pipeline.py`), metrics (`common/metrics.py`), `main.py` tổng hợp kết quả thí nghiệm | **Chương 5, 6 & Slide:** Cơ chế XGBoost/MLP, phương pháp Tuning, bảng so sánh tổng hợp 6 mô hình, kết luận, slide thuyết trình |

---

## 3. Chi tiết công việc từng thành viên

### Người 1 (A): Data Lead & Baseline Models

#### Mảng Dữ liệu & Tiền xử lý (ĐÃ HOÀN TẤT 100%)
- [x] **Phân tích EDA & Đánh giá chất lượng:** Hoàn thành báo cáo phân tích toàn diện 2 bộ dữ liệu (tương quan, phân phối, outlier, missing).
- [x] **Đóng gói dữ liệu sạch & chia tập chuẩn (`data/processed/`):**
  - **King County (`d1_train.csv` & `d1_test.csv`):**
    + Loại bỏ dòng lỗi gõ phím `id = 15870` (33 phòng ngủ nhưng chỉ 1,620 sqft).
    + Parse `date` thành `sale_year`, `sale_month`.
    + Tạo đặc trưng mới: `house_age = sale_year - yr_built`, cờ nhị phân `renovated` (1 nếu đã cải tạo, 0 nếu chưa).
    + Chia tập theo nhóm `id` bằng `GroupShuffleSplit(test_size=0.2, random_state=42)` triệt tiêu 100% ID trùng lặp, sau đó loại bỏ cột `id`.
  - **Indian Urban Housing (`d2_train.csv` & `d2_test.csv`):**
    + Loại bỏ cột `Property_ID`.
    + Lọc bỏ 33 căn ngoại lai diện tích phi thực tế: `Super_Area_SqFt < 100`.
    + Xác thực logic: `Floor_Number <= Total_Floors`, `Carpet_Area_SqFt <= Super_Area_SqFt`.
    + Chia tập ngẫu nhiên 80/20 (`random_state=42`) từ `train.csv` (cô lập file `test.csv` của Kaggle do không có nhãn).
- [x] **Cung cấp file cấu hình `d1/config_d1.py` và `d2/config_d2.py`:**
  - Định nghĩa sẵn: `NUM`, `CAT`, `ORD`, `TARGET` khớp 100% với dữ liệu processed.
- [x] **Xây dựng hệ thống kiểm thử tự động & Báo cáo độc lập:**
  - Viết `prepare_data.py`, `check_data.py`.
  - Bộ kiểm thử tự động 95 tests trong `tests/` (độ tái lập hash SHA-256, kiểm thử đơn vị làm sạch, tính toàn vẹn, tính nhất quán chia tập, kiểm thử khói smoke pipeline, kiểm thử tiêm lỗi đột biến fault injection).
  - Xuất báo cáo kiểm thử hoàn chỉnh tại `reports/TEST_REPORT.md` và `BAO_CAO_TIEN_XU_LY_NGUOI_1.md`.

#### Thuật toán phụ trách
- [ ] **Thuật toán 1: Linear Regression (Baseline)**
  - Huấn luyện trên cả 2 bộ (`d1_train` và `d2_train`).
  - Áp dụng biến đổi mục tiêu $\ln(1+y)$ và chuẩn hóa biến số qua `StandardScaler`.
  - Phân tích đa cộng tuyến (VIF giữa các biến diện tích).
  - Đo lường R², RMSLE, MAPE (CV 5-fold và Test).
- [ ] **Thuật toán 2: Ridge / Lasso Regression**
  - Tinh chỉnh siêu tham số `alpha` bằng 5-fold CV.
  - Phân tích Lasso để phát hiện các biến bị triệt tiêu hệ số về 0.
  - Đánh giá vai trò của Regularization L1/L2 so với OLS thông thường.

---

### Người 2 (B): Decision Tree, Random Forest & Trực quan hóa

#### Mảng việc chung
- [ ] Đọc trực tiếp 4 file dữ liệu sạch từ `data/processed/`.
- [ ] Xây dựng mô-đun đồ thị `common/plots.py`:
  - Biểu đồ Predicted vs Actual trên tập Test.
  - Biểu đồ phần dư (Residual Plots).
  - Đồ thị đường cong Overfitting: R² của Train vs Test theo độ sâu `max_depth`.
  - Trực quan hóa một cây nông (`max_depth = 3`) giải thích cơ chế phân nhánh.
  - Biểu đồ xếp hạng mức độ quan trọng đặc trưng (Feature Importance) từ Random Forest trên cả 2 dataset.

#### Thuật toán phụ trách
- [ ] **Thuật toán 1: Decision Tree Regressor**
  - Viết trong `common/models/trees.py`.
  - Tuning: `max_depth`, `min_samples_split`, `min_samples_leaf`.
  - Đánh giá hiện tượng Overfitting khi cây không bị giới hạn độ sâu.
- [ ] **Thuật toán 2: Random Forest Regressor**
  - Viết trong `common/models/trees.py`.
  - Tuning: `n_estimators`, `max_depth`, `max_features`, `min_samples_leaf`.
  - So sánh đối chiếu với Decision Tree: Ensemble giúp giảm phương sai (variance reduction) vượt trội như thế nào.

---

### Người 3 (C): MLP, XGBoost/LightGBM & Khung Pipeline chung

#### Mảng việc chung
- [ ] Đọc trực tiếp 4 file dữ liệu sạch từ `data/processed/` và cấu hình từ `config_d1.py`, `config_d2.py`.
- [ ] Xây dựng `common/pipeline.py`:
  - `ColumnTransformer`: `OneHotEncoder(handle_unknown='ignore')` cho `CAT`, `OrdinalEncoder` cho `ORD`, `StandardScaler` cho `NUM`.
  - Tích hợp `TransformedTargetRegressor` cho biến đổi $\ln(1+y)$.
  - 5-Fold Cross Validation.
- [ ] Xây dựng `common/metrics.py`: R², MAE, RMSE, RMSLE, MAPE.
- [ ] Viết `main.py`: Tự động gọi huấn luyện 6 mô hình trên 2 dataset và xuất bảng so sánh tổng hợp vào `results/summary_comparison.csv`.

#### Thuật toán phụ trách
- [ ] **Thuật toán 1: Neural Network (MLPRegressor)**
  - Viết trong `common/models/neural_boost.py`.
  - Đảm bảo dữ liệu đầu vào đã scale chuẩn.
  - Tuning: `hidden_layer_sizes`, `activation`, `learning_rate_init`, `alpha`, bật `early_stopping=True`.
  - Vẽ loss curve theo từng epoch.
- [ ] **Thuật toán 2: XGBoost hoặc LightGBM (Mô hình mạnh nhất)**
  - Viết trong `common/models/neural_boost.py`.
  - Tuning: `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`.
  - Early stopping trên tập validation.
  - Phân tích độ quan trọng và tương tác đặc trưng bằng SHAP values.

---

## 4. Cấu trúc thư mục dự án chuẩn

```text
BTL/
├── data/
│   ├── raw/                         # Dữ liệu gốc (kc_house_data.csv, train.csv, test.csv)
│   └── processed/                   # [CẢ NHÓM DÙNG CHUNG - ĐÃ CÓ SẴN]
│       ├── d1_train.csv             # King County Train (17,291 dòng × 22 cột)
│       ├── d1_test.csv              # King County Test  (4,321 dòng × 22 cột)
│       ├── d2_train.csv             # Indian Urban Train (63,973 dòng × 17 cột)
│       ├── d2_test.csv              # Indian Urban Test  (15,994 dòng × 17 cột)
│       ├── d1_clean.csv             # Toàn bộ Bộ 1 đã làm sạch (21,612 dòng)
│       └── d2_clean.csv             # Toàn bộ Bộ 2 đã làm sạch (79,967 dòng)
├── d1/                              # [Người 1 phụ trách]
│   ├── clean_d1.py                  # Module làm sạch Bộ 1
│   └── config_d1.py                 # Cấu hình đặc trưng Bộ 1 (TARGET, NUM, CAT, ORD)
├── d2/                              # [Người 1 phụ trách]
│   ├── clean_d2.py                  # Module làm sạch Bộ 2
│   └── config_d2.py                 # Cấu hình đặc trưng Bộ 2 (TARGET, NUM, CAT, ORD)
├── common/                          # [Cả nhóm phối hợp triển khai mô hình]
│   ├── pipeline.py                  # Người 3
│   ├── metrics.py                   # Người 3
│   ├── plots.py                     # Người 2
│   └── models/
│       ├── linear.py                # Người 1: Linear, Ridge, Lasso
│       ├── trees.py                 # Người 2: Decision Tree, Random Forest
│       └── neural_boost.py          # Người 3: MLP, XGBoost/LightGBM
├── tests/                           # [Hệ thống kiểm thử tự động - 95 tests]
│   ├── test_clean.py                # Unit test hàm clean_d1, clean_d2
│   ├── test_split.py                # Kiểm tra chia dữ liệu, không rò rỉ ID
│   ├── test_config.py               # Kiểm tra tính khớp nối của config
│   ├── test_data_integrity.py       # Kiểm tra missing, dtypes, range
│   ├── test_reproducibility.py      # Kiểm tra tái lập bằng hash SHA-256
│   ├── test_smoke.py                # Pipeline tối thiểu (Ridge vs Dummy)
│   ├── test_fault_injection.py      # Kiểm thử tiêm lỗi đột biến
│   └── test_audit_independent.py    # Kiểm thử độc lập toàn diện
├── reports/                         # Báo cáo kiểm định chất lượng độc lập
│   ├── TEST_REPORT.md               # Báo cáo nghiệm thu kỹ thuật chính thức
│   └── INDEPENDENT_TEST_REPORT.md   # Báo cáo kiểm định độc lập ban đầu
├── prepare_data.py                  # Script chia tập dữ liệu chuẩn
├── check_data.py                    # Script kiểm tra nhanh tính toàn vẹn dữ liệu
├── main.py                          # Chạy toàn bộ thí nghiệm & xuất kết quả
├── models/                          # Lưu trữ file .pkl
├── results/                         # Bảng CSV so sánh kết quả và đồ thị
├── BAO_CAO_TIEN_XU_LY_NGUOI_1.md    # Báo cáo tiền xử lý của Người 1
└── requirements.txt
```

---

## 5. Kế hoạch hoàn thành & Phân chia viết báo cáo

| Chương trong Báo cáo | Nội dung chi tiết | Người viết | Trạng thái |
|---|---|---|:---:|
| **Chương 1: Giới thiệu & Khám phá dữ liệu (EDA)** | Tổng quan 2 bài toán dự đoán giá nhà (King County & Indian Urban), thống kê mô tả, biểu đồ phân phối giá, tương quan đặc trưng | **Người 1** | Đã sẵn sàng |
| **Chương 2: Đánh giá chất lượng & Tiền xử lý** | Phân tích thiếu/trùng lặp (0%), xử lý ngoại lai 33 phòng và 33 căn < 100 sqft, kỹ thuật GroupShuffleSplit chống rò rỉ, nghiệm thu 95 tests | **Người 1** | Đã hoàn tất |
| **Chương 3: Hồi quy tuyến tính & Baseline** | Cơ chế Linear Regression, Ridge/Lasso, phân tích đa cộng tuyến (VIF), siêu tham số alpha, đánh giá R² trên 2 bộ | **Người 1** | Đang triển khai |
| **Chương 4: Nhóm mô hình Cây (Trees)** | Cơ chế Decision Tree, Random Forest; phân tích hiện tượng Overfitting khi tăng độ sâu; vai trò của Ensemble (giảm phương sai) | **Người 2** | Chờ thực hiện |
| **Chương 5: Nhóm mô hình Nâng cao & Boosting** | Cơ chế MLP (Mạng nơ-ron đa tầng), XGBoost/LightGBM; cơ chế Early Stopping; giải thích mô hình bằng SHAP values | **Người 3** | Chờ thực hiện |
| **Chương 6: Đánh giá so sánh & Kết luận** | Bảng tổng hợp so sánh 6 mô hình trên 2 dataset (R², RMSLE, MAPE, MAE); thảo luận kết quả; làm Slide thuyết trình | **Người 3** *(Tổng hợp cùng cả nhóm)* | Chờ thực hiện |
