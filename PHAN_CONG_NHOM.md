# Phân công nhiệm vụ nhóm: Dự đoán giá bất động sản (House Price Prediction)

Môn Học máy (Machine Learning) - PTIT. Nhóm 3 người, 2 bộ dữ liệu (King County & Indian Urban Housing), 6 thuật toán hồi quy.

---

## 1. Cơ chế phối hợp nhóm (Mô hình Dữ liệu Sạch Tập trung - Cách 1)

Nhóm thống nhất áp dụng **Cách 1: Người 1 cung cấp sẵn 2 file dữ liệu đã làm sạch (`data/processed/`)**:

* **Người 1 (A - Bạn):** Đã hoàn tất toàn bộ phân tích khám phá (EDA), xử lý ngoại lai và trích xuất đặc trưng (Feature Engineering) cho cả 2 bộ dữ liệu, xuất thành phẩm là 2 file CSV sạch chuẩn:
  * `data/processed/d1_clean.csv` (King County: 21,612 dòng, đã loại ngoại lai 33 phòng, trích xuất tuổi nhà, bỏ ID).
  * `data/processed/d2_clean.csv` (Indian Urban: 79,967 dòng, đã lọc bỏ 33 căn `< 100 sqft`, kiểm tra tính logic số tầng/diện tích).
* **Người 2 (B) & Người 3 (C):** **KHÔNG cần phải tải dữ liệu thô hay viết code tiền xử lý lại**. Chỉ cần đọc trực tiếp 2 file CSV sạch do Người 1 cung cấp để tập trung toàn lực vào huấn luyện mô hình, vẽ biểu đồ và tinh chỉnh siêu tham số.
* **Người 1** phụ trách thêm 2 mô hình hồi quy Baseline: **Linear Regression** và **Ridge/Lasso**.

---

## 2. Bảng phân công tổng quát

| Thành viên | Vai trò chính | Thuật toán phụ trách | Nguồn dữ liệu sử dụng | Mảng việc chung phụ trách | Phần viết báo cáo |
|---|---|---|---|---|---|
| **Người 1 (A - Bạn)** | **Data & Baseline Lead** | Linear Regression, Ridge / Lasso | Tự xử lý từ dữ liệu gốc ra 2 file CSV sạch | Cung cấp `d1_clean.csv`, `d2_clean.csv`, script clean | **Chương 1 & 2:** Tổng quan đề tài, EDA, toàn bộ quá trình làm sạch tạo 2 file CSV, Linear, Ridge/Lasso |
| **Người 2 (B)** | **Tree Models & Visualization** | Decision Tree, Random Forest | Đọc trực tiếp từ `data/processed/` | Viết mô-đun vẽ đồ thị (`common/plots.py`), Residual, Overfitting, Feature Importance | **Chương 3:** Cơ chế DT & RF, phân tích Overfitting của cây đơn, sức mạnh giảm phương sai của Random Forest |
| **Người 3 (C)** | **Advanced Models & Pipeline Lead** | XGBoost / LightGBM, MLP (Neural Net) | Đọc trực tiếp từ `data/processed/` | Dựng khung pipeline chung (`common/pipeline.py`), metric, `main.py` tổng hợp kết quả | **Chương 4, 5 & Slide:** Cơ chế XGBoost/MLP, phương pháp Tuning, bảng so sánh tổng hợp, kết luận, slide thuyết trình |

---

## 3. Chi tiết công việc từng thành viên

### Người 1 (A - Bạn): Data Lead & Baseline Models

#### Mảng Dữ liệu & Tiền xử lý (ĐÃ HOÀN THÀNH - Có 2 file CSV sạch)
- [x] **Phân tích EDA & Đánh giá chất lượng:** Hoàn thành báo cáo phân tích toàn diện 2 bộ dữ liệu (tương quan, phân phối, outlier).
- [ ] **Đóng gói dữ liệu sạch (`data/processed/`):**
  - **King County (`d1_clean.csv`):**
    + Loại bỏ cột `id`.
    + Loại bỏ bản ghi lỗi nhập liệu: `id = 15870` (33 phòng ngủ nhưng chỉ có 1,620 sqft).
    + Tách cột `date` thành `sale_year`, `sale_month`.
    + Tạo đặc trưng mới: `house_age = sale_year - yr_built`, cờ nhị phân `renovated` (1 nếu đã cải tạo, 0 nếu chưa).
  - **Indian Urban Housing (`d2_clean.csv`):**
    + Loại bỏ cột `Property_ID`.
    + Lọc bỏ 33 căn ngoại lai diện tích phi thực tế: `Super_Area_SqFt < 100` (nhỏ nhất là 16 sqft).
    + Xác thực logic: `Floor_Number <= Total_Floors`, `Carpet_Area_SqFt <= Super_Area_SqFt`.
    + Giữ lại các căn biệt thự giá cao (> 1,000 Lakhs) vì thuộc phân khúc hạng sang thực tế.
- [ ] **Cung cấp file cấu hình `config.py`:**
  - Định nghĩa sẵn cho Người 2 và Người 3 danh sách: cột số (`NUMERICAL_COLS`), cột phân loại One-Hot (`CATEGORICAL_COLS`), cột thứ tự Ordinal (`ORDINAL_COLS`) và cột mục tiêu `TARGET = 'price'` (hoặc `'Price_INR_Lakhs'`).

#### Thuật toán phụ trách
- [ ] **Thuật toán 1: Linear Regression (Baseline)**
  - Huấn luyện trên cả 2 file `d1_clean.csv` và `d2_clean.csv`.
  - Áp dụng biến đổi mục tiêu $\ln(1+y)$ và chuẩn hóa biến số.
  - Phân tích đa cộng tuyến (VIF giữa các biến diện tích).
  - Đo lường R², RMSLE, MAPE (CV và Test).
- [ ] **Thuật toán 2: Ridge / Lasso Regression**
  - Tinh chỉnh siêu tham số `alpha` bằng 5-fold CV.
  - Phân tích Lasso để phát hiện các biến bị triệt tiêu hệ số về 0.
  - Đánh giá vai trò của Regularization so với Linear Regression thông thường.

---

### Người 2 (B): Decision Tree, Random Forest & Trực quan hóa

#### Mảng việc chung
- [ ] Đọc trực tiếp 2 file dữ liệu sạch từ `data/processed/`.
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
  - Đánh giá hiện tượng Overfitting khi cây không bị giới hạn.
- [ ] **Thuật toán 2: Random Forest Regressor**
  - Viết trong `common/models/trees.py`.
  - Tuning: `n_estimators`, `max_depth`, `max_features`, `min_samples_leaf`.
  - So sánh đối chiếu với Decision Tree: Ensemble giúp giảm phương sai (variance reduction) vượt trội như thế nào.

---

### Người 3 (C): MLP, XGBoost/LightGBM & Khung Pipeline chung

#### Mảng việc chung
- [ ] Đọc trực tiếp 2 file dữ liệu sạch từ `data/processed/`.
- [ ] Xây dựng `common/pipeline.py`:
  - `ColumnTransformer`: One-Hot encoder cho biến phân loại, `StandardScaler` cho biến số.
  - Tích hợp `TransformedTargetRegressor` cho biến đổi $\ln(1+y)$.
  - Phân chia tập chuẩn 80% Train - 20% Test (`random_state = 42`), 5-Fold Cross Validation.
- [ ] Xây dựng `common/metrics.py`: R², MAE, RMSE, RMSLE, MAPE.
- [ ] Viết `main.py`: Tự động gọi huấn luyện 6 mô hình trên 2 dataset và xuất bảng so sánh tổng hợp vào `results/summary_comparison.csv`.

#### Thuật toán phụ trách
- [ ] **Thuật toán 1: Neural Network (MLPRegressor)**
  - Viết trong `common/models/neural_boost.py`.
  - Bắt buộc kiểm tra dữ liệu đầu vào đã scale chuẩn.
  - Tuning: `hidden_layer_sizes`, `activation`, `learning_rate_init`, `alpha`, bật `early_stopping=True`.
  - Vẽ loss curve theo từng epoch.
- [ ] **Thuật toán 2: XGBoost hoặc LightGBM (Mô hình tốt nhất)**
  - Viết trong `common/models/neural_boost.py`.
  - Tuning: `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`.
  - Early stopping trên tập validation.
  - Phân tích độ quan trọng và tương tác đặc trưng bằng SHAP values.

---

## 4. 

## 5. Cấu trúc thư mục dự án chuẩn

```text
BTL/
├── data/
│   ├── raw/                         # Dữ liệu gốc tải từ Kaggle (nếu cần lưu trữ)
│   │   ├── kc_house_data.csv
│   │   └── indian_urban_train.csv
│   └── processed/                   # [QUAN TRỌNG NHẤT: CẢ NHÓM DÙNG CHUNG]
│       ├── d1_clean.csv             # King County đã làm sạch (21,612 dòng)
│       └── d2_clean.csv             # Indian Urban đã làm sạch (79,967 dòng)
├── d1/                              # [Người 1 phụ trách]
│   ├── eda_d1.ipynb                 # Notebook EDA King County
│   └── make_clean_d1.py             # Script sinh ra d1_clean.csv
├── d2/                              # [Người 1 phụ trách]
│   ├── eda_d2.ipynb                 # Notebook EDA Indian Urban
│   └── make_clean_d2.py             # Script sinh ra d2_clean.csv
├── common/                          # [Cả nhóm phối hợp]
│   ├── pipeline.py                  # Người 3
│   ├── metrics.py                   # Người 3
│   ├── plots.py                     # Người 2
│   └── models/
│       ├── linear.py                # Người 1: Linear, Ridge, Lasso
│       ├── trees.py                 # Người 2: Decision Tree, Random Forest
│       └── neural_boost.py          # Người 3: MLP, XGBoost/LightGBM
├── main.py                          # Chạy toàn bộ thí nghiệm & xuất kết quả
├── models/                          # Lưu trữ file .pkl
├── results/                         # Bảng CSV so sánh kết quả và đồ thị
├── BAO_CAO_TIEN_XU_LY_NGUOI_1.md    # Báo cáo tiền xử lý mẫu của Người 1
└── requirements.txt
```

---

## 6. Kế hoạch hoàn thành & Phân chia viết báo cáo

| Chương trong Báo cáo | Nội dung chi tiết | Người viết |
|---|---|---|
| **Chương 1: Giới thiệu & Khám phá dữ liệu (EDA)** | Tổng quan 2 bài toán dự đoán giá nhà (King County & Indian Urban), thống kê mô tả, phân tích biểu đồ tương quan, phân phối giá | **Người 1** *(Chuyển từ PDF)* |
| **Chương 2: Đánh giá chất lượng & Tiền xử lý** | Phân tích thiếu/trùng lặp (0%), xử lý ngoại lai 33 phòng và 33 căn < 100 sqft, biến đổi log1p target, quy trình tạo ra 2 file CSV sạch | **Người 1** *(Chuyển từ PDF)* |
| **Chương 3: Hồi quy tuyến tính & Baseline** | Cơ chế Linear Regression, Ridge/Lasso, phân tích đa cộng tuyến (VIF), siêu tham số alpha, đánh giá R² trên 2 bộ | **Người 1** |
| **Chương 4: Nhóm mô hình Cây (Trees)** | Cơ chế Decision Tree, Random Forest; phân tích hiện tượng Overfitting khi tăng độ sâu; vai trò của Ensemble (giảm phương sai) | **Người 2** |
| **Chương 5: Nhóm mô hình Nâng cao & Boosting** | Cơ chế MLP (Mạng nơ-ron đa tầng), XGBoost/LightGBM; cơ chế Early Stopping; giải thích mô hình bằng SHAP values | **Người 3** |
| **Chương 6: Đánh giá so sánh & Kết luận** | Bảng tổng hợp so sánh 6 mô hình trên 2 dataset (R², RMSLE, MAPE, MAE); thảo luận kết quả; làm Slide thuyết trình | **Người 3** *(Tổng hợp cùng cả nhóm)* |
