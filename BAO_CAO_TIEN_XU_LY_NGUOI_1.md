# BÁO CÁO KỸ THUẬT: PHÂN TÍCH, ĐÁNH GIÁ VÀ TIỀN XỬ LÝ DỮ LIỆU
**Đề tài:** Dự đoán giá bất động sản (House Price Prediction)  
**Học phần:** Học máy (Machine Learning) - Học viện Công nghệ Bưu chính Viễn thông (PTIT)  
**Người thực hiện:** Người 1 (Vai trò: Data & Baseline Lead)  
**Sản phẩm hoàn thành bàn giao:** 2 tập dữ liệu sạch tại `data/processed/d1_clean.csv` và `data/processed/d2_clean.csv`

---

## 1. TỔNG QUAN HAI TẬP DỮ LIỆU

Nhằm đánh giá năng lực dự báo của các thuật toán học máy trong nhiều bối cảnh thị trường thực tế khác nhau, đề tài tiến hành thực nghiệm song song trên 2 tập dữ liệu bất động sản đại diện:

| Tiêu chí | Dataset 1: King County House Sales | Dataset 2: Indian Urban Housing |
|---|---|---|
| **Khu vực nghiên cứu** | Quận King (gồm TP. Seattle), bang Washington, Mỹ | 10 đại đô thị lớn tại Ấn Độ (Mumbai, Delhi, Bangalore, Pune...) |
| **Loại hình nhà ở** | Nhà ở thấp tầng riêng lẻ vùng ngoại ô (Single-family homes) | Căn hộ chung cư, nhà phố, biệt thự đô thị nén |
| **Kích thước ban đầu** | 21,613 dòng × 21 cột | 80,000 dòng × 18 cột |
| **Biến mục tiêu (Target)** | `price` (Đô la Mỹ - USD) | `Price_INR_Lakhs` (Lakhs - Rúp Ấn Độ; 1 Lakh ≈ 30 triệu VNĐ) |
| **Đặc trưng tương quan cao nhất** | `sqft_living` ($r = 0.702$), `grade` ($r = 0.667$) | `Super_Area_SqFt` ($r \approx 0.635$), `City` (Mumbai, Delhi cao gấp 3–5 lần) |

---

## 2. KIỂM TRA CHẤT LƯỢNG DỮ LIỆU (DATA QUALITY ASSESSMENT)

### 2.1. Giá trị khuyết thiếu (Missing Values)
- **Dataset 1 (King County):** $0$ giá trị rỗng ($0.00\%$) trên toàn bộ 21,613 dòng và 21 cột.
- **Dataset 2 (Indian Urban):** $0$ giá trị rỗng ($0.00\%$) trên 80,000 dòng và 18 cột.
- **Kết luận:** Dữ liệu hoàn toàn sạch về cấu trúc, không cần áp dụng thuật toán điền khuyết thiếu phức tạp (Imputation), triệt tiêu hoàn toàn nguy cơ rò rỉ dữ liệu (Data Leakage).

### 2.2. Dữ liệu trùng lặp (Duplicate Records)
- **Dataset 1:** $0$ bản ghi trùng lặp ($0.00\%$). Toàn bộ `id` đại diện cho các giao dịch độc lập.
- **Dataset 2:** $0$ bản ghi trùng lặp ($0.00\%$). Mã `Property_ID` là duy nhất trên toàn tập.

### 2.3. Dữ liệu ngoại lai (Outliers) và Chiến lược xử lý
Sau khi phân tích trực quan qua biểu đồ phân tán (Scatter Plot) và Boxplot, Người 1 phát hiện và đã xử lý các điểm dị biệt sau:

1. **King County - Lỗi gõ phím về số phòng ngủ:**
   - *Hiện tượng:* Bản ghi có `id = 15870` khai báo có tới **33 phòng ngủ**, nhưng tổng diện tích sinh hoạt chỉ có $1,620\text{ sqft}$ (khoảng $150\text{ m}^2$) và 1.75 phòng tắm.
   - *Xử lý:* Đây là lỗi đánh máy rõ ràng khi nhập liệu (thực tế có thể là 3 phòng ngủ). Để tránh làm sai lệch độ dốc của mô hình hồi quy tuyến tính, bản ghi này **đã bị loại bỏ**.

2. **Indian Urban - Ngoại lai diện tích cực nhỏ:**
   - *Hiện tượng:* Phát hiện **33 căn hộ** có tổng diện tích `Super_Area_SqFt < 100` (căn nhỏ nhất chỉ $16\text{ sqft} \approx 1.5\text{ m}^2$).
   - *Xử lý:* Trong quy hoạch đô thị thực tế, diện tích nhà ở không thể nhỏ hơn $100\text{ sqft}$ (đây là các gian hàng ki-ốt hoặc lỗi nhập dữ liệu). **33 bản ghi này đã được lọc bỏ**.

3. **Hiện tượng giá nhà siêu đắt (Phân khúc biệt thự cao cấp):**
   - *King County:* Có 1,146 căn nhà ($5.3\%$) có giá trên $\$1.1\text{M USD}$ (cao nhất $\$7.7\text{M}$).
   - *Indian Urban:* Một số bất động sản hạng sang tại trung tâm Mumbai có giá trên $1,000\text{ Lakhs}$ (cao nhất $3,334\text{ Lakhs}$).
   - *Quyết định:* **Giữ lại toàn bộ** các căn này vì đây là bất động sản thực tế, không phải nhiễu. Để khắc phục hiện tượng phân phối bị lệch phải nghiêm trọng (right-skewed), nhóm áp dụng phép biến đổi logarit tự nhiên:
     $$y_{\text{trans}} = \ln(1 + \text{Target})$$
     Biến đổi này giúp kéo các giá trị cực lớn về phân phối chuẩn, giúp thuật toán hồi quy hội tụ ổn định.

---

## 3. QUY TRÌNH TIỀN XỬ LÝ & KỸ THUẬT ĐẶC TRƯNG (FEATURE ENGINEERING)

Người 1 đã thực thi quy trình chuẩn hóa và trích xuất đặc trưng để tạo ra 2 file CSV sạch:

### 3.1. Đối với King County (`data/processed/d1_clean.csv`)
- **Loại bỏ đặc trưng không có giá trị học máy:** Cột `id`.
- **Trích xuất thời gian:** Tách chuỗi `date` (định dạng `YYYYMMDDT...`) thành `sale_year` (năm giao dịch) và `sale_month` (tháng giao dịch).
- **Tạo đặc trưng tuổi nhà:** Tính toán `house_age = sale_year - yr_built`.
- **Tạo cờ cải tạo:** `renovated = 1` nếu `yr_renovated > 0`, ngược lại bằng $0$.
- **Kết quả:** Tập dữ liệu sạch gồm **21,612 dòng** và **22 cột**.

### 3.2. Đối với Indian Urban Housing (`data/processed/d2_clean.csv`)
- **Loại bỏ đặc trưng định danh:** Cột `Property_ID`.
- **Lọc ngoại lai:** Lọc bỏ 33 căn hộ có diện tích `Super_Area_SqFt < 100`.
- **Kiểm định tính hợp lệ logic:**
  + Xác nhận $100\%$ các dòng thỏa mãn: `Floor_Number <= Total_Floors`.
  + Xác nhận $100\%$ các dòng thỏa mãn: `Carpet_Area_SqFt <= Super_Area_SqFt`.
- **Kết quả:** Tập dữ liệu sạch gồm **79,967 dòng** và **17 cột**.

---

## 4. METADATA VÀ CẤU HÌNH DÀNH CHO CÁC THÀNH VIÊN TRONG NHÓM

Các thành viên Người 2 (Decision Tree / Random Forest) và Người 3 (MLP / XGBoost) chỉ cần đọc trực tiếp 2 file sạch từ `data/processed/` với các thông số sau:

### Bộ 1: `d1_clean.csv`
* **File path:** `data/processed/d1_clean.csv`
* **Target:** `'price'`
* **Biến số liên tục (`NUMERICAL`):** `bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`, `waterfront`, `view`, `condition`, `grade`, `sqft_above`, `sqft_basement`, `yr_built`, `yr_renovated`, `zipcode`, `lat`, `long`, `sqft_living15`, `sqft_lot15`, `sale_year`, `sale_month`, `house_age`, `renovated`.

### Bộ 2: `d2_clean.csv`
* **File path:** `data/processed/d2_clean.csv`
* **Target:** `'Price_INR_Lakhs'`
* **Biến danh mục One-Hot:** `City`, `Property_Type`
* **Biến thứ tự Ordinal:** `Locality_Type`, `Furnishing_Status`
* **Biến số (`NUMERICAL`):** `BHK`, `Bathrooms`, `Super_Area_SqFt`, `Carpet_Area_SqFt`, `Floor_Number`, `Total_Floors`, `Age_of_Property`, `Parking`, `Lift_Available`, `Gated_Community`, `Distance_to_Metro_km`, `Distance_to_City_Center_km`.

---

## 5. KẾT QUẢ VÀ KẾ HOẠCH BƯỚC TIẾP THEO

1. **Dữ liệu:** Toàn bộ công đoạn khảo sát, đánh giá và tiền xử lý dữ liệu của đề tài đã hoàn tất $100\%$, sẵn sàng để toàn đội đưa vào mô hình học máy.
2. **Nhiệm vụ tiếp theo của Người 1:**
   - Cài đặt mô hình chuẩn **Linear Regression** trên cả 2 dataset sạch để đo điểm chuẩn Baseline (kỳ vọng $R^2 \approx 0.70 - 0.72$).
   - Cài đặt mô hình điều quy **Ridge và Lasso Regression**, tinh chỉnh tham số $\alpha$, phân tích các biến bị Lasso đưa về $0$.
   - Kiểm tra đa cộng tuyến giữa các biến diện tích (`sqft_above` vs `sqft_living`, `Super_Area` vs `Carpet_Area`).

