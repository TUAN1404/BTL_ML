"""Cấu hình đặc trưng cho Bộ 2: Indian Urban Housing
File: d2/config_d2.py

Định nghĩa phân loại biến:
- TARGET : Biến mục tiêu cần dự đoán (Price_INR_Lakhs, đơn vị Lakhs - Rúp Ấn Độ)
- CAT    : Biến phân loại danh mục (Categorical Nominal - cần One-Hot Encoding)
- ORD    : Biến thứ bậc (Categorical Ordinal - mã hóa theo thứ tự tăng dần tiện nghi)
- NUM    : Biến định lượng số (Numerical - cần StandardScaler cho Linear/MLP)

LƯU Ý QUAN TRỌNG CHO MÔ HÌNH TUYẾN TÍNH (Linear / Ridge / Lasso):
- Đa cộng tuyến cao (High Multicollinearity): Super_Area_SqFt và Carpet_Area_SqFt
  có hệ số tương quan rất lớn (r > 0.95) vì cùng phản ánh diện tích căn hộ.
  Người huấn luyện mô hình tuyến tính cần lưu ý kiểm tra hệ số phóng đại phương sai (VIF)
  hoặc áp dụng điều quy L1/L2 (Ridge/Lasso) để tránh bất ổn định tham số trọng số.
"""

import os
import pandas as pd

# Biến mục tiêu
TARGET = "Price_INR_Lakhs"

# Biến phân loại danh mục (Nominal Categorical)
CAT = ["City", "Property_Type"]

# Biến thứ bậc (Ordinal Categorical) - Thứ tự từ thấp đến cao dựa trên giá trị unique thực tế
ORD = {
    "Locality_Type": ["Affordable", "Mid-Range", "Premium"],
    "Furnishing_Status": ["Unfurnished", "Semi-Furnished", "Furnished"],
}

# Biến số (Numerical)
NUM = [
    "BHK",
    "Bathrooms",
    "Super_Area_SqFt",
    "Carpet_Area_SqFt",
    "Floor_Number",
    "Total_Floors",
    "Age_of_Property",
    "Parking",
    "Lift_Available",
    "Gated_Community",
    "Distance_to_Metro_km",
    "Distance_to_City_Center_km",
]


def validate_config():
    """Kiểm tra tính nhất quán giữa file cấu hình và dữ liệu processed thực tế."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, "data", "processed", "d2_train.csv")

    if not os.path.exists(train_path):
        print(f"[!] Chưa tìm thấy file dữ liệu processed: {train_path}")
        return False

    df = pd.read_csv(train_path, nrows=5)
    actual_cols = set(df.columns)
    config_cols = set([TARGET] + CAT + NUM + list(ORD.keys()))

    missing_in_config = actual_cols - config_cols
    extra_in_config = config_cols - actual_cols

    print("=" * 60)
    print("KIỂM TRA CẤU HÌNH BỘ 2 (Indian Urban Housing):")
    print(f"- Tổng số cột trong data/processed/d2_train.csv: {len(actual_cols)}")
    print(f"- Tổng số cột trong config_d2.py               : {len(config_cols)}")
    print(f"  + TARGET : 1 cột ('{TARGET}')")
    print(f"  + CAT    : {len(CAT)} cột {CAT}")
    print(f"  + ORD    : {len(ORD)} cột {list(ORD.keys())}")
    print(f"  + NUM    : {len(NUM)} cột")

    assert not missing_in_config, f"Có cột trong dữ liệu nhưng thiếu trong config: {missing_in_config}"
    assert not extra_in_config, f"Có cột trong config nhưng không có trong dữ liệu: {extra_in_config}"

    print("[V] XÁC NHẬN: Mọi cột đều khớp 100%, không bị bỏ sót hoặc thừa cột nào!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    validate_config()

