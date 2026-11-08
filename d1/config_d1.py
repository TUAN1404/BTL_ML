"""Cấu hình đặc trưng cho Bộ 1: King County House Sales
File: d1/config_d1.py

Định nghĩa phân loại biến:
- TARGET : Biến mục tiêu cần dự đoán (price, đơn vị USD)
- CAT    : Biến phân loại danh mục (Categorical - cần One-Hot Encoding)
- ORD    : Biến thứ bậc (Ordinal - cần mã hóa thứ tự, bộ 1 không có)
- NUM    : Biến định lượng số (Numerical - cần StandardScaler cho Linear/MLP)

LƯU Ý QUAN TRỌNG CHO MÔ HÌNH TUYẾN TÍNH (Linear / Ridge / Lasso):
- Đa cộng tuyến hoàn hảo (Collinearity): sqft_above + sqft_basement = sqft_living.
  Nếu đưa cả 3 biến vào mô hình hồi quy tuyến tính không có điều quy (OLS), ma trận X^T X
  sẽ bị suy biến hoặc phương sai ước lượng hệ số tăng vọt. Cần cân nhắc loại bỏ 1 trong 3 biến
  hoặc sử dụng Ridge/Lasso để co hệ số.
"""

import os
import pandas as pd

# Biến mục tiêu
TARGET = "price"

# Biến phân loại danh mục (Categorical)
CAT = ["zipcode"]

# Biến thứ bậc (Ordinal)
ORD = {}

# Biến số (Numerical)
NUM = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "lat",
    "long",
    "sqft_living15",
    "sqft_lot15",
    "sale_year",
    "sale_month",
    "house_age",
    "renovated",
    "waterfront",
    "view",
    "condition",
    "grade",
]


def validate_config():
    """Kiểm tra tính nhất quán giữa file cấu hình và dữ liệu processed thực tế."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, "data", "processed", "d1_train.csv")

    if not os.path.exists(train_path):
        print(f"[!] Chưa tìm thấy file dữ liệu processed: {train_path}")
        return False

    df = pd.read_csv(train_path, nrows=5)
    actual_cols = set(df.columns)
    config_cols = set([TARGET] + CAT + NUM + list(ORD.keys()))

    missing_in_config = actual_cols - config_cols
    extra_in_config = config_cols - actual_cols

    print("=" * 60)
    print("KIỂM TRA CẤU HÌNH BỘ 1 (King County):")
    print(f"- Tổng số cột trong data/processed/d1_train.csv: {len(actual_cols)}")
    print(f"- Tổng số cột trong config_d1.py               : {len(config_cols)}")
    print(f"  + TARGET : 1 cột ('{TARGET}')")
    print(f"  + CAT    : {len(CAT)} cột {CAT}")
    print(f"  + ORD    : {len(ORD)} cột")
    print(f"  + NUM    : {len(NUM)} cột")

    assert not missing_in_config, f"Có cột trong dữ liệu nhưng thiếu trong config: {missing_in_config}"
    assert not extra_in_config, f"Có cột trong config nhưng không có trong dữ liệu: {extra_in_config}"

    print("[V] XÁC NHẬN: Mọi cột đều khớp 100%, không bị bỏ sót hoặc thừa cột nào!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    validate_config()

