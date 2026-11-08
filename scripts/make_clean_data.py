"""Script thực hiện toàn bộ quy trình tiền xử lý và sinh ra 2 file CSV sạch:
1. data/processed/d1_clean.csv (King County)
2. data/processed/d2_clean.csv (Indian Urban Housing)

Người thực hiện: Người 1 (Data Lead)
"""

import os
import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")


def clean_king_county():
    print("--- ĐANG TIỀN XỬ LÝ BỘ 1: King County ---")
    raw_path = os.path.join(
        DATA_DIR, "d1_king_county", "kc_house_data.csv"
    )  # hoặc data/raw/kc_house_data.csv
    if not os.path.exists(raw_path):
        raw_path = os.path.join(DATA_DIR, "raw", "kc_house_data.csv")

    if not os.path.exists(raw_path):
        print(f"[!] Không tìm thấy file gốc tại {raw_path}")
        return

    df = pd.read_csv(raw_path)
    print(f"Số lượng ban đầu: {df.shape}")

    # 1. Bỏ cột id
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # 2. Xóa ngoại lai 33 phòng ngủ (id 15870)
    df = df[df["bedrooms"] < 30]

    # 3. Trích xuất thời gian từ date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df["sale_year"] = df["date"].dt.year
        df["sale_month"] = df["date"].dt.month
        df = df.drop(columns=["date"])

    # 4. Tạo feature: house_age và renovated
    df["house_age"] = df["sale_year"] - df["yr_built"]
    df["renovated"] = (df["yr_renovated"] > 0).astype(int)

    out_file = os.path.join(PROCESSED_DIR, "d1_clean.csv")
    df.to_csv(out_file, index=False)
    print(f"[V] Đã xuất {out_file} - Shape: {df.shape} (Kỳ vọng: (21612, 22))")


def clean_indian_urban():
    print("\n--- ĐANG TIỀN XỬ LÝ BỘ 2: Indian Urban Housing ---")
    raw_path = os.path.join(
        DATA_DIR, "d2_indian_urban", "train.csv"
    )  # hoặc data/raw/train.csv
    if not os.path.exists(raw_path):
        raw_path = os.path.join(DATA_DIR, "raw", "train.csv")

    if not os.path.exists(raw_path):
        print(f"[!] Không tìm thấy file gốc tại {raw_path}")
        return

    df = pd.read_csv(raw_path)
    print(f"Số lượng ban đầu: {df.shape}")

    # 1. Bỏ cột Property_ID
    if "Property_ID" in df.columns:
        df = df.drop(columns=["Property_ID"])

    # 2. Lọc bỏ 33 căn hộ có diện tích cực nhỏ < 100 sqft
    df = df[df["Super_Area_SqFt"] >= 100]

    # 3. Kiểm tra logic
    df = df[df["Floor_Number"] <= df["Total_Floors"]]
    df = df[df["Carpet_Area_SqFt"] <= df["Super_Area_SqFt"]]

    out_file = os.path.join(PROCESSED_DIR, "d2_clean.csv")
    df.to_csv(out_file, index=False)
    print(f"[V] Đã xuất {out_file} - Shape: {df.shape} (Kỳ vọng: (79967, 17))")


if __name__ == "__main__":
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    clean_king_county()
    clean_indian_urban()
    print("\nHoàn tất tạo 2 file CSV sạch!")

