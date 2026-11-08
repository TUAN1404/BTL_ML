"""Script điều phối tiền xử lý và chia dữ liệu (Data Preparation Pipeline)
File: prepare_data.py

Nhiệm vụ:
1. Bộ 1 (King County):
   - Đọc data/raw/kc_house_data.csv, gọi hàm clean_d1.
   - Chia 80/20 bằng GroupShuffleSplit(test_size=0.2, random_state=42), groups = cột 'id'
     để triệt tiêu hoàn toàn rò rỉ dữ liệu (cùng một căn nhà không rơi vào cả 2 tập).
   - Bỏ cột 'id' rồi lưu vào data/processed/d1_train.csv và d1_test.csv.

2. Bộ 2 (Indian Urban Housing):
   - Kiểm tra data/raw/test.csv có cột 'Price_INR_Lakhs' không:
     + Nếu KHÔNG: chỉ dùng train.csv, gọi hàm clean_d2, chia train_test_split(test_size=0.2, random_state=42),
       lưu vào data/processed/d2_train.csv và d2_test.csv.
     + Nếu CÓ: hỏi ý kiến người dùng trước khi quyết định.

3. In shape của cả 4 file và đảm bảo chạy lại nhiều lần cho kết quả nhất quán 100%.
"""

import os
import sys
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

# Import các hàm làm sạch chuyên biệt từ từng thư mục
from d1.clean_d1 import clean_d1
from d2.clean_d2 import clean_d2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")


def prepare_dataset_1():
    print("\n" + "=" * 70)
    print("TIẾN TRÌNH XỬ LÝ BỘ 1: King County House Sales")
    print("=" * 70)

    raw_path = os.path.join(DATA_RAW, "kc_house_data.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Không tìm thấy file: {raw_path}")

    # 1. Đọc và làm sạch
    df_raw = pd.read_csv(raw_path)
    df_clean = clean_d1(df_raw)

    # 2. Chia tập 80/20 bằng GroupShuffleSplit theo cột id
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df_clean, groups=df_clean["id"]))

    d1_train = df_clean.iloc[train_idx].copy()
    d1_test = df_clean.iloc[test_idx].copy()

    # Kiểm tra rò rỉ ID
    overlap_ids = set(d1_train["id"]).intersection(set(d1_test["id"]))
    print(f"[GroupShuffleSplit] Số ID bị trùng lặp giữa Train và Test: {len(overlap_ids)} (Kỳ vọng: 0)")

    # 3. Bỏ cột id trước khi lưu để mô hình không học nhầm
    d1_train.drop(columns=["id"], inplace=True)
    d1_test.drop(columns=["id"], inplace=True)

    # 4. Lưu vào data/processed/ (bao gồm cả file clean tổng thể)
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    train_file = os.path.join(DATA_PROCESSED, "d1_train.csv")
    test_file = os.path.join(DATA_PROCESSED, "d1_test.csv")
    clean_file = os.path.join(DATA_PROCESSED, "d1_clean.csv")

    d1_train.to_csv(train_file, index=False)
    d1_test.to_csv(test_file, index=False)
    d1_clean = df_clean.drop(columns=["id"])
    d1_clean.to_csv(clean_file, index=False)

    print(f"[V] Đã lưu d1_train.csv: {d1_train.shape[0]:,} dòng × {d1_train.shape[1]} cột")
    print(f"[V] Đã lưu d1_test.csv : {d1_test.shape[0]:,} dòng × {d1_test.shape[1]} cột")
    print(f"[V] Đã lưu d1_clean.csv: {d1_clean.shape[0]:,} dòng × {d1_clean.shape[1]} cột")

    return d1_train, d1_test


def prepare_dataset_2():
    print("\n" + "=" * 70)
    print("TIẾN TRÌNH XỬ LÝ BỘ 2: Indian Urban Housing")
    print("=" * 70)

    train_raw_path = os.path.join(DATA_RAW, "train.csv")
    test_raw_path = os.path.join(DATA_RAW, "test.csv")

    if not os.path.exists(train_raw_path):
        raise FileNotFoundError(f"Không tìm thấy file: {train_raw_path}")

    # 1. Kiểm tra test.csv có nhãn Price_INR_Lakhs hay không
    has_target_in_test = False
    if os.path.exists(test_raw_path):
        test_cols = pd.read_csv(test_raw_path, nrows=1).columns
        has_target_in_test = "Price_INR_Lakhs" in test_cols

    if has_target_in_test:
        print("[!] Phát hiện file test.csv có sẵn cột 'Price_INR_Lakhs'.")
        choice = input("Bạn muốn dùng test.csv chính thức hay tự chia 80/20 từ train.csv? (1: Dùng test.csv, 2: Tự chia): ")
        if choice.strip() == "1":
            df_train_raw = pd.read_csv(train_raw_path)
            df_test_raw = pd.read_csv(test_raw_path)
            d2_train = clean_d2(df_train_raw)
            d2_test = clean_d2(df_test_raw)
        else:
            df_train_raw = pd.read_csv(train_raw_path)
            df_clean = clean_d2(df_train_raw)
            d2_train, d2_test = train_test_split(df_clean, test_size=0.2, random_state=42)
    else:
        print("[*] File test.csv KHÔNG có cột mục tiêu 'Price_INR_Lakhs' (tập test kín của Kaggle).")
        print("    -> Quy tắc: Chỉ dùng train.csv, làm sạch và tự chia 80/20 (random_state=42).")
        df_train_raw = pd.read_csv(train_raw_path)
        df_clean = clean_d2(df_train_raw)
        d2_train, d2_test = train_test_split(df_clean, test_size=0.2, random_state=42)

    # 2. Lưu vào data/processed/ (bao gồm cả file clean tổng thể)
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    train_file = os.path.join(DATA_PROCESSED, "d2_train.csv")
    test_file = os.path.join(DATA_PROCESSED, "d2_test.csv")
    clean_file = os.path.join(DATA_PROCESSED, "d2_clean.csv")

    d2_train.to_csv(train_file, index=False)
    d2_test.to_csv(test_file, index=False)
    df_clean.to_csv(clean_file, index=False)

    print(f"[V] Đã lưu d2_train.csv: {d2_train.shape[0]:,} dòng × {d2_train.shape[1]} cột")
    print(f"[V] Đã lưu d2_test.csv : {d2_test.shape[0]:,} dòng × {d2_test.shape[1]} cột")
    print(f"[V] Đã lưu d2_clean.csv: {df_clean.shape[0]:,} dòng × {df_clean.shape[1]} cột")

    return d2_train, d2_test


def main():
    print(f"{'#' * 70}")
    print("# BẮT ĐẦU QUY TRÌNH CHUẨN BỊ VÀ CHIA DỮ LIỆU DÙNG CHUNG (PREPARE DATA)")
    print(f"{'#' * 70}")

    d1_train, d1_test = prepare_dataset_1()
    d2_train, d2_test = prepare_dataset_2()

    print("\n" + "#" * 70)
    print("# TỔNG KẾT SHAPE CỦA 4 FILE ĐÃ LƯU TRONG data/processed/")
    print("#" * 70)
    print(f"1. d1_train.csv : {d1_train.shape[0]:>7,} dòng × {d1_train.shape[1]} cột (King County - Tập Huấn luyện)")
    print(f"2. d1_test.csv  : {d1_test.shape[0]:>7,} dòng × {d1_test.shape[1]} cột (King County - Tập Kiểm thử)")
    print(f"3. d2_train.csv : {d2_train.shape[0]:>7,} dòng × {d2_train.shape[1]} cột (Indian Urban - Tập Huấn luyện)")
    print(f"4. d2_test.csv  : {d2_test.shape[0]:>7,} dòng × {d2_test.shape[1]} cột (Indian Urban - Tập Kiểm thử)")
    print("-" * 70)
    print("Script đã hoàn thành và sẵn sàng chạy lại bất kỳ lúc nào với kết quả đồng nhất 100%!")


if __name__ == "__main__":
    main()

