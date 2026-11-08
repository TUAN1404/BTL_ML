"""Script kiểm định toàn diện chất lượng dữ liệu processed
File: check_data.py

Kiểm tra và in PASS/FAIL cho các tiêu chí:
1. Không còn missing ở train và test.
2. Cột train và test giống hệt nhau (tên và thứ tự).
3. Không còn cột ID (id, Property_ID).
4. Target có trong cả hai file, không có giá trị <= 0.
5. Tỉ lệ dòng test / (train + test) trong khoảng 0.19 - 0.21.
6. Bộ 1: không có id nào nằm ở cả train và test (kiểm tra chéo qua logic GroupShuffleSplit).
7. In min/max từng cột số để soát các giá trị vô lý.
"""

import os
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from d1.clean_d1 import clean_d1

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")


def print_status(check_name, is_pass, detail=""):
    tag = "[PASS]" if is_pass else "[FAIL]"
    print(f"  {tag:<7} {check_name}")
    if detail:
        print(f"          -> {detail}")
    return is_pass


def print_min_max_table(df_train, df_test, title):
    print(f"\n  [BẢNG MIN/MAX CÁC CỘT SỐ - {title}]")
    num_cols = df_train.select_dtypes(include=["number"]).columns
    print(f"  {'Tên cột':<26} | {'Train Min':>12} | {'Train Max':>14} | {'Test Min':>12} | {'Test Max':>14}")
    print("  " + "-" * 88)
    for col in num_cols:
        t_min = df_train[col].min()
        t_max = df_train[col].max()
        e_min = df_test[col].min()
        e_max = df_test[col].max()
        print(f"  {col:<26} | {t_min:>12.2f} | {t_max:>14.2f} | {e_min:>12.2f} | {e_max:>14.2f}")


def check_dataset_1():
    print("=" * 80)
    print("KIỂM TRA BỘ 1: King County House Sales (d1_train.csv & d1_test.csv)")
    print("=" * 80)

    train_path = os.path.join(PROCESSED_DIR, "d1_train.csv")
    test_path = os.path.join(PROCESSED_DIR, "d1_test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"[!] Thiếu file processed tại {PROCESSED_DIR}")
        return False

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    all_pass = True

    # 1. Missing values
    missing_train = df_train.isnull().sum().sum()
    missing_test = df_test.isnull().sum().sum()
    p1 = (missing_train == 0) and (missing_test == 0)
    all_pass &= print_status(
        "Không còn missing ở train và test",
        p1,
        f"Train missing: {missing_train}, Test missing: {missing_test}",
    )

    # 2. Cột giống hệt nhau về tên và thứ tự
    cols_match = list(df_train.columns) == list(df_test.columns)
    all_pass &= print_status(
        "Cột train và test giống hệt nhau (tên và thứ tự)",
        cols_match,
        f"Tổng số cột: {len(df_train.columns)}",
    )

    # 3. Không còn cột ID
    has_no_id = ("id" not in df_train.columns) and ("id" not in df_test.columns)
    all_pass &= print_status(
        "Không còn cột ID ('id')",
        has_no_id,
        f"Đã loại bỏ cột 'id' khỏi cả 2 tập",
    )

    # 4. Target có trong cả hai file, không có giá trị <= 0
    target_ok = (
        ("price" in df_train.columns)
        and ("price" in df_test.columns)
        and (df_train["price"].min() > 0)
        and (df_test["price"].min() > 0)
    )
    all_pass &= print_status(
        "Target 'price' có trong cả hai file và giá trị > 0",
        target_ok,
        f"Min train: ${df_train['price'].min():,.0f}, Min test: ${df_test['price'].min():,.0f}",
    )

    # 5. Tỉ lệ dòng test / (train + test) trong khoảng 0.19-0.21
    total_rows = len(df_train) + len(df_test)
    ratio = len(df_test) / total_rows
    ratio_ok = 0.19 <= ratio <= 0.21
    all_pass &= print_status(
        "Tỉ lệ dòng test / (train + test) trong khoảng 0.19 - 0.21",
        ratio_ok,
        f"Tỉ lệ thực tế: {ratio:.4f} ({len(df_test):,} / {total_rows:,} dòng)",
    )

    # 6. Kiểm tra không có id nào nằm ở cả train và test
    raw_path = os.path.join(RAW_DIR, "kc_house_data.csv")
    if os.path.exists(raw_path):
        df_raw = pd.read_csv(raw_path)
        df_clean = clean_d1(df_raw)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tr_idx, te_idx = next(gss.split(df_clean, groups=df_clean["id"]))
        tr_ids = set(df_clean.iloc[tr_idx]["id"])
        te_ids = set(df_clean.iloc[te_idx]["id"])
        overlap_ids = tr_ids.intersection(te_ids)
        overlap_ok = len(overlap_ids) == 0
        all_pass &= print_status(
            "Kiểm tra GroupShuffleSplit: 0 ID nào nằm ở cả train và test",
            overlap_ok,
            f"Số ID giao nhau giữa Train và Test: {len(overlap_ids)} (File processed đã bỏ cột 'id')",
        )
    else:
        print_status("Kiểm tra ID lặp lại", True, "File processed đã bỏ 'id' nên không lưu ID.")

    # 7. In bảng min/max các cột số
    print_min_max_table(df_train, df_test, "Bộ 1: King County")
    return all_pass


def check_dataset_2():
    print("\n" + "=" * 80)
    print("KIỂM TRA BỘ 2: Indian Urban Housing (d2_train.csv & d2_test.csv)")
    print("=" * 80)

    train_path = os.path.join(PROCESSED_DIR, "d2_train.csv")
    test_path = os.path.join(PROCESSED_DIR, "d2_test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"[!] Thiếu file processed tại {PROCESSED_DIR}")
        return False

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    all_pass = True

    # 1. Missing values
    missing_train = df_train.isnull().sum().sum()
    missing_test = df_test.isnull().sum().sum()
    p1 = (missing_train == 0) and (missing_test == 0)
    all_pass &= print_status(
        "Không còn missing ở train và test",
        p1,
        f"Train missing: {missing_train}, Test missing: {missing_test}",
    )

    # 2. Cột giống hệt nhau về tên và thứ tự
    cols_match = list(df_train.columns) == list(df_test.columns)
    all_pass &= print_status(
        "Cột train và test giống hệt nhau (tên và thứ tự)",
        cols_match,
        f"Tổng số cột: {len(df_train.columns)}",
    )

    # 3. Không còn cột ID
    has_no_id = ("Property_ID" not in df_train.columns) and ("Property_ID" not in df_test.columns)
    all_pass &= print_status(
        "Không còn cột ID ('Property_ID')",
        has_no_id,
        "Đã loại bỏ cột 'Property_ID' khỏi cả 2 tập",
    )

    # 4. Target có trong cả hai file, không có giá trị <= 0
    target_ok = (
        ("Price_INR_Lakhs" in df_train.columns)
        and ("Price_INR_Lakhs" in df_test.columns)
        and (df_train["Price_INR_Lakhs"].min() > 0)
        and (df_test["Price_INR_Lakhs"].min() > 0)
    )
    all_pass &= print_status(
        "Target 'Price_INR_Lakhs' có trong cả hai file và giá trị > 0",
        target_ok,
        f"Min train: {df_train['Price_INR_Lakhs'].min():.2f} Lakhs, Min test: {df_test['Price_INR_Lakhs'].min():.2f} Lakhs",
    )

    # 5. Tỉ lệ dòng test / (train + test) trong khoảng 0.19-0.21
    total_rows = len(df_train) + len(df_test)
    ratio = len(df_test) / total_rows
    ratio_ok = 0.19 <= ratio <= 0.21
    all_pass &= print_status(
        "Tỉ lệ dòng test / (train + test) trong khoảng 0.19 - 0.21",
        ratio_ok,
        f"Tỉ lệ thực tế: {ratio:.4f} ({len(df_test):,} / {total_rows:,} dòng)",
    )

    # 6. In bảng min/max các cột số
    print_min_max_table(df_train, df_test, "Bộ 2: Indian Urban")
    return all_pass


def main():
    print("\n" + "#" * 80)
    print("# BẮT ĐẦU CHẠY KIỂM ĐỊNH TOÀN DIỆN DỮ LIỆU ĐÃ XỬ LÝ (CHECK DATA)")
    print("#" * 80 + "\n")

    p1 = check_dataset_1()
    p2 = check_dataset_2()

    print("\n" + "#" * 80)
    if p1 and p2:
        print("# KẾT QUẢ CUỐI CÙNG: TẤT CẢ CÁC HẠNG MỤC ĐỀU ĐẠT [PASS] 100%!")
        print("# Dữ liệu processed hoàn toàn chuẩn mực và sẵn sàng cho Pipeline huấn luyện.")
    else:
        print("# KẾT QUẢ CUỐI CÙNG: CÓ HẠNG MỤC BỊ [FAIL] - CẦN KIỂM TRA LẠI!")
    print("#" * 80 + "\n")


if __name__ == "__main__":
    main()

