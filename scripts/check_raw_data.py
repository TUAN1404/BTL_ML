"""Script kiểm tra dữ liệu thô (Raw Data Audit)
Chỉ in báo cáo, tuyệt đối không chỉnh sửa dữ liệu.
"""

import os
import pandas as pd

# Đường dẫn file dữ liệu thô
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D1_RAW = os.path.join(BASE_DIR, "data", "raw", "kc_house_data.csv")
D2_TRAIN_RAW = os.path.join(BASE_DIR, "data", "raw", "train.csv")
D2_TEST_RAW = os.path.join(BASE_DIR, "data", "raw", "test.csv")


def inspect_general(df, name):
    print(f"\n{'=' * 70}")
    print(f"BÁO CÁO TỔNG QUAN: {name}")
    print(f"{'=' * 70}")
    print(f"- Kích thước (Shape): {df.shape[0]:,} dòng × {df.shape[1]} cột")
    print(f"- Số dòng trùng lặp toàn bộ (Exact duplicates): {df.duplicated().sum()}")

    # Missing values
    missing = df.isnull().sum()
    has_missing = missing[missing > 0]
    print(f"- Số cột bị thiếu dữ liệu: {len(has_missing)}")
    if len(has_missing) > 0:
        print("  Chi tiết missing:")
        for col, cnt in has_missing.items():
            print(f"  + {col}: {cnt:,} dòng ({cnt / len(df) * 100:.2f}%)")
    else:
        print("  -> Tuyệt vời: 0% giá trị khuyết thiếu (0 missing values)")

    # Data types
    print("\n- Kiểu dữ liệu từng cột (Data Types):")
    for col, dtype in df.dtypes.items():
        print(f"  {col:<28}: {str(dtype):<10}")

    # Describe numeric columns
    num_cols = df.select_dtypes(include=["number"]).columns
    print(f"\n- Thống kê mô tả describe() ({len(num_cols)} cột số):")
    print(df[num_cols].describe().round(2).to_string())

    # Categorical columns unique values
    cat_cols = df.select_dtypes(include=["object"]).columns
    print(f"\n- Giá trị unique ({len(cat_cols)} cột phân loại/chuỗi):")
    for col in cat_cols:
        uniq = df[col].unique()
        if len(uniq) <= 15:
            print(f"  + {col} ({len(uniq)} giá trị): {list(uniq)}")
        else:
            print(f"  + {col} ({len(uniq)} giá trị - mẫu 5 đầu): {list(uniq[:5])}...")


def check_dataset_1():
    print(f"\n{'#' * 70}")
    print("# KIỂM TRA CHUYÊN SÂU BỘ 1: King County House Sales (kc_house_data.csv)")
    print(f"{'#' * 70}")

    df1 = pd.read_csv(D1_RAW)
    inspect_general(df1, "Bộ 1: King County House Sales")

    print(f"\n--- CÁC KIỂM TRA ĐẶC THÙ BỘ 1 ---")

    # 1. Trùng lặp ID
    dup_ids = df1["id"].duplicated().sum()
    print(f"1. Số ID bị lặp lại (df['id'].duplicated().sum()): {dup_ids:,} dòng")
    if dup_ids > 0:
        sample_dup_id = df1[df1["id"].duplicated(keep=False)]["id"].iloc[0]
        sample_rows = df1[df1["id"] == sample_dup_id][["id", "date", "price", "bedrooms", "bathrooms", "sqft_living"]]
        print(f"   -> Ví dụ một ID lặp lại ({sample_dup_id}):")
        print(sample_rows.to_string(index=False))
        print("   -> Nhận xét: Cùng một căn nhà nhưng ngày bán và giá bán khác nhau -> Đây là NHÀ ĐƯỢC BÁN LẠI.")

    # 2. Số dòng bất thường về phòng
    bed_0 = (df1["bedrooms"] == 0).sum()
    bath_0 = (df1["bathrooms"] == 0).sum()
    bed_33 = (df1["bedrooms"] == 33).sum()
    print(f"2. Kiểm tra số phòng:")
    print(f"   - Số dòng bedrooms == 0 : {bed_0} dòng")
    print(f"   - Số dòng bathrooms == 0: {bath_0} dòng")
    print(f"   - Số dòng bedrooms == 33: {bed_33} dòng")
    if bed_33 > 0:
        row_33 = df1[df1["bedrooms"] == 33][["id", "price", "bedrooms", "sqft_living", "sqft_lot"]]
        print(f"     Chi tiết dòng 33 phòng ngủ:\n{row_33.to_string(index=False)}")

    # 3. Định dạng date
    print(f"3. Kiểm tra định dạng cột 'date':")
    sample_dates = df1["date"].head(3).tolist()
    print(f"   - Mẫu giá trị date: {sample_dates}")
    try:
        parsed_dates = pd.to_datetime(df1["date"], format="%Y%m%dT%H%M%S", errors="raise")
        print(f"   - Parse thử định dạng '%Y%m%dT%H%M%S': THÀNH CÔNG 100%, không có lỗi!")
    except Exception as e:
        print(f"   - Parse thử bị lỗi: {e}")

    # 4. Kiểm tra house_age âm
    sale_year = pd.to_datetime(df1["date"]).dt.year
    house_age = sale_year - df1["yr_built"]
    neg_age = (house_age < 0).sum()
    print(f"4. Kiểm tra house_age (năm bán - yr_built):")
    print(f"   - Số lượng giá trị âm (house_age < 0): {neg_age} dòng")
    if neg_age > 0:
        print(df1[house_age < 0][["id", "date", "yr_built"]].head())


def check_dataset_2():
    print(f"\n{'#' * 70}")
    print("# KIỂM TRA CHUYÊN SÂU BỘ 2: Indian Urban Housing (train.csv & test.csv)")
    print(f"{'#' * 70}")

    df2_train = pd.read_csv(D2_TRAIN_RAW)
    df2_test = pd.read_csv(D2_TEST_RAW)

    inspect_general(df2_train, "Bộ 2: Indian Urban Housing (Train)")

    print(f"\n--- CÁC KIỂM TRA ĐẶC THÙ BỘ 2 ---")

    # 1. Kiểm tra test.csv có target không
    has_target_in_test = "Price_INR_Lakhs" in df2_test.columns
    print(f"1. File test.csv có cột 'Price_INR_Lakhs' không: {has_target_in_test}")
    print(f"   (Số cột train: {df2_train.shape[1]} cột vs Số cột test: {df2_test.shape[1]} cột)")
    print("   -> Kết luận: test.csv là tập test kín của Kaggle (không có nhãn giá). Nhóm PHẢI tự chia train/test từ train.csv.")

    # 2. Super_Area_SqFt < 100
    small_area = (df2_train["Super_Area_SqFt"] < 100).sum()
    print(f"2. Số dòng Super_Area_SqFt < 100: {small_area} dòng")
    if small_area > 0:
        print(f"   - Diện tích nhỏ nhất: {df2_train['Super_Area_SqFt'].min():.2f} sqft")

    # 3. Floor_Number > Total_Floors
    floor_invalid = (df2_train["Floor_Number"] > df2_train["Total_Floors"]).sum()
    print(f"3. Số dòng Floor_Number > Total_Floors: {floor_invalid} dòng")

    # 4. Carpet_Area_SqFt > Super_Area_SqFt
    carpet_invalid = (df2_train["Carpet_Area_SqFt"] > df2_train["Super_Area_SqFt"]).sum()
    print(f"4. Số dòng Carpet_Area_SqFt > Super_Area_SqFt: {carpet_invalid} dòng")

    # 5. Unique của Locality_Type và Furnishing_Status
    print(f"5. Giá trị unique và phân bố của các cột thứ bậc (Ordinal):")
    print("   - Locality_Type:")
    for val, count in df2_train["Locality_Type"].value_counts().items():
        print(f"     + '{val}': {count:,} dòng")
    print("   - Furnishing_Status:")
    for val, count in df2_train["Furnishing_Status"].value_counts().items():
        print(f"     + '{val}': {count:,} dòng")


if __name__ == "__main__":
    check_dataset_1()
    check_dataset_2()

