"""Module làm sạch dữ liệu Bộ 1: King County House Sales
File: d1/clean_d1.py

Chức năng:
- Hàm clean_d1(df) -> df
- Loại bỏ dòng bedrooms == 33 (lỗi gõ phím)
- Parse date (%Y%m%dT%H%M%S), tạo sale_year, sale_month
- Tạo house_age = sale_year - yr_built (in cảnh báo nếu âm, đề xuất cách xử lý)
- Tạo renovated = (yr_renovated > 0) dạng 0/1, bỏ yr_renovated và date
- GIỮ lại cột id để phục vụ chia train/test theo nhóm (GroupShuffleSplit)
- Không one-hot, không scale, không log, giữ nguyên price
"""

import os
import pandas as pd


def clean_d1(df: pd.DataFrame) -> pd.DataFrame:
    """Làm sạch dữ liệu King County theo quy tắc logic rõ ràng.

    Tham số:
        df (pd.DataFrame): DataFrame dữ liệu gốc (kc_house_data.csv).

    Trả về:
        pd.DataFrame: DataFrame sau khi làm sạch (không sửa DataFrame đầu vào).
    """
    # 1. Tạo bản sao độc lập, không thay đổi df gốc
    cleaned_df = df.copy()
    initial_rows = len(cleaned_df)
    print(f"[clean_d1] Số dòng trước khi làm sạch: {initial_rows:,}")

    # 2. Xử lý dòng bedrooms == 33 (lỗi gõ phím)
    bad_bed_mask = cleaned_df["bedrooms"] == 33
    bad_bed_count = bad_bed_mask.sum()
    if bad_bed_count > 0:
        cleaned_df = cleaned_df[~bad_bed_mask]
        print(
            f"[clean_d1] Đã loại bỏ {bad_bed_count} dòng có 'bedrooms == 33' (lỗi gõ phím/typo)."
        )

    # 3. Parse cột date (định dạng %Y%m%dT%H%M%S), tạo sale_year, sale_month
    date_series = pd.to_datetime(cleaned_df["date"], format="%Y%m%dT%H%M%S")
    cleaned_df["sale_year"] = date_series.dt.year
    cleaned_df["sale_month"] = date_series.dt.month

    # 4. Tạo house_age = sale_year - yr_built
    cleaned_df["house_age"] = cleaned_df["sale_year"] - cleaned_df["yr_built"]
    neg_age_count = (cleaned_df["house_age"] < 0).sum()
    if neg_age_count > 0:
        print(
            f"[clean_d1] [CẢNH BÁO] Phát hiện {neg_age_count} dòng có house_age < 0 (giá trị -1)."
        )
        print(
            "           Đề xuất xử lý: Các căn nhà này bán năm 2014 nhưng năm xây ghi 2015 "
            "(nhà bán trên giấy/pre-construction). Nên dùng np.maximum(0, house_age) để đưa về 0. "
            "(Hiện tại giữ nguyên giá trị theo yêu cầu, chưa tự xóa/sửa)."
        )

    # 5. Tạo renovated = (yr_renovated > 0) dạng 0/1, bỏ yr_renovated và date
    cleaned_df["renovated"] = (cleaned_df["yr_renovated"] > 0).astype(int)
    cleaned_df.drop(columns=["yr_renovated", "date"], inplace=True)

    # 6. GIỮ lại cột id (dùng để chia train/test theo nhóm GroupShuffleSplit)
    # Price giữ nguyên gốc, không scale, không log, không one-hot

    final_rows = len(cleaned_df)
    print(
        f"[clean_d1] Số dòng sau khi làm sạch: {final_rows:,} (chênh lệch: -{initial_rows - final_rows} dòng)"
    )
    print(f"[clean_d1] Kích thước kết quả: {cleaned_df.shape}")

    return cleaned_df


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_csv = os.path.join(BASE_DIR, "data", "raw", "kc_house_data.csv")

    if os.path.exists(raw_csv):
        raw_df = pd.read_csv(raw_csv)
        result_df = clean_d1(raw_df)
        print("\nKiểm tra 5 dòng đầu kết quả:")
        print(
            result_df[
                [
                    "id",
                    "price",
                    "bedrooms",
                    "sale_year",
                    "sale_month",
                    "house_age",
                    "renovated",
                ]
            ].head()
        )
    else:
        print(f"Không tìm thấy file: {raw_csv}")
