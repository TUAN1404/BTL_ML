"""Module làm sạch dữ liệu Bộ 2: Indian Urban Housing
File: d2/clean_d2.py

Chức năng:
- Hàm clean_d2(df) -> df
- Bỏ cột Property_ID
- Bỏ các dòng Super_Area_SqFt < 100 (lỗi nhập liệu diện tích cực nhỏ)
- Kiểm tra logic:
  + Floor_Number > Total_Floors
  + Carpet_Area_SqFt > Super_Area_SqFt
  In số dòng vi phạm. Chỉ bỏ nếu < 1% tổng số dòng, ngược lại cảnh báo.
- Không one-hot, không scale, không log, giữ Price_INR_Lakhs nguyên gốc
- Không sửa df đầu vào (dùng copy)
- In số dòng trước và sau khi làm sạch
"""

import os
import pandas as pd


def clean_d2(df: pd.DataFrame) -> pd.DataFrame:
    """Làm sạch dữ liệu Indian Urban Housing theo quy tắc logic rõ ràng.

    Tham số:
        df (pd.DataFrame): DataFrame dữ liệu gốc (train.csv).

    Trả về:
        pd.DataFrame: DataFrame sau khi làm sạch (không sửa DataFrame đầu vào).
    """
    # 1. Tạo bản sao độc lập, không sửa df gốc
    cleaned_df = df.copy()
    initial_rows = len(cleaned_df)
    threshold_1_percent = 0.01 * initial_rows
    print(f"[clean_d2] Số dòng trước khi làm sạch: {initial_rows:,}")

    # 2. Bỏ cột Property_ID (định danh không có giá trị học máy)
    if "Property_ID" in cleaned_df.columns:
        cleaned_df.drop(columns=["Property_ID"], inplace=True)
        print("[clean_d2] Đã loại bỏ cột 'Property_ID'.")

    # 3. Bỏ các dòng Super_Area_SqFt < 100 (lỗi nhập liệu)
    small_area_mask = cleaned_df["Super_Area_SqFt"] < 100
    small_area_count = small_area_mask.sum()
    if small_area_count > 0:
        cleaned_df = cleaned_df[~small_area_mask]
        print(
            f"[clean_d2] Đã loại bỏ {small_area_count} dòng có 'Super_Area_SqFt < 100' "
            f"(diện tích cực nhỏ không hợp lý đối với nhà ở đô thị)."
        )

    # 4. Kiểm tra logic 1: Floor_Number > Total_Floors
    floor_viol_mask = cleaned_df["Floor_Number"] > cleaned_df["Total_Floors"]
    floor_viol_count = floor_viol_mask.sum()
    print(
        f"[clean_d2] Kiểm tra logic 'Floor_Number > Total_Floors': Vi phạm {floor_viol_count:,} dòng "
        f"({floor_viol_count / initial_rows * 100:.2f}%)."
    )
    if floor_viol_count > 0:
        if floor_viol_count <= threshold_1_percent:
            cleaned_df = cleaned_df[~floor_viol_mask]
            print(
                f"[clean_d2] -> Số vi phạm < 1% ({floor_viol_count} dòng) -> Đã loại bỏ."
            )
        else:
            print(
                f"[clean_d2] [CẢNH BÁO] Số vi phạm vượt quá 1% ({floor_viol_count:,} dòng). "
                "Cần báo cáo và xác nhận trước khi xóa (chưa xóa)."
            )

    # 5. Kiểm tra logic 2: Carpet_Area_SqFt > Super_Area_SqFt
    carpet_viol_mask = (
        cleaned_df["Carpet_Area_SqFt"] > cleaned_df["Super_Area_SqFt"]
    )
    carpet_viol_count = carpet_viol_mask.sum()
    print(
        f"[clean_d2] Kiểm tra logic 'Carpet_Area_SqFt > Super_Area_SqFt': Vi phạm {carpet_viol_count:,} dòng "
        f"({carpet_viol_count / initial_rows * 100:.2f}%)."
    )
    if carpet_viol_count > 0:
        if carpet_viol_count <= threshold_1_percent:
            cleaned_df = cleaned_df[~carpet_viol_mask]
            print(
                f"[clean_d2] -> Số vi phạm < 1% ({carpet_viol_count} dòng) -> Đã loại bỏ."
            )
        else:
            print(
                f"[clean_d2] [CẢNH BÁO] Số vi phạm vượt quá 1% ({carpet_viol_count:,} dòng). "
                "Cần báo cáo và xác nhận trước khi xóa (chưa xóa)."
            )

    # 6. Không one-hot, không scale, không log, giữ Price_INR_Lakhs nguyên gốc
    final_rows = len(cleaned_df)
    print(
        f"[clean_d2] Số dòng sau khi làm sạch: {final_rows:,} (chênh lệch: -{initial_rows - final_rows} dòng)"
    )
    print(f"[clean_d2] Kích thước kết quả: {cleaned_df.shape}")

    return cleaned_df


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_csv = os.path.join(BASE_DIR, "data", "raw", "train.csv")

    if os.path.exists(raw_csv):
        raw_df = pd.read_csv(raw_csv)
        result_df = clean_d2(raw_df)
        print("\nKiểm tra 5 dòng đầu kết quả:")
        print(
            result_df[
                [
                    "City",
                    "Property_Type",
                    "BHK",
                    "Super_Area_SqFt",
                    "Price_INR_Lakhs",
                ]
            ].head()
        )
    else:
        print(f"Không tìm thấy file: {raw_csv}")
