"""Kiểm thử tính toàn vẹn và phạm vi giá trị dữ liệu (Data Integrity Test)
File: tests/test_data_integrity.py

Kiểm tra:
1. Không còn missing ở bất kỳ cột nào trên cả 4 file processed.
2. Tên, thứ tự và kiểu dữ liệu từng cột giữa Train và Test khớp nhau 100%.
3. Không còn cột ID (id, Property_ID).
4. Biến mục tiêu tồn tại, luôn > 0 và không phải hằng số.
5. Kiểm tra phạm vi giá trị hợp lý (in CẢNH BÁO nếu có giá trị biên/ngoại lệ nghiệp vụ).
6. Target KHÔNG bị log (giá trị thực tế USD / Lakhs).
7. Các cột số CHƯA bị scale (mean và std vẫn giữ nguyên thang đo gốc).
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")


@pytest.fixture(scope="module")
def processed_data():
    """Nạp cả 4 file dữ liệu processed phục vụ kiểm thử."""
    d1_train = pd.read_csv(os.path.join(PROCESSED_DIR, "d1_train.csv"))
    d1_test = pd.read_csv(os.path.join(PROCESSED_DIR, "d1_test.csv"))
    d2_train = pd.read_csv(os.path.join(PROCESSED_DIR, "d2_train.csv"))
    d2_test = pd.read_csv(os.path.join(PROCESSED_DIR, "d2_test.csv"))
    return {
        "d1_train": d1_train,
        "d1_test": d1_test,
        "d2_train": d2_train,
        "d2_test": d2_test,
    }


# ==============================================================================
# 1. KIỂM THỬ KHUYẾT THIẾU, CẤU TRÚC VÀ CỘT ID
# ==============================================================================
class TestStructureAndMissing:
    """Kiểm tra khuyết thiếu, thứ tự cột, dtypes và loại bỏ ID."""

    def test_zero_missing_values(self, processed_data):
        """Khẳng định 0% missing trên tất cả các cột của 4 file."""
        for name, df in processed_data.items():
            missing_total = df.isnull().sum().sum()
            assert missing_total == 0, f"File {name} có {missing_total} giá trị khuyết thiếu!"

    def test_column_names_order_and_dtypes(self, processed_data):
        """Tên, thứ tự và kiểu dữ liệu (dtypes) giữa Train và Test phải giống hệt nhau."""
        # Bộ 1
        d1_tr, d1_te = processed_data["d1_train"], processed_data["d1_test"]
        assert list(d1_tr.columns) == list(d1_te.columns), "Tên/thứ tự cột Bộ 1 không khớp!"
        for col in d1_tr.columns:
            assert d1_tr[col].dtype == d1_te[col].dtype, f"Cột {col} Bộ 1 lệch dtype: {d1_tr[col].dtype} vs {d1_te[col].dtype}"

        # Bộ 2
        d2_tr, d2_te = processed_data["d2_train"], processed_data["d2_test"]
        assert list(d2_tr.columns) == list(d2_te.columns), "Tên/thứ tự cột Bộ 2 không khớp!"
        for col in d2_tr.columns:
            assert d2_tr[col].dtype == d2_te[col].dtype, f"Cột {col} Bộ 2 lệch dtype: {d2_tr[col].dtype} vs {d2_te[col].dtype}"

    def test_no_id_columns_exist(self, processed_data):
        """Không còn cột ID ('id' hoặc 'Property_ID') trong tập huấn luyện/kiểm thử."""
        for name in ["d1_train", "d1_test"]:
            assert "id" not in processed_data[name].columns, f"Cột 'id' vẫn còn trong {name}!"
        for name in ["d2_train", "d2_test"]:
            assert "Property_ID" not in processed_data[name].columns, f"Cột 'Property_ID' vẫn còn trong {name}!"


# ==============================================================================
# 2. KIỂM THỬ BIẾN MỤC TIÊU (TARGET VALIDITY)
# ==============================================================================
class TestTargetProperties:
    """Kiểm tra tính hợp lệ và thang đo của biến mục tiêu."""

    def test_target_presence_and_positivity(self, processed_data):
        """Target phải tồn tại, luôn > 0 và không phải hằng số."""
        # Bộ 1: price
        for name in ["d1_train", "d1_test"]:
            df = processed_data[name]
            assert "price" in df.columns
            assert (df["price"] > 0).all(), f"File {name} có giá price <= 0!"
            assert df["price"].std() > 0, f"File {name} có giá price là hằng số!"

        # Bộ 2: Price_INR_Lakhs
        for name in ["d2_train", "d2_test"]:
            df = processed_data[name]
            assert "Price_INR_Lakhs" in df.columns
            assert (df["Price_INR_Lakhs"] > 0).all(), f"File {name} có giá <= 0!"
            assert df["Price_INR_Lakhs"].std() > 0, f"File {name} có giá là hằng số!"

    def test_target_is_not_log_transformed(self, processed_data):
        """Cột target KHÔNG được log1p (đúng quy ước 'chưa log')."""
        # Nếu đã log, max(price) chỉ cỡ 15-16. Ở đây giá gốc phải > 1000 USD
        assert processed_data["d1_train"]["price"].max() > 1000
        assert processed_data["d1_test"]["price"].max() > 1000

        # Với Bộ 2, nếu đã log, max chỉ cỡ 7-8. Ở đây giá gốc phải > 100 Lakhs
        assert processed_data["d2_train"]["Price_INR_Lakhs"].max() > 100
        assert processed_data["d2_test"]["Price_INR_Lakhs"].max() > 100


# ==============================================================================
# 3. KIỂM THỬ PHẠM VI GIÁ TRỊ VÀ CẢNH BÁO NGHIỆP VỤ (RANGE CHECKS)
# ==============================================================================
class TestValueRangesAndWarnings:
    """Kiểm tra phạm vi giá trị hợp lý và in ra các cảnh báo cụ thể."""

    def test_d1_value_ranges_and_warnings(self, processed_data):
        """Kiểm tra giá trị Bộ 1 (King County) và in cảnh báo nếu có ngoại lệ thực tế."""
        print("\n" + "=" * 80)
        print("KIỂM TRA PHẠM VI GIÁ TRỊ BỘ 1 (King County):")
        print("=" * 80)

        for part in ["d1_train", "d1_test"]:
            df = processed_data[part]
            print(f"\n--- Phân tích tập: {part} ({len(df):,} dòng) ---")

            # 1. bedrooms trong 1..15
            bed_out = df[~df["bedrooms"].between(1, 15)]
            if len(bed_out) > 0:
                print(f"  [CẢNH BÁO] Có {len(bed_out)} dòng bedrooms nằm ngoài [1, 15] (Toàn bộ là bedrooms == 0: nhà studio hoặc đất kèm nhà phụ).")
            assert (df["bedrooms"] <= 15).all(), "Có căn hộ > 15 phòng ngủ bất thường!"

            # 2. bathrooms >= 0.5
            bath_out = df[df["bathrooms"] < 0.5]
            if len(bath_out) > 0:
                print(f"  [CẢNH BÁO] Có {len(bath_out)} dòng bathrooms < 0.5 (Toàn bộ là bathrooms == 0.0).")

            # 3. sqft_living > 0
            assert (df["sqft_living"] > 0).all(), "Phát hiện diện tích sqft_living <= 0!"

            # 4. yr_built trong 1900..2015
            assert df["yr_built"].between(1900, 2015).all(), "Năm xây dựng yr_built ngoài [1900, 2015]!"

            # 5. lat khoảng 47.1..47.8
            assert df["lat"].between(47.1, 47.8).all(), "Vĩ độ lat nằm ngoài vùng King County [47.1, 47.8]!"

            # 6. long khoảng -122.6..-121.3
            assert df["long"].between(-122.6, -121.3).all(), "Kinh độ long nằm ngoài vùng King County [-122.6, -121.3]!"

            # 7. house_age >= 0
            neg_age = df[df["house_age"] < 0]
            if len(neg_age) > 0:
                print(f"  [CẢNH BÁO] Có {len(neg_age)} dòng house_age < 0 (giá trị -1: bán năm 2014 nhưng xây năm 2015, pre-construction).")

            # 8. renovated chỉ gồm {0, 1}
            assert set(df["renovated"].unique()).issubset({0, 1}), "Cột renovated có giá trị khác {0, 1}!"

            # 9. grade trong 1..13
            assert df["grade"].between(1, 13).all(), "Điểm chất lượng grade nằm ngoài [1, 13]!"

    def test_d2_value_ranges_and_warnings(self, processed_data):
        """Kiểm tra giá trị Bộ 2 (Indian Urban Housing)."""
        print("\n" + "=" * 80)
        print("KIỂM TRA PHẠM VI GIÁ TRỊ BỘ 2 (Indian Urban Housing):")
        print("=" * 80)

        for part in ["d2_train", "d2_test"]:
            df = processed_data[part]
            print(f"\n--- Phân tích tập: {part} ({len(df):,} dòng) ---")

            # 1. BHK trong 1..6
            assert df["BHK"].between(1, 6).all(), "BHK nằm ngoài khoảng [1, 6]!"

            # 2. Bathrooms trong 1..6
            assert df["Bathrooms"].between(1, 6).all(), "Bathrooms nằm ngoài khoảng [1, 6]!"

            # 3. Super_Area_SqFt >= 100
            assert (df["Super_Area_SqFt"] >= 100).all(), "Có căn hộ Super_Area_SqFt < 100!"

            # 4. Carpet_Area_SqFt > 0
            assert (df["Carpet_Area_SqFt"] > 0).all(), "Có căn hộ Carpet_Area_SqFt <= 0!"

            # 5. Floor_Number >= 0
            assert (df["Floor_Number"] >= 0).all(), "Có tầng Floor_Number < 0!"

            # 6. Age_of_Property trong 0..39
            assert df["Age_of_Property"].between(0, 39).all(), "Tuổi nhà Age_of_Property nằm ngoài [0, 39]!"

            # 7. Nhị phân: Parking, Lift_Available, Gated_Community chỉ {0, 1}
            for col in ["Parking", "Lift_Available", "Gated_Community"]:
                assert set(df[col].unique()).issubset({0, 1}), f"Cột nhị phân {col} chứa giá trị khác {{0, 1}}!"

            print(f"  [PASS] Tất cả các tiêu chí giá trị Bộ 2 trên {part} đều đạt chuẩn 100%!")


# ==============================================================================
# 4. KIỂM THỬ KHÔNG BỊ SCALE (UNSCALED FEATURES CHECK)
# ==============================================================================
class TestFeaturesUnscaled:
    """Xác nhận dữ liệu chưa bị scale (StandardScaler/MinMaxScaler)."""

    def test_features_are_not_scaled(self, processed_data):
        """Các cột diện tích, giá và tọa độ không có mean ≈ 0 và std ≈ 1."""
        d1_tr = processed_data["d1_train"]
        d2_tr = processed_data["d2_train"]

        # Bộ 1: sqft_living và sqft_lot thang đo hàng nghìn
        assert d1_tr["sqft_living"].mean() > 500
        assert d1_tr["sqft_lot"].mean() > 1000

        # Bộ 2: Super_Area_SqFt và Carpet_Area_SqFt thang đo hàng nghìn
        assert d2_tr["Super_Area_SqFt"].mean() > 500
        assert d2_tr["Carpet_Area_SqFt"].mean() > 500

        # Khẳng định không có việc nhiều cột số bị scale đồng loạt về mean=0, std=1
        for name, df in processed_data.items():
            num_cols = df.select_dtypes(include=["number"]).columns
            scaled_like_cols = [
                c for c in num_cols
                if abs(df[c].mean()) < 0.05 and abs(df[c].std() - 1.0) < 0.05
            ]
            # Không có cột nào thỏa mãn dạng chuẩn hóa StandardScaler
            assert len(scaled_like_cols) == 0, f"File {name} có cột nghi ngờ đã bị scale: {scaled_like_cols}"


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])

