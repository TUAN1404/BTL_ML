"""Bộ kiểm thử đơn vị cho hàm clean_d1 và clean_d2
File: tests/test_clean.py

Sử dụng DataFrame nhỏ tự tạo, độc lập hoàn toàn với file dữ liệu thật.
Bao gồm các kiểm tra nghiệp vụ và các trường hợp biên (edge cases).
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

# Thêm đường dẫn gốc vào sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from d1.clean_d1 import clean_d1
from d2.clean_d2 import clean_d2


# ==============================================================================
# 1. KIỂM THỬ HÀM clean_d1 (King County)
# ==============================================================================
class TestCleanD1Unit:
    """Kiểm thử đơn vị cho clean_d1 với mock DataFrame."""

    @pytest.fixture
    def mock_d1_df(self):
        """Tạo DataFrame giả lập có đầy đủ các trường hợp cần test cho clean_d1."""
        return pd.DataFrame(
            {
                "id": [101, 102, 103, 104],
                "date": [
                    "20141013T000000",
                    "20150215T000000",
                    "20140720T000000",
                    "20151105T000000",
                ],
                "price": [221900.0, 538000.0, 640000.0, 450000.0],
                "bedrooms": [3, 4, 33, 2],  # index 2 là dòng 33 phòng ngủ (ngoại lai)
                "bathrooms": [1.0, 2.5, 1.75, 1.0],
                "sqft_living": [1180, 2570, 1620, 1000],
                "yr_built": [1955, 1990, 1960, 2010],
                "yr_renovated": [0, 2005, 0, 0],  # index 1 đã cải tạo
            }
        )

    def test_bedrooms_33_dropped_and_normal_kept(self, mock_d1_df):
        """Dòng bedrooms == 33 bị loại, các dòng bình thường được giữ nguyên."""
        cleaned = clean_d1(mock_d1_df)

        assert len(cleaned) == 3
        assert 33 not in cleaned["bedrooms"].values
        assert set(cleaned["id"].values) == {101, 102, 104}

    def test_date_parsed_correctly(self, mock_d1_df):
        """date định dạng 'YYYYMMDDT...' được parse thành sale_year và sale_month."""
        cleaned = clean_d1(mock_d1_df)

        # ID 101: 20141013T000000 -> 2014, 10
        row_101 = cleaned[cleaned["id"] == 101].iloc[0]
        assert row_101["sale_year"] == 2014
        assert row_101["sale_month"] == 10

        # ID 102: 20150215T000000 -> 2015, 2
        row_102 = cleaned[cleaned["id"] == 102].iloc[0]
        assert row_102["sale_year"] == 2015
        assert row_102["sale_month"] == 2

    def test_house_age_calculation(self, mock_d1_df):
        """house_age = sale_year - yr_built."""
        cleaned = clean_d1(mock_d1_df)

        # ID 101: 2014 - 1955 = 59
        row_101 = cleaned[cleaned["id"] == 101].iloc[0]
        assert row_101["house_age"] == 59

        # ID 104: 2015 - 2010 = 5
        row_104 = cleaned[cleaned["id"] == 104].iloc[0]
        assert row_104["house_age"] == 5

    def test_renovated_flag_and_dropped_columns(self, mock_d1_df):
        """renovated = 1 khi yr_renovated > 0, = 0 khi == 0; bỏ yr_renovated và date."""
        cleaned = clean_d1(mock_d1_df)

        # ID 101: yr_renovated = 0 -> renovated = 0
        assert cleaned[cleaned["id"] == 101].iloc[0]["renovated"] == 0
        # ID 102: yr_renovated = 2005 -> renovated = 1
        assert cleaned[cleaned["id"] == 102].iloc[0]["renovated"] == 1

        assert "date" not in cleaned.columns
        assert "yr_renovated" not in cleaned.columns

    def test_id_column_preserved(self, mock_d1_df):
        """Cột 'id' VẪN CÒN sau clean_d1 (dùng cho GroupShuffleSplit)."""
        cleaned = clean_d1(mock_d1_df)
        assert "id" in cleaned.columns

    def test_price_not_modified(self, mock_d1_df):
        """price không bị thay đổi (không log1p, không scale)."""
        cleaned = clean_d1(mock_d1_df)

        # Kiểm tra giá trị ban đầu của ID 101
        assert cleaned[cleaned["id"] == 101].iloc[0]["price"] == 221900.0

    def test_input_dataframe_immutability(self, mock_d1_df):
        """Hàm clean_d1 không làm thay đổi DataFrame đầu vào."""
        df_copy_before = mock_d1_df.copy(deep=True)
        _ = clean_d1(mock_d1_df)
        pd.testing.assert_frame_equal(mock_d1_df, df_copy_before)

    def test_edge_case_empty_dataframe(self):
        """Trường hợp biên: DataFrame rỗng nhưng đủ cột."""
        empty_df = pd.DataFrame(
            columns=[
                "id",
                "date",
                "price",
                "bedrooms",
                "bathrooms",
                "sqft_living",
                "yr_built",
                "yr_renovated",
            ]
        )
        cleaned = clean_d1(empty_df)
        assert len(cleaned) == 0
        assert "house_age" in cleaned.columns
        assert "renovated" in cleaned.columns
        assert "date" not in cleaned.columns

    def test_edge_case_missing_required_column(self):
        """Trường hợp biên: DataFrame thiếu cột bắt buộc -> Báo lỗi KeyError rõ ràng."""
        invalid_df = pd.DataFrame({"id": [1], "price": [100000]})  # Thiếu 'bedrooms', 'date', etc.

        with pytest.raises(KeyError) as exc_info:
            clean_d1(invalid_df)

        # Kiểm tra lỗi có chỉ rõ cột bị thiếu không
        assert "bedrooms" in str(exc_info.value) or "date" in str(exc_info.value)

    def test_edge_case_invalid_date_format(self):
        """Trường hợp biên: date không đúng định dạng %Y%m%dT%H%M%S -> Báo lỗi ValueError."""
        wrong_date_df = pd.DataFrame(
            {
                "id": [1],
                "date": ["2014-10-13"],  # Sai định dạng ISO không có 'T000000'
                "price": [100000],
                "bedrooms": [3],
                "yr_built": [2000],
                "yr_renovated": [0],
            }
        )
        with pytest.raises(ValueError) as exc_info:
            clean_d1(wrong_date_df)
        assert "does not match format" in str(exc_info.value) or "doesn't match format" in str(exc_info.value)


# ==============================================================================
# 2. KIỂM THỬ HÀM clean_d2 (Indian Urban Housing)
# ==============================================================================
class TestCleanD2Unit:
    """Kiểm thử đơn vị cho clean_d2 với mock DataFrame."""

    @pytest.fixture
    def mock_d2_df(self):
        """Tạo DataFrame giả lập có đầy đủ các trường hợp cần test cho clean_d2."""
        return pd.DataFrame(
            {
                "Property_ID": ["PROP-01", "PROP-02", "PROP-03", "PROP-04", "PROP-05"],
                "City": ["Mumbai", "Delhi", "Bangalore", "Pune", "Chennai"],
                "Super_Area_SqFt": [1200.0, 99.9, 100.0, 50.0, 1500.0],  # 99.9 và 50 bị loại, 100.0 giữ
                "Carpet_Area_SqFt": [1000.0, 80.0, 90.0, 40.0, 1200.0],
                "Floor_Number": [5, 2, 0, 1, 10],
                "Total_Floors": [10, 5, 4, 3, 20],
                "Price_INR_Lakhs": [150.0, 25.0, 80.0, 15.0, 220.0],
            }
        )

    def test_property_id_dropped(self, mock_d2_df):
        """Cột Property_ID bị loại bỏ."""
        cleaned = clean_d2(mock_d2_df)
        assert "Property_ID" not in cleaned.columns

    def test_super_area_filtering_and_boundary_value_100(self, mock_d2_df):
        """Dòng < 100 bị loại; dòng ĐÚNG BẰNG 100.0 ĐƯỢC GIỮ."""
        cleaned = clean_d2(mock_d2_df)

        # 5 dòng ban đầu: 99.9 (PROP-02) và 50.0 (PROP-04) bị loại -> còn 3 dòng
        assert len(cleaned) == 3

        # Căn hộ có diện tích đúng bằng 100.0 (PROP-03) PHẢI CÒN
        assert 100.0 in cleaned["Super_Area_SqFt"].values
        # Các căn < 100 không còn
        assert (cleaned["Super_Area_SqFt"] < 100).sum() == 0

    def test_price_not_modified(self, mock_d2_df):
        """Price_INR_Lakhs không bị thay đổi."""
        cleaned = clean_d2(mock_d2_df)
        # Căn PROP-01 giữ nguyên giá 150.0
        assert cleaned["Price_INR_Lakhs"].iloc[0] == 150.0

    def test_input_dataframe_immutability(self, mock_d2_df):
        """Hàm clean_d2 không làm thay đổi DataFrame đầu vào."""
        df_copy_before = mock_d2_df.copy(deep=True)
        _ = clean_d2(mock_d2_df)
        pd.testing.assert_frame_equal(mock_d2_df, df_copy_before)

    def test_edge_case_empty_dataframe(self):
        """Trường hợp biên: DataFrame rỗng nhưng đủ cột."""
        empty_df = pd.DataFrame(
            columns=[
                "Property_ID",
                "Super_Area_SqFt",
                "Carpet_Area_SqFt",
                "Floor_Number",
                "Total_Floors",
                "Price_INR_Lakhs",
            ]
        )
        cleaned = clean_d2(empty_df)
        assert len(cleaned) == 0
        assert "Property_ID" not in cleaned.columns

    def test_edge_case_missing_required_column(self):
        """Trường hợp biên: Thiếu cột diện tích Super_Area_SqFt -> Báo lỗi KeyError rõ ràng."""
        invalid_df = pd.DataFrame(
            {"City": ["Delhi"], "Price_INR_Lakhs": [100.0]}  # Thiếu 'Super_Area_SqFt'
        )

        with pytest.raises(KeyError) as exc_info:
            clean_d2(invalid_df)

        assert "Super_Area_SqFt" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
