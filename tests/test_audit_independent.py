"""Bộ kiểm thử độc lập cho đồ án Học máy (PTIT)
File: tests/test_audit_independent.py
Chạy bằng: pytest -v tests/test_audit_independent.py
"""

import os
import sys
import subprocess
import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Thêm BASE_DIR vào sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from d1.clean_d1 import clean_d1
from d2.clean_d2 import clean_d2
from d1 import config_d1
from d2 import config_d2

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")


# ==============================================================================
# 1. KIỂM THỬ DỮ LIỆU THÔ (RAW DATA INTEGRITY)
# ==============================================================================
class TestRawDataIntegrity:
    """Kiểm tra sự tồn tại và tính nguyên vẹn của dữ liệu gốc tại data/raw/."""

    def test_raw_files_exist(self):
        """Xác nhận 3 file dữ liệu gốc tồn tại đầy đủ."""
        assert os.path.exists(os.path.join(RAW_DIR, "kc_house_data.csv"))
        assert os.path.exists(os.path.join(RAW_DIR, "train.csv"))
        assert os.path.exists(os.path.join(RAW_DIR, "test.csv"))

    def test_raw_files_unmodified_dimensions(self):
        """Xác nhận dữ liệu thô không bị chỉnh sửa (giữ đúng kích thước gốc)."""
        df1 = pd.read_csv(os.path.join(RAW_DIR, "kc_house_data.csv"))
        df2_train = pd.read_csv(os.path.join(RAW_DIR, "train.csv"))
        df2_test = pd.read_csv(os.path.join(RAW_DIR, "test.csv"))

        assert df1.shape == (21613, 21), f"D1 raw sai shape: {df1.shape}"
        assert df2_train.shape == (80000, 18), f"D2 train raw sai shape: {df2_train.shape}"
        assert df2_test.shape == (20000, 17), f"D2 test raw sai shape: {df2_test.shape}"


# ==============================================================================
# 2. KIỂM THỬ HÀM clean_d1 (d1/clean_d1.py)
# ==============================================================================
class TestCleanD1Logic:
    """Kiểm thử chuyên sâu hàm clean_d1."""

    @pytest.fixture(autouse=True)
    def setup_data(self):
        self.raw_df = pd.read_csv(os.path.join(RAW_DIR, "kc_house_data.csv"))

    def test_clean_d1_immutability(self):
        """Quy tắc: Hàm không được làm thay đổi DataFrame đầu vào."""
        df_copy = self.raw_df.copy()
        _ = clean_d1(df_copy)
        pd.testing.assert_frame_equal(df_copy, self.raw_df)

    def test_clean_d1_drops_33_bedrooms_only(self):
        """Chỉ loại bỏ đúng 1 dòng ngoại lai lỗi gõ phím bedrooms == 33."""
        cleaned = clean_d1(self.raw_df)
        assert len(self.raw_df) - len(cleaned) == 1
        assert (cleaned["bedrooms"] == 33).sum() == 0
        assert 15870 not in cleaned.index

    def test_clean_d1_feature_engineering(self):
        """Kiểm tra các đặc trưng tạo mới: sale_year, sale_month, house_age, renovated."""
        cleaned = clean_d1(self.raw_df)
        assert "sale_year" in cleaned.columns
        assert "sale_month" in cleaned.columns
        assert "house_age" in cleaned.columns
        assert "renovated" in cleaned.columns

        # yr_renovated và date phải bị xóa
        assert "date" not in cleaned.columns
        assert "yr_renovated" not in cleaned.columns

        # renovated chỉ nhận 0 hoặc 1
        assert set(cleaned["renovated"].unique()).issubset({0, 1})

    def test_clean_d1_preserves_id_and_raw_target(self):
        """Hàm clean_d1 phải giữ lại cột 'id' (để chia nhóm) và giữ nguyên giá gốc."""
        cleaned = clean_d1(self.raw_df)
        assert "id" in cleaned.columns
        assert "price" in cleaned.columns
        assert cleaned["price"].min() == self.raw_df["price"].min()
        assert cleaned["price"].max() == self.raw_df["price"].max()

    def test_clean_d1_no_pipeline_leakage(self):
        """KHÔNG được scale, One-Hot, hay log biến mục tiêu."""
        cleaned = clean_d1(self.raw_df)
        # zipcode vẫn là số nguyên danh mục, chưa bị One-Hot
        assert cleaned["zipcode"].dtype == np.int64
        # target chưa bị log1p (giá vẫn hàng trăm nghìn USD)
        assert cleaned["price"].mean() > 100000


# ==============================================================================
# 3. KIỂM THỬ HÀM clean_d2 (d2/clean_d2.py)
# ==============================================================================
class TestCleanD2Logic:
    """Kiểm thử chuyên sâu hàm clean_d2."""

    @pytest.fixture(autouse=True)
    def setup_data(self):
        self.raw_df = pd.read_csv(os.path.join(RAW_DIR, "train.csv"))

    def test_clean_d2_immutability(self):
        """Quy tắc: Hàm không được làm thay đổi DataFrame đầu vào."""
        df_copy = self.raw_df.copy()
        _ = clean_d2(df_copy)
        pd.testing.assert_frame_equal(df_copy, self.raw_df)

    def test_clean_d2_drops_small_area_outliers(self):
        """Loại bỏ chính xác 33 dòng có Super_Area_SqFt < 100."""
        cleaned = clean_d2(self.raw_df)
        assert len(self.raw_df) - len(cleaned) == 33
        assert (cleaned["Super_Area_SqFt"] < 100).sum() == 0

    def test_clean_d2_drops_property_id(self):
        """Cột Property_ID phải bị loại bỏ."""
        cleaned = clean_d2(self.raw_df)
        assert "Property_ID" not in cleaned.columns

    def test_clean_d2_no_pipeline_leakage(self):
        """KHÔNG được One-Hot hay scale hay log1p."""
        cleaned = clean_d2(self.raw_df)
        # Các cột danh mục vẫn là kiểu object
        assert cleaned["City"].dtype == object
        assert cleaned["Locality_Type"].dtype == object
        assert cleaned["Property_Type"].dtype == object
        assert cleaned["Furnishing_Status"].dtype == object
        # Target giữ nguyên giá trị thực tế Lakhs
        assert cleaned["Price_INR_Lakhs"].mean() > 50


# ==============================================================================
# 4. KIỂM THỬ QUY TRÌNH CHIA DỮ LIỆU & FILE PROCESSED (prepare_data.py)
# ==============================================================================
class TestPrepareDataAndProcessedFiles:
    """Kiểm thử kết quả của prepare_data.py và các file trong data/processed/."""

    @pytest.fixture(autouse=True)
    def load_processed(self):
        self.d1_train = pd.read_csv(os.path.join(PROCESSED_DIR, "d1_train.csv"))
        self.d1_test = pd.read_csv(os.path.join(PROCESSED_DIR, "d1_test.csv"))
        self.d2_train = pd.read_csv(os.path.join(PROCESSED_DIR, "d2_train.csv"))
        self.d2_test = pd.read_csv(os.path.join(PROCESSED_DIR, "d2_test.csv"))

    def test_d1_shapes_and_split_ratio(self):
        """Bộ 1: Train 17,291; Test 4,321; Tỉ lệ test nằm trong [0.19, 0.21]."""
        assert self.d1_train.shape == (17291, 22)
        assert self.d1_test.shape == (4321, 22)
        ratio = len(self.d1_test) / (len(self.d1_train) + len(self.d1_test))
        assert 0.19 <= ratio <= 0.21

    def test_d2_shapes_and_split_ratio(self):
        """Bộ 2: Train 63,973; Test 15,994; Tỉ lệ test nằm trong [0.19, 0.21]."""
        assert self.d2_train.shape == (63973, 17)
        assert self.d2_test.shape == (15994, 17)
        ratio = len(self.d2_test) / (len(self.d2_train) + len(self.d2_test))
        assert 0.19 <= ratio <= 0.21

    def test_d1_group_split_zero_id_overlap(self):
        """Kiểm tra cốt lõi: 0 ID nào được phép xuất hiện ở cả Train và Test."""
        raw_df = pd.read_csv(os.path.join(RAW_DIR, "kc_house_data.csv"))
        df_clean = clean_d1(raw_df)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tr_idx, te_idx = next(gss.split(df_clean, groups=df_clean["id"]))
        tr_ids = set(df_clean.iloc[tr_idx]["id"])
        te_ids = set(df_clean.iloc[te_idx]["id"])
        assert len(tr_ids.intersection(te_ids)) == 0

    def test_no_missing_values(self):
        """Tất cả 4 file processed phải có 0% missing values."""
        assert self.d1_train.isnull().sum().sum() == 0
        assert self.d1_test.isnull().sum().sum() == 0
        assert self.d2_train.isnull().sum().sum() == 0
        assert self.d2_test.isnull().sum().sum() == 0

    def test_column_consistency_train_test(self):
        """Tập train và test trong cùng bộ phải giống hệt nhau về tên và thứ tự cột."""
        assert list(self.d1_train.columns) == list(self.d1_test.columns)
        assert list(self.d2_train.columns) == list(self.d2_test.columns)

    def test_no_id_columns_in_final_processed(self):
        """Cột ID ('id', 'Property_ID') không được phép có mặt trong train/test."""
        assert "id" not in self.d1_train.columns
        assert "id" not in self.d1_test.columns
        assert "Property_ID" not in self.d2_train.columns
        assert "Property_ID" not in self.d2_test.columns

    def test_target_integrity(self):
        """Biến mục tiêu phải tồn tại và luôn dương."""
        assert "price" in self.d1_train.columns
        assert (self.d1_train["price"] > 0).all()
        assert (self.d1_test["price"] > 0).all()

        assert "Price_INR_Lakhs" in self.d2_train.columns
        assert (self.d2_train["Price_INR_Lakhs"] > 0).all()
        assert (self.d2_test["Price_INR_Lakhs"] > 0).all()


# ==============================================================================
# 5. KIỂM THỬ ĐỒNG BỘ CẤU HÌNH (config_d1.py & config_d2.py)
# ==============================================================================
class TestConfigsAlignment:
    """Kiểm thử tính khớp nối giữa cấu hình và dữ liệu."""

    def test_config_d1_completeness(self):
        """Mọi cột trong d1_train phải nằm trong TARGET, CAT, NUM, ORD và không thừa thiếu."""
        d1_train = pd.read_csv(os.path.join(PROCESSED_DIR, "d1_train.csv"))
        cfg_cols = set([config_d1.TARGET] + config_d1.CAT + config_d1.NUM + list(config_d1.ORD.keys()))
        data_cols = set(d1_train.columns)
        assert cfg_cols == data_cols

    def test_config_d2_completeness(self):
        """Mọi cột trong d2_train phải nằm trong TARGET, CAT, NUM, ORD và không thừa thiếu."""
        d2_train = pd.read_csv(os.path.join(PROCESSED_DIR, "d2_train.csv"))
        cfg_cols = set([config_d2.TARGET] + config_d2.CAT + config_d2.NUM + list(config_d2.ORD.keys()))
        data_cols = set(d2_train.columns)
        assert cfg_cols == data_cols

    def test_config_d2_ordinal_categories_exact(self):
        """Thứ tự ordinal trong config_d2 phải chứa đầy đủ các nhãn thực tế."""
        d2_train = pd.read_csv(os.path.join(PROCESSED_DIR, "d2_train.csv"))
        for col, order in config_d2.ORD.items():
            unique_vals = set(d2_train[col].unique())
            assert set(order) == unique_vals, f"Thứ tự ordinal cột {col} không khớp giá trị thực tế"


# ==============================================================================
# 6. KIỂM THỬ CHẠY THẬT SCRIPT check_data.py
# ==============================================================================
class TestCheckDataExecution:
    """Chạy trực tiếp script check_data.py và kiểm tra mã thoát."""

    def test_check_data_script_exit_code_zero(self):
        """Chạy script check_data.py, mã thoát phải bằng 0."""
        result = subprocess.run(
            [sys.executable, os.path.join(BASE_DIR, "check_data.py")],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        assert result.returncode == 0, f"check_data.py thất bại:\n{result.stderr}"
        assert "[PASS]" in result.stdout
        assert "[FAIL]" not in result.stdout

