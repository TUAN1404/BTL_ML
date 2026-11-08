"""Bộ kiểm thử tự động toàn diện (Automated Test Suite)
File: tests/test_data_pipeline.py

Kiểm thử:
1. Tính toàn vẹn cấu trúc & logic dữ liệu Bộ 1 (King County)
2. Tính toàn vẹn cấu trúc & logic dữ liệu Bộ 2 (Indian Urban Housing)
3. Tính nhất quán của file cấu hình d1/config_d1.py và d2/config_d2.py
4. Thử nghiệm tích hợp (Integration Test): Chạy thử một Scikit-Learn Pipeline
   thực tế (ColumnTransformer + LinearRegression/Ridge) trên cả 2 dataset
   để chứng minh dữ liệu sẵn sàng 100% cho mô hình học máy.
"""

import os
import unittest
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

# Đường dẫn thư mục
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

# Import cấu hình
import sys
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from d1 import config_d1
from d2 import config_d2


class TestDataset1KingCounty(unittest.TestCase):
    """Kiểm thử tính đúng đắn của dữ liệu Bộ 1: King County House Sales."""

    @classmethod
    def setUpClass(cls):
        cls.train_path = os.path.join(PROCESSED_DIR, "d1_train.csv")
        cls.test_path = os.path.join(PROCESSED_DIR, "d1_test.csv")
        cls.train_df = pd.read_csv(cls.train_path)
        cls.test_df = pd.read_csv(cls.test_path)

    def test_01_files_exist(self):
        """Kiểm tra các file processed của Bộ 1 tồn tại."""
        self.assertTrue(os.path.exists(self.train_path), "Thiếu file d1_train.csv")
        self.assertTrue(os.path.exists(self.test_path), "Thiếu file d1_test.csv")

    def test_02_shapes_and_split_ratio(self):
        """Kiểm tra kích thước và tỉ lệ chia 80/20."""
        self.assertEqual(self.train_df.shape[0], 17291)
        self.assertEqual(self.test_df.shape[0], 4321)
        self.assertEqual(self.train_df.shape[1], 22)
        self.assertEqual(self.test_df.shape[1], 22)

        total = len(self.train_df) + len(self.test_df)
        ratio = len(self.test_df) / total
        self.assertAlmostEqual(ratio, 0.2, delta=0.01, msg="Tỉ lệ test lệch quá 1%")

    def test_03_no_missing_values(self):
        """Xác nhận không có giá trị khuyết thiếu (0% missing)."""
        self.assertEqual(self.train_df.isnull().sum().sum(), 0)
        self.assertEqual(self.test_df.isnull().sum().sum(), 0)

    def test_04_id_removed(self):
        """Xác nhận cột 'id' đã bị loại bỏ khỏi cả 2 file."""
        self.assertNotIn("id", self.train_df.columns)
        self.assertNotIn("id", self.test_df.columns)

    def test_05_outlier_33_bedrooms_removed(self):
        """Xác nhận không còn dòng lỗi 33 phòng ngủ."""
        self.assertEqual((self.train_df["bedrooms"] == 33).sum(), 0)
        self.assertEqual((self.test_df["bedrooms"] == 33).sum(), 0)

    def test_06_target_validity(self):
        """Biến mục tiêu 'price' phải luôn dương và không có inf/nan."""
        self.assertIn(config_d1.TARGET, self.train_df.columns)
        self.assertIn(config_d1.TARGET, self.test_df.columns)
        self.assertTrue((self.train_df[config_d1.TARGET] > 0).all())
        self.assertTrue((self.test_df[config_d1.TARGET] > 0).all())

    def test_07_config_consistency(self):
        """Cấu hình config_d1.py phải khớp 100% với các cột trong file."""
        expected_cols = set([config_d1.TARGET] + config_d1.CAT + config_d1.NUM + list(config_d1.ORD.keys()))
        actual_cols = set(self.train_df.columns)
        self.assertEqual(expected_cols, actual_cols, "Config d1 không khớp với cột dữ liệu!")


class TestDataset2IndianUrban(unittest.TestCase):
    """Kiểm thử tính đúng đắn của dữ liệu Bộ 2: Indian Urban Housing."""

    @classmethod
    def setUpClass(cls):
        cls.train_path = os.path.join(PROCESSED_DIR, "d2_train.csv")
        cls.test_path = os.path.join(PROCESSED_DIR, "d2_test.csv")
        cls.train_df = pd.read_csv(cls.train_path)
        cls.test_df = pd.read_csv(cls.test_path)

    def test_01_files_exist(self):
        """Kiểm tra các file processed của Bộ 2 tồn tại."""
        self.assertTrue(os.path.exists(self.train_path), "Thiếu file d2_train.csv")
        self.assertTrue(os.path.exists(self.test_path), "Thiếu file d2_test.csv")

    def test_02_shapes_and_split_ratio(self):
        """Kiểm tra kích thước và tỉ lệ chia 80/20."""
        self.assertEqual(self.train_df.shape[0], 63973)
        self.assertEqual(self.test_df.shape[0], 15994)
        self.assertEqual(self.train_df.shape[1], 17)
        self.assertEqual(self.test_df.shape[1], 17)

        total = len(self.train_df) + len(self.test_df)
        ratio = len(self.test_df) / total
        self.assertAlmostEqual(ratio, 0.2, delta=0.01, msg="Tỉ lệ test lệch quá 1%")

    def test_03_no_missing_values(self):
        """Xác nhận không có giá trị khuyết thiếu (0% missing)."""
        self.assertEqual(self.train_df.isnull().sum().sum(), 0)
        self.assertEqual(self.test_df.isnull().sum().sum(), 0)

    def test_04_property_id_removed(self):
        """Xác nhận cột 'Property_ID' đã bị loại bỏ khỏi cả 2 file."""
        self.assertNotIn("Property_ID", self.train_df.columns)
        self.assertNotIn("Property_ID", self.test_df.columns)

    def test_05_small_area_outliers_removed(self):
        """Xác nhận toàn bộ căn hộ có diện tích < 100 sqft đã bị loại bỏ."""
        self.assertTrue((self.train_df["Super_Area_SqFt"] >= 100).all())
        self.assertTrue((self.test_df["Super_Area_SqFt"] >= 100).all())

    def test_06_logical_consistency(self):
        """Kiểm tra tính nhất quán logic: số tầng và diện tích lọt lòng."""
        self.assertTrue((self.train_df["Floor_Number"] <= self.train_df["Total_Floors"]).all())
        self.assertTrue((self.train_df["Carpet_Area_SqFt"] <= self.train_df["Super_Area_SqFt"]).all())

    def test_07_target_validity(self):
        """Biến mục tiêu 'Price_INR_Lakhs' phải luôn dương."""
        self.assertIn(config_d2.TARGET, self.train_df.columns)
        self.assertIn(config_d2.TARGET, self.test_df.columns)
        self.assertTrue((self.train_df[config_d2.TARGET] > 0).all())
        self.assertTrue((self.test_df[config_d2.TARGET] > 0).all())

    def test_08_config_consistency(self):
        """Cấu hình config_d2.py phải khớp 100% với các cột trong file."""
        expected_cols = set([config_d2.TARGET] + config_d2.CAT + config_d2.NUM + list(config_d2.ORD.keys()))
        actual_cols = set(self.train_df.columns)
        self.assertEqual(expected_cols, actual_cols, "Config d2 không khớp với cột dữ liệu!")


class TestBaselinePipelineIntegration(unittest.TestCase):
    """Kiểm thử tích hợp Pipeline thực tế: Đưa dữ liệu processed vào huấn luyện mô hình."""

    def test_01_pipeline_dataset_1(self):
        """Huấn luyện thử nghiệm mô hình Ridge Regression trên Bộ 1 (King County)."""
        train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "d1_train.csv"))
        test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "d1_test.csv"))

        X_train = train_df.drop(columns=[config_d1.TARGET])
        y_train = np.log1p(train_df[config_d1.TARGET])
        X_test = test_df.drop(columns=[config_d1.TARGET])
        y_test = np.log1p(test_df[config_d1.TARGET])

        # Pipeline chuẩn
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), config_d1.NUM),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), config_d1.CAT),
            ]
        )
        model = Pipeline([("pre", preprocessor), ("reg", Ridge(alpha=1.0))])

        # Fit trên train, predict trên test
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        r2 = r2_score(y_test, preds)
        # R2 của mô hình tuyến tính trên King County kỳ vọng >= 0.70
        self.assertGreater(r2, 0.70, f"R2 đạt {r2:.4f}, thấp hơn kỳ vọng baseline 0.70")
        print(f"\n  [V] Pipeline Test Bộ 1 (King County Ridge): R² = {r2:.4f} (ĐẠT)")

    def test_02_pipeline_dataset_2(self):
        """Huấn luyện thử nghiệm mô hình Ridge Regression trên Bộ 2 (Indian Urban)."""
        train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "d2_train.csv"))
        test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "d2_test.csv"))

        X_train = train_df.drop(columns=[config_d2.TARGET])
        y_train = np.log1p(train_df[config_d2.TARGET])
        X_test = test_df.drop(columns=[config_d2.TARGET])
        y_test = np.log1p(test_df[config_d2.TARGET])

        ord_categories = [config_d2.ORD[col] for col in config_d2.ORD.keys()]

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), config_d2.NUM),
                ("ord", OrdinalEncoder(categories=ord_categories), list(config_d2.ORD.keys())),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), config_d2.CAT),
            ]
        )
        model = Pipeline([("pre", preprocessor), ("reg", Ridge(alpha=10.0))])

        # Fit trên train, predict trên test
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        r2 = r2_score(y_test, preds)
        # R2 của mô hình tuyến tính trên Indian Urban kỳ vọng >= 0.70
        self.assertGreater(r2, 0.70, f"R2 đạt {r2:.4f}, thấp hơn kỳ vọng baseline 0.70")
        print(f"  [V] Pipeline Test Bộ 2 (Indian Urban Ridge): R² = {r2:.4f} (ĐẠT)")


if __name__ == "__main__":
    unittest.main(verbosity=2)

