"""Bộ kiểm thử độc lập cho cấu hình đặc trưng (config_d1.py, config_d2.py)
đối chiếu với dữ liệu đã xử lý (data/processed).

File: tests/test_config.py
Kiểm tra:
1. Mọi cột trong NUM, CAT, ORD, TARGET đều tồn tại trong file train tương ứng.
2. Không cột nào xuất hiện ở hai nhóm cùng lúc (vừa NUM vừa CAT, TARGET trùng feature,...).
3. Mọi cột của file processed đều thuộc NUM, CAT, ORD hoặc TARGET (không bỏ sót cột nào).
4. Cột trong NUM thực sự là kiểu số; cột trong CAT/ORD thực sự là kiểu phân loại / chuỗi / mã vùng.
5. Bộ 2: ORD có đủ mọi giá trị unique thật của Locality_Type và Furnishing_Status, không thừa, không thiếu, thứ tự tăng dần hợp lý.
6. zipcode ở Bộ 1: số lượng unique, kiểm tra độ phủ giữa train và test (có giá trị nào chỉ xuất hiện ở test?).
7. City, Property_Type ở Bộ 2: kiểm tra độ phủ giữa train và test (có giá trị nào chỉ xuất hiện ở test?).
"""

import os
import pandas as pd
import numpy as np
import pytest

from d1.config_d1 import (
    NUM as NUM_D1,
    CAT as CAT_D1,
    ORD as ORD_D1,
    TARGET as TARGET_D1,
)
from d2.config_d2 import (
    NUM as NUM_D2,
    CAT as CAT_D2,
    ORD as ORD_D2,
    TARGET as TARGET_D2,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D1_TRAIN_PATH = os.path.join(BASE_DIR, "data", "processed", "d1_train.csv")
D1_TEST_PATH = os.path.join(BASE_DIR, "data", "processed", "d1_test.csv")
D2_TRAIN_PATH = os.path.join(BASE_DIR, "data", "processed", "d2_train.csv")
D2_TEST_PATH = os.path.join(BASE_DIR, "data", "processed", "d2_test.csv")


@pytest.fixture(scope="module")
def d1_data():
    df_train = pd.read_csv(D1_TRAIN_PATH)
    df_test = pd.read_csv(D1_TEST_PATH)
    return {"train": df_train, "test": df_test}


@pytest.fixture(scope="module")
def d2_data():
    df_train = pd.read_csv(D2_TRAIN_PATH)
    df_test = pd.read_csv(D2_TEST_PATH)
    return {"train": df_train, "test": df_test}


class TestConfigFeaturePresenceAndCompleteness:
    """Kiểm tra sự tồn tại và tính đầy đủ của các biến định nghĩa trong config."""

    def test_d1_all_config_features_exist_in_train(self, d1_data):
        """Mọi cột trong NUM, CAT, ORD, TARGET đều phải tồn tại trong d1_train."""
        train_cols = set(d1_data["train"].columns)
        all_config_cols = [TARGET_D1] + list(CAT_D1) + list(NUM_D1) + list(ORD_D1.keys())

        for col in all_config_cols:
            assert col in train_cols, f"[NGHIÊM TRỌNG] Cột '{col}' có trong config_d1 nhưng thiếu trong d1_train.csv"

    def test_d2_all_config_features_exist_in_train(self, d2_data):
        """Mọi cột trong NUM, CAT, ORD, TARGET đều phải tồn tại trong d2_train."""
        train_cols = set(d2_data["train"].columns)
        all_config_cols = [TARGET_D2] + list(CAT_D2) + list(NUM_D2) + list(ORD_D2.keys())

        for col in all_config_cols:
            assert col in train_cols, f"[NGHIÊM TRỌNG] Cột '{col}' có trong config_d2 nhưng thiếu trong d2_train.csv"

    def test_d1_no_column_belongs_to_multiple_groups(self):
        """Không cột nào của Bộ 1 xuất hiện ở hai nhóm cùng lúc."""
        groups = {
            "TARGET": [TARGET_D1],
            "CAT": list(CAT_D1),
            "ORD": list(ORD_D1.keys()),
            "NUM": list(NUM_D1),
        }
        seen = {}
        duplicates = []
        for grp_name, col_list in groups.items():
            for col in col_list:
                if col in seen:
                    duplicates.append((col, seen[col], grp_name))
                else:
                    seen[col] = grp_name

        assert not duplicates, f"[NGHIÊM TRỌNG] Trùng lặp nhóm trong config_d1: {duplicates}"

    def test_d2_no_column_belongs_to_multiple_groups(self):
        """Không cột nào của Bộ 2 xuất hiện ở hai nhóm cùng lúc."""
        groups = {
            "TARGET": [TARGET_D2],
            "CAT": list(CAT_D2),
            "ORD": list(ORD_D2.keys()),
            "NUM": list(NUM_D2),
        }
        seen = {}
        duplicates = []
        for grp_name, col_list in groups.items():
            for col in col_list:
                if col in seen:
                    duplicates.append((col, seen[col], grp_name))
                else:
                    seen[col] = grp_name

        assert not duplicates, f"[NGHIÊM TRỌNG] Trùng lặp nhóm trong config_d2: {duplicates}"

    def test_d1_no_omitted_columns_in_processed_data(self, d1_data):
        """Mọi cột của file d1_train đều phải thuộc một trong 4 nhóm: NUM, CAT, ORD, TARGET."""
        actual_cols = set(d1_data["train"].columns)
        config_cols = set([TARGET_D1] + list(CAT_D1) + list(NUM_D1) + list(ORD_D1.keys()))

        omitted_cols = actual_cols - config_cols
        assert not omitted_cols, f"[TRUNG BÌNH] Có cột trong d1_train bị bỏ sót khỏi config_d1: {omitted_cols}"

    def test_d2_no_omitted_columns_in_processed_data(self, d2_data):
        """Mọi cột của file d2_train đều phải thuộc một trong 4 nhóm: NUM, CAT, ORD, TARGET."""
        actual_cols = set(d2_data["train"].columns)
        config_cols = set([TARGET_D2] + list(CAT_D2) + list(NUM_D2) + list(ORD_D2.keys()))

        omitted_cols = actual_cols - config_cols
        assert not omitted_cols, f"[TRUNG BÌNH] Có cột trong d2_train bị bỏ sót khỏi config_d2: {omitted_cols}"


class TestConfigDataTypes:
    """Kiểm tra tính tương thích giữa phân nhóm config và kiểu dữ liệu thực tế."""

    def test_d1_data_types_compliance(self, d1_data):
        """NUM phải là số, CAT phải là biến phân loại (mã zipcode), TARGET phải là số."""
        train = d1_data["train"]

        # NUM
        for col in NUM_D1:
            assert pd.api.types.is_numeric_dtype(train[col]), (
                f"[NGHIÊM TRỌNG] Cột NUM '{col}' trong d1_train không phải là kiểu số (dtype={train[col].dtype})"
            )

        # TARGET
        assert pd.api.types.is_numeric_dtype(train[TARGET_D1]), (
            f"[NGHIÊM TRỌNG] Cột TARGET '{TARGET_D1}' trong d1_train không phải là kiểu số"
        )

        # CAT (zipcode - là mã vùng phân loại dạng số nguyên)
        for col in CAT_D1:
            # zipcode có thể lưu dạng int64 hoặc object/string, nhưng đại diện cho discrete category
            assert col in train.columns, f"Cột CAT '{col}' không tồn tại trong d1_train"

    def test_d2_data_types_compliance(self, d2_data):
        """NUM phải là số, CAT/ORD phải là dạng phân loại (object/string), TARGET là số."""
        train = d2_data["train"]

        # NUM
        for col in NUM_D2:
            assert pd.api.types.is_numeric_dtype(train[col]), (
                f"[NGHIÊM TRỌNG] Cột NUM '{col}' trong d2_train không phải là kiểu số (dtype={train[col].dtype})"
            )

        # TARGET
        assert pd.api.types.is_numeric_dtype(train[TARGET_D2]), (
            f"[NGHIÊM TRỌNG] Cột TARGET '{TARGET_D2}' trong d2_train không phải là kiểu số"
        )

        # CAT
        for col in CAT_D2:
            assert train[col].dtype == object or isinstance(train[col].dtype, pd.CategoricalDtype), (
                f"[TRUNG BÌNH] Cột CAT '{col}' trong d2_train không phải kiểu phân loại chuỗi (dtype={train[col].dtype})"
            )

        # ORD
        for col in ORD_D2.keys():
            assert train[col].dtype == object or isinstance(train[col].dtype, pd.CategoricalDtype), (
                f"[TRUNG BÌNH] Cột ORD '{col}' trong d2_train không phải kiểu phân loại chuỗi (dtype={train[col].dtype})"
            )


class TestD2OrdinalValuesAndOrdering:
    """Kiểm tra tính chính xác và đầy đủ của biến thứ bậc ORD trong Bộ 2."""

    def test_d2_locality_type_ordinal_values_and_order(self, d2_data):
        """ORD Locality_Type có đủ giá trị unique, không thừa, không thiếu và thứ tự hợp lý."""
        actual_train_values = set(d2_data["train"]["Locality_Type"].unique())
        actual_test_values = set(d2_data["test"]["Locality_Type"].unique())
        config_values = ORD_D2["Locality_Type"]

        # Kiểm tra không thiếu, không thừa
        assert set(config_values) == actual_train_values, (
            f"[NGHIÊM TRỌNG] Locality_Type trong config_d2 ({config_values}) "
            f"không khớp với dữ liệu train ({actual_train_values})"
        )
        assert set(config_values) == actual_test_values, (
            f"[NGHIÊM TRỌNG] Locality_Type trong test có giá trị lạ so với config: "
            f"{actual_test_values - set(config_values)}"
        )

        # Kiểm tra thứ tự logic: Affordable < Mid-Range < Premium
        expected_order = ["Affordable", "Mid-Range", "Premium"]
        assert config_values == expected_order, (
            f"[TRUNG BÌNH] Thứ tự Locality_Type không đúng logic tăng dần giá trị: {config_values} != {expected_order}"
        )

    def test_d2_furnishing_status_ordinal_values_and_order(self, d2_data):
        """ORD Furnishing_Status có đủ giá trị unique, không thừa, không thiếu và thứ tự hợp lý."""
        actual_train_values = set(d2_data["train"]["Furnishing_Status"].unique())
        actual_test_values = set(d2_data["test"]["Furnishing_Status"].unique())
        config_values = ORD_D2["Furnishing_Status"]

        # Kiểm tra không thiếu, không thừa
        assert set(config_values) == actual_train_values, (
            f"[NGHIÊM TRỌNG] Furnishing_Status trong config_d2 ({config_values}) "
            f"không khớp với dữ liệu train ({actual_train_values})"
        )
        assert set(config_values) == actual_test_values, (
            f"[NGHIÊM TRỌNG] Furnishing_Status trong test có giá trị lạ so với config: "
            f"{actual_test_values - set(config_values)}"
        )

        # Kiểm tra thứ tự logic: Unfurnished < Semi-Furnished < Furnished
        expected_order = ["Unfurnished", "Semi-Furnished", "Furnished"]
        assert config_values == expected_order, (
            f"[TRUNG BÌNH] Thứ tự Furnishing_Status không đúng logic tăng dần tiện nghi: {config_values} != {expected_order}"
        )


class TestCategoricalCoverageAndOutliers:
    """Kiểm tra độ phủ danh mục giữa tập Train và tập Test (Unseen category leakage / missingness)."""

    def test_d1_zipcode_unique_count_and_coverage(self, d1_data):
        """Kiểm tra số lượng unique zipcode và xác nhận không có zipcode nào chỉ xuất hiện ở test."""
        train_zips = set(d1_data["train"]["zipcode"].unique())
        test_zips = set(d1_data["test"]["zipcode"].unique())
        all_zips = train_zips.union(test_zips)

        unseen_in_train = test_zips - train_zips
        unseen_in_test = train_zips - test_zips

        print(f"\n[D1 Zipcode Coverage Analysis]")
        print(f"  + Tổng số mã zipcode unique: {len(all_zips)}")
        print(f"  + Số zipcode có trong train  : {len(train_zips)}")
        print(f"  + Số zipcode có trong test   : {len(test_zips)}")
        print(f"  + Zipcode chỉ có trong test (unseen in train): {len(unseen_in_train)} {unseen_in_train}")
        print(f"  + Zipcode chỉ có trong train (không có ở test): {len(unseen_in_test)} {unseen_in_test}")

        # Khẳng định 70 mã zipcode và không có mã lạ ở test
        assert len(all_zips) == 70, f"[TRUNG BÌNH] Tổng số zipcode kỳ vọng 70, thực tế: {len(all_zips)}"
        assert len(train_zips) == 70, f"[NGHIÊM TRỌNG] Tập train thiếu mã zipcode ({len(train_zips)}/70)"
        assert len(unseen_in_train) == 0, (
            f"[NGHIÊM TRỌNG] Có {len(unseen_in_train)} mã zipcode chỉ xuất hiện ở test mà không có ở train: {unseen_in_train}"
        )

    def test_d2_city_and_property_type_coverage(self, d2_data):
        """Kiểm tra các biến City, Property_Type ở Bộ 2: không được có giá trị nào chỉ xuất hiện ở test."""
        # City
        train_cities = set(d2_data["train"]["City"].unique())
        test_cities = set(d2_data["test"]["City"].unique())
        unseen_cities = test_cities - train_cities

        # Property_Type
        train_prop = set(d2_data["train"]["Property_Type"].unique())
        test_prop = set(d2_data["test"]["Property_Type"].unique())
        unseen_prop = test_prop - train_prop

        print(f"\n[D2 Categorical Coverage Analysis]")
        print(f"  + City unique train: {len(train_cities)}, test: {len(test_cities)}")
        print(f"  + City unseen in train: {len(unseen_cities)} {unseen_cities}")
        print(f"  + Property_Type unique train: {len(train_prop)}, test: {len(test_prop)}")
        print(f"  + Property_Type unseen in train: {len(unseen_prop)} {unseen_prop}")

        assert len(unseen_cities) == 0, (
            f"[NGHIÊM TRỌNG] Có thành phố chỉ xuất hiện ở test mà không có ở train: {unseen_cities}"
        )
        assert len(unseen_prop) == 0, (
            f"[NGHIÊM TRỌNG] Có loại hình BĐS chỉ xuất hiện ở test mà không có ở train: {unseen_prop}"
        )

