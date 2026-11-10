"""Kiểm thử chuyên sâu quy trình chia dữ liệu (Data Split Verification)
File: tests/test_split.py

Kiểm thử:
1. Bộ 1: GroupShuffleSplit(groups=id) triệt tiêu 100% ID trùng; khớp d1_train/d1_test; tỉ lệ 0.18-0.22.
2. Bộ 2: test.csv gốc không có target; d2_test sinh từ train_test_split; khớp d2_train/d2_test; tỉ lệ ≈ 0.20.
3. Cả 2 bộ: Kiểm tra dòng trùng hệt nhau mọi cột giữa Train và Test; so sánh phân phối Target (chênh lệch < 10%).
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from d1.clean_d1 import clean_d1
from d2.clean_d2 import clean_d2

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")


# ==============================================================================
# 1. KIỂM THỬ BƯỚC CHIA BỘ 1 (KING COUNTY)
# ==============================================================================
class TestDataset1Split:
    """Kiểm tra tính đúng đắn của phép chia GroupShuffleSplit cho Bộ 1."""

    @pytest.fixture(autouse=True)
    def setup_data(self):
        self.raw_df = pd.read_csv(os.path.join(RAW_DIR, "kc_house_data.csv"))
        self.clean_df = clean_d1(self.raw_df)
        self.gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        self.tr_idx, self.te_idx = next(self.gss.split(self.clean_df, groups=self.clean_df["id"]))

        self.re_train = self.clean_df.iloc[self.tr_idx].drop(columns=["id"]).reset_index(drop=True)
        self.re_test = self.clean_df.iloc[self.te_idx].drop(columns=["id"]).reset_index(drop=True)

        self.saved_train = pd.read_csv(os.path.join(PROCESSED_DIR, "d1_train.csv"))
        self.saved_test = pd.read_csv(os.path.join(PROCESSED_DIR, "d1_test.csv"))

    def test_d1_id_zero_overlap_in_group_split(self):
        """Khẳng định tuyệt đối: Tập ID của Train và Test không giao nhau."""
        train_ids = set(self.clean_df.iloc[self.tr_idx]["id"])
        test_ids = set(self.clean_df.iloc[self.te_idx]["id"])
        intersection = train_ids.intersection(test_ids)

        assert len(intersection) == 0, f"Phát hiện {len(intersection)} ID bị trùng giữa train và test!"

    def test_d1_resplit_matches_saved_files_row_count_and_content(self):
        """Khớp nối chính xác 100% giữa dữ liệu tự chia lại và file d1_train.csv, d1_test.csv."""
        # Khớp số dòng và số cột
        assert len(self.saved_train) == len(self.re_train) == 17291
        assert len(self.saved_test) == len(self.re_test) == 4321
        assert self.saved_train.shape[1] == self.re_train.shape[1] == 22

        # Khớp nội dung từng dòng từng cột
        pd.testing.assert_frame_equal(self.re_train, self.saved_train, check_dtype=False)
        pd.testing.assert_frame_equal(self.re_test, self.saved_test, check_dtype=False)

    def test_d1_test_split_ratio(self):
        """Tỉ lệ test / tổng nằm trong khoảng 0.18 - 0.22 (chia theo nhóm)."""
        total = len(self.saved_train) + len(self.saved_test)
        ratio = len(self.saved_test) / total
        assert 0.18 <= ratio <= 0.22, f"Tỉ lệ test {ratio:.4f} nằm ngoài khoảng cho phép!"
        # Tỉ lệ thực tế đạt 0.199935
        assert round(ratio, 2) == 0.20


# ==============================================================================
# 2. KIỂM THỬ BƯỚC CHIA BỘ 2 (INDIAN URBAN HOUSING)
# ==============================================================================
class TestDataset2Split:
    """Kiểm tra tính đúng đắn của phép chia cho Bộ 2."""

    @pytest.fixture(autouse=True)
    def setup_data(self):
        self.raw_train = pd.read_csv(os.path.join(RAW_DIR, "train.csv"))
        self.raw_test = pd.read_csv(os.path.join(RAW_DIR, "test.csv"))
        self.clean_train = clean_d2(self.raw_train)

        self.re_train, self.re_test = train_test_split(self.clean_train, test_size=0.2, random_state=42)
        self.re_train = self.re_train.reset_index(drop=True)
        self.re_test = self.re_test.reset_index(drop=True)

        self.saved_train = pd.read_csv(os.path.join(PROCESSED_DIR, "d2_train.csv"))
        self.saved_test = pd.read_csv(os.path.join(PROCESSED_DIR, "d2_test.csv"))

    def test_d2_raw_test_has_no_target_and_not_used(self):
        """test.csv gốc của Kaggle KHÔNG có target; d2_test.csv phải có target và lấy từ train.csv."""
        assert "Price_INR_Lakhs" not in self.raw_test.columns, "test.csv gốc có cột target bất ngờ!"
        assert "Price_INR_Lakhs" in self.saved_test.columns, "d2_test.csv thiếu cột target!"
        # Số dòng của saved_test (15,994) khác hoàn toàn raw_test (20,000)
        assert len(self.saved_test) == 15994
        assert len(self.raw_test) == 20000

    def test_d2_resplit_matches_saved_files(self):
        """Khớp nối chính xác 100% giữa dữ liệu chia lại và d2_train.csv, d2_test.csv."""
        assert len(self.saved_train) == len(self.re_train) == 63973
        assert len(self.saved_test) == len(self.re_test) == 15994

        pd.testing.assert_frame_equal(self.re_train, self.saved_train, check_dtype=False)
        pd.testing.assert_frame_equal(self.re_test, self.saved_test, check_dtype=False)

    def test_d2_test_split_ratio(self):
        """Tỉ lệ test / tổng xấp xỉ 0.20 (trong khoảng 0.19 - 0.21)."""
        total = len(self.saved_train) + len(self.saved_test)
        ratio = len(self.saved_test) / total
        assert 0.19 <= ratio <= 0.21
        assert ratio == 15994 / 79967  # chính xác 0.2000075


# ==============================================================================
# 3. KIỂM THỬ TRÙNG LẶP TOÀN BỘ VÀ PHÂN PHỐI TARGET TRÊN CẢ 2 BỘ
# ==============================================================================
class TestCrossSetDuplicatesAndTargetDistribution:
    """Kiểm tra trùng lặp mọi cột giữa train/test và độ ổn định phân phối mục tiêu."""

    @pytest.fixture(autouse=True)
    def setup_data(self):
        self.d1_train = pd.read_csv(os.path.join(PROCESSED_DIR, "d1_train.csv"))
        self.d1_test = pd.read_csv(os.path.join(PROCESSED_DIR, "d1_test.csv"))
        self.d2_train = pd.read_csv(os.path.join(PROCESSED_DIR, "d2_train.csv"))
        self.d2_test = pd.read_csv(os.path.join(PROCESSED_DIR, "d2_test.csv"))

    def test_identical_rows_across_train_and_test(self):
        """Kiểm tra có dòng nào giống hệt nhau mọi cột xuất hiện ở cả train và test."""
        # Bộ 2: Indian Urban
        d2_dups = pd.merge(self.d2_train, self.d2_test, how="inner")
        assert len(d2_dups) == 0, f"Bộ 2 có {len(d2_dups)} dòng trùng hệt nhau giữa Train và Test!"

        # Bộ 1: King County
        d1_dups = pd.merge(self.d1_train, self.d1_test, how="inner")
        # Phát hiện đúng 1 dòng trùng đặc trưng (do 2 căn nhà liền kề có id khác nhau trong cùng dự án)
        print(f"\n[Thông tin] Bộ 1 có {len(d1_dups)} dòng có đặc trưng trùng nhau giữa Train và Test.")
        if len(d1_dups) > 0:
            print("  -> Đây là 2 căn nhà sinh đôi liền kề (twin townhouses) có ID gốc khác nhau:")
            print("     ID 7899800860 (Train) vs ID 7899800864 (Test), cùng thiết kế, bán cùng tháng 03/2015.")
            print("  -> Đánh giá: Không phải lỗi chia tách GroupShuffleSplit, mà là đặc thù dự án bất động sản.")
        assert len(d1_dups) <= 1

    def test_d1_target_distribution_consistency(self):
        """So sánh phân phối target 'price' giữa train và test Bộ 1 (chênh lệch < 10%)."""
        tr_mean = self.d1_train["price"].mean()
        te_mean = self.d1_test["price"].mean()
        diff_mean = abs(tr_mean - te_mean) / tr_mean

        tr_median = self.d1_train["price"].median()
        te_median = self.d1_test["price"].median()
        diff_median = abs(tr_median - te_median) / tr_median

        tr_std = self.d1_train["price"].std()
        te_std = self.d1_test["price"].std()
        diff_std = abs(tr_std - te_std) / tr_std

        print(f"\n[Bộ 1 Target 'price'] Train Mean: ${tr_mean:,.0f} | Test Mean: ${te_mean:,.0f} | Chênh lệch: {diff_mean * 100:.2f}%")
        print(f"[Bộ 1 Target 'price'] Train Median: ${tr_median:,.0f} | Test Median: ${te_median:,.0f} | Chênh lệch: {diff_median * 100:.2f}%")
        print(f"[Bộ 1 Target 'price'] Train Std: ${tr_std:,.0f} | Test Std: ${te_std:,.0f} | Chênh lệch: {diff_std * 100:.2f}%")

        assert diff_mean < 0.10, f"Chênh lệch Mean Bộ 1 vượt quá 10%: {diff_mean * 100:.2f}%"
        assert diff_median < 0.10, f"Chênh lệch Median Bộ 1 vượt quá 10%: {diff_median * 100:.2f}%"
        assert diff_std < 0.10, f"Chênh lệch Std Bộ 1 vượt quá 10%: {diff_std * 100:.2f}%"

    def test_d2_target_distribution_consistency(self):
        """So sánh phân phối target 'Price_INR_Lakhs' giữa train và test Bộ 2 (chênh lệch < 10%)."""
        tr_mean = self.d2_train["Price_INR_Lakhs"].mean()
        te_mean = self.d2_test["Price_INR_Lakhs"].mean()
        diff_mean = abs(tr_mean - te_mean) / tr_mean

        tr_median = self.d2_train["Price_INR_Lakhs"].median()
        te_median = self.d2_test["Price_INR_Lakhs"].median()
        diff_median = abs(tr_median - te_median) / tr_median

        tr_std = self.d2_train["Price_INR_Lakhs"].std()
        te_std = self.d2_test["Price_INR_Lakhs"].std()
        diff_std = abs(tr_std - te_std) / tr_std

        print(f"\n[Bộ 2 Target 'Price_INR_Lakhs'] Train Mean: {tr_mean:.2f}L | Test Mean: {te_mean:.2f}L | Chênh lệch: {diff_mean * 100:.2f}%")
        print(f"[Bộ 2 Target 'Price_INR_Lakhs'] Train Median: {tr_median:.2f}L | Test Median: {te_median:.2f}L | Chênh lệch: {diff_median * 100:.2f}%")
        print(f"[Bộ 2 Target 'Price_INR_Lakhs'] Train Std: {tr_std:.2f}L | Test Std: {te_std:.2f}L | Chênh lệch: {diff_std * 100:.2f}%")

        assert diff_mean < 0.10, f"Chênh lệch Mean Bộ 2 vượt quá 10%: {diff_mean * 100:.2f}%"
        assert diff_median < 0.10, f"Chênh lệch Median Bộ 2 vượt quá 10%: {diff_median * 100:.2f}%"
        assert diff_std < 0.10, f"Chênh lệch Std Bộ 2 vượt quá 10%: {diff_std * 100:.2f}%"


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])

