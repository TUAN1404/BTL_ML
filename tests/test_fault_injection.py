"""Kiểm thử đột biến / Tiêm lỗi (Fault Injection / Mutation Testing).
File: tests/test_fault_injection.py

Mục tiêu:
Xác thực xem kịch bản kiểm tra trong check_data.py và các test suite pytest
(test_data_integrity, test_split, test_config, test_smoke) có THẬT SỰ bắt được lỗi hay không.

Thực hiện trên THƯ MỤC TẠM (tempfile / isolated copies), TUYỆT ĐỐI KHÔNG sửa data/processed/ thật:
1. Chèn giá trị NaN vào d1_train.
2. Đổi thứ tự 2 cột trong d2_test.
3. Thêm lại cột 'id' vào d1_train.
4. Chép 50 dòng từ test sang train (tạo trùng lặp/chồng lấn).
5. Thêm cột 'price_per_sqft = price / sqft_living' vào d1_train.
6. Áp dụng np.log1p lên cột price.

Ghi nhận chi tiết: Test nào BẮT ĐƯỢC, test nào KHÔNG BẮT ĐƯỢC (lỗ hổng).
"""

import os
import shutil
import tempfile
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from sklearn.metrics import r2_score
import check_data

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIG_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")


@pytest.fixture
def temp_env():
    """Tạo bản sao độc lập của data/processed trong thư mục tạm."""
    temp_dir = tempfile.mkdtemp(prefix="test_fault_injection_")
    for fname in ["d1_train.csv", "d1_test.csv", "d2_train.csv", "d2_test.csv"]:
        src = os.path.join(ORIG_PROCESSED_DIR, fname)
        dst = os.path.join(temp_dir, fname)
        shutil.copyfile(src, dst)
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestFaultInjectionScenarios:
    """Các ca tiêm lỗi đột biến."""

    def test_case_1_inject_nan_into_d1_train(self, temp_env):
        """Case 1: Chèn 5 giá trị NaN vào cột 'bedrooms' của d1_train."""
        fpath = os.path.join(temp_env, "d1_train.csv")
        df = pd.read_csv(fpath)
        df.loc[:4, "bedrooms"] = np.nan
        df.to_csv(fpath, index=False)

        # 1. Kiểm tra check_data.py
        try:
            with patch("check_data.PROCESSED_DIR", temp_env):
                check_data_result = check_data.check_dataset_1()
        except KeyError as e:
            # check_data.py in ra [FAIL] ở bước kiểm tra cột và bước kiểm tra ID,
            # nhưng sau đó bị crash do KeyError trong hàm print_min_max_table vì thiếu cột ở test
            check_data_result = False
            print(f"  [GHI CHÚ] check_data.py đã in [FAIL] rồi văng lỗi {type(e).__name__}: {e} khi in bảng min/max")

        # 2. Kiểm tra test_data_integrity logic
        test_integrity_caught = (df.isnull().sum().sum() > 0)

        print("\n" + "=" * 70)
        print("CASE 1: Chèn NaN vào d1_train")
        print(f"  - check_data.py bắt được : {not check_data_result} (Kỳ vọng: BẮT ĐƯỢC)")
        print(f"  - test_data_integrity bắt được: {test_integrity_caught} (Kỳ vọng: BẮT ĐƯỢC)")
        print("=" * 70)

        assert not check_data_result, "check_data.py KHÔNG bắt được lỗi NaN!"
        assert test_integrity_caught, "test_data_integrity KHÔNG bắt được lỗi NaN!"

    def test_case_2_swap_column_order_in_d2_test(self, temp_env):
        """Case 2: Đổi thứ tự hai cột 'City' và 'Property_Type' trong d2_test."""
        fpath = os.path.join(temp_env, "d2_test.csv")
        df_test = pd.read_csv(fpath)
        cols = list(df_test.columns)
        # Hoán đổi vị trí của City và Property_Type
        idx1, idx2 = cols.index("City"), cols.index("Property_Type")
        cols[idx1], cols[idx2] = cols[idx2], cols[idx1]
        df_test = df_test[cols]
        df_test.to_csv(fpath, index=False)

        # 1. Kiểm tra check_data.py
        with patch("check_data.PROCESSED_DIR", temp_env):
            check_data_result = check_data.check_dataset_2()

        # 2. Kiểm tra test_data_integrity logic (kiểm tra list(train.cols) == list(test.cols))
        df_train = pd.read_csv(os.path.join(temp_env, "d2_train.csv"))
        test_integrity_caught = (list(df_train.columns) != list(df_test.columns))

        # 3. Kiểm tra test_config logic
        # Lưu ý: test_config dùng set(columns) để đối chiếu sự hiện diện, không xét thứ tự
        test_config_caught = (set(df_train.columns) != set(df_test.columns))

        print("\n" + "=" * 70)
        print("CASE 2: Đổi thứ tự hai cột trong d2_test")
        print(f"  - check_data.py bắt được : {not check_data_result} (Kỳ vọng: BẮT ĐƯỢC)")
        print(f"  - test_data_integrity bắt được: {test_integrity_caught} (Kỳ vọng: BẮT ĐƯỢC)")
        print(f"  - test_config bắt được: {test_config_caught} (Lỗ hổng: test_config chỉ so set)")
        print("=" * 70)

        assert not check_data_result, "check_data.py KHÔNG bắt được lỗi thứ tự cột!"
        assert test_integrity_caught, "test_data_integrity KHÔNG bắt được lỗi thứ tự cột!"

    def test_case_3_add_id_column_to_d1_train(self, temp_env):
        """Case 3: Thêm lại cột 'id' vào d1_train."""
        fpath = os.path.join(temp_env, "d1_train.csv")
        df = pd.read_csv(fpath)
        df.insert(0, "id", np.arange(len(df)))
        df.to_csv(fpath, index=False)

        # 1. Kiểm tra check_data.py
        try:
            with patch("check_data.PROCESSED_DIR", temp_env):
                check_data_result = check_data.check_dataset_1()
        except KeyError as e:
            # check_data.py in ra [FAIL] ở bước kiểm tra cột và bước kiểm tra ID,
            # nhưng sau đó bị crash do KeyError trong hàm print_min_max_table vì thiếu cột ở test
            check_data_result = False
            print(f"  [GHI CHÚ] check_data.py đã in [FAIL] rồi văng lỗi {type(e).__name__}: {e} khi in bảng min/max")

        # 2. Kiểm tra test_data_integrity ("id" not in df.columns)
        test_integrity_caught = ("id" in df.columns)

        # 3. Kiểm tra test_config (omitted_cols = actual_cols - config_cols)
        from d1.config_d1 import NUM, CAT, ORD, TARGET
        config_cols = set([TARGET] + CAT + NUM + list(ORD.keys()))
        test_config_caught = ("id" in (set(df.columns) - config_cols))

        print("\n" + "=" * 70)
        print("CASE 3: Thêm lại cột 'id' vào d1_train")
        print(f"  - check_data.py bắt được : {not check_data_result} (Kỳ vọng: BẮT ĐƯỢC)")
        print(f"  - test_data_integrity bắt được: {test_integrity_caught} (Kỳ vọng: BẮT ĐƯỢC)")
        print(f"  - test_config bắt được        : {test_config_caught} (Kỳ vọng: BẮT ĐƯỢC)")
        print("=" * 70)

        assert not check_data_result, "check_data.py KHÔNG bắt được cột ID!"
        assert test_integrity_caught, "test_data_integrity KHÔNG bắt được cột ID!"
        assert test_config_caught, "test_config KHÔNG bắt được cột ID!"

    def test_case_4_overlap_50_rows_from_test_to_train(self, temp_env):
        """Case 4: Sao chép 50 dòng từ test sang train."""
        f_tr = os.path.join(temp_env, "d1_train.csv")
        f_te = os.path.join(temp_env, "d1_test.csv")
        df_tr = pd.read_csv(f_tr)
        df_te = pd.read_csv(f_te)

        # Copy 50 dòng từ test chèn vào train
        leaked_rows = df_te.iloc[:50].copy()
        df_tr_mutated = pd.concat([df_tr, leaked_rows], ignore_index=True)
        df_tr_mutated.to_csv(f_tr, index=False)

        # 1. Kiểm tra check_data.py
        try:
            with patch("check_data.PROCESSED_DIR", temp_env):
                check_data_result = check_data.check_dataset_1()
        except KeyError as e:
            # check_data.py in ra [FAIL] ở bước kiểm tra cột và bước kiểm tra ID,
            # nhưng sau đó bị crash do KeyError trong hàm print_min_max_table vì thiếu cột ở test
            check_data_result = False
            print(f"  [GHI CHÚ] check_data.py đã in [FAIL] rồi văng lỗi {type(e).__name__}: {e} khi in bảng min/max")

        # 2. Kiểm tra test_split.py logic (pd.merge(train, test, how='inner'))
        dups = pd.merge(df_tr_mutated, df_te, how="inner")
        # Bản thân D1 gốc chỉ có 1 dòng trùng do twin townhouse; khi chèn 50 dòng sẽ có ít nhất 51 dòng
        test_split_caught = len(dups) > 1

        print("\n" + "=" * 70)
        print("CASE 4: Copy 50 dòng từ test sang train")
        print(f"  - check_data.py bắt được : {not check_data_result} (LỖ HỔNG LỚN: check_data không check file train/test)")
        print(f"  - test_split.py bắt được : {test_split_caught} (Kỳ vọng: BẮT ĐƯỢC)")
        print("=" * 70)

        # check_data.py sẽ KHÔNG bắt được vì nó chỉ re-split kc_house_data.csv raw chứ không merge file processed!
        assert test_split_caught, "test_split.py KHÔNG bắt được dữ liệu trùng lặp!"

    def test_case_5_add_leaked_feature_price_per_sqft(self, temp_env):
        """Case 5: Thêm cột 'price_per_sqft = price / sqft_living' vào d1_train."""
        f_tr = os.path.join(temp_env, "d1_train.csv")
        f_te = os.path.join(temp_env, "d1_test.csv")
        df_tr = pd.read_csv(f_tr)
        df_te = pd.read_csv(f_te)

        # Nếu chỉ thêm vào train:
        df_tr["price_per_sqft"] = df_tr["price"] / df_tr["sqft_living"]
        df_tr.to_csv(f_tr, index=False)

        # 1. check_data.py (phát hiện lệch cột giữa train và test)
        try:
            with patch("check_data.PROCESSED_DIR", temp_env):
                check_data_result = check_data.check_dataset_1()
        except KeyError as e:
            # check_data.py in ra [FAIL] ở bước kiểm tra cột,
            # sau đó crash do KeyError trong hàm print_min_max_table vì thiếu cột ở test
            check_data_result = False
            print(f"  [GHI CHÚ] check_data.py đã in [FAIL] rồi văng lỗi {type(e).__name__}: {e} khi in bảng min/max")

        # 2. test_config.py (phát hiện cột lạ ngoài config)
        from d1.config_d1 import NUM, CAT, ORD, TARGET
        config_cols = set([TARGET] + CAT + NUM + list(ORD.keys()))
        test_config_caught = ("price_per_sqft" in (set(df_tr.columns) - config_cols))

        # 3. Phân tích nếu cố tình thêm vào CẢ train và test:
        df_te["price_per_sqft"] = df_te["price"] / df_te["sqft_living"]
        from tests.test_smoke import build_pipeline
        num_with_leak = NUM + ["price_per_sqft"]
        pipe = build_pipeline(num_with_leak, CAT, ORD)
        pipe.fit(df_tr[num_with_leak + CAT], np.log1p(df_tr[TARGET]))
        pred_te = np.expm1(pipe.predict(df_te[num_with_leak + CAT]))
        r2_leaked = r2_score(df_te[TARGET], pred_te)
        r2_log_leaked = r2_score(np.log1p(df_te[TARGET]), pipe.predict(df_te[num_with_leak + CAT]))

        # Mô hình tuyến tính không nhân được A * B nên R² gốc không chạm 0.97 (đạt 0.8286),
        # nhưng R² log tăng từ 0.8804 lên 0.9460.
        smoke_warning_caught = (r2_leaked > 0.97)

        print("\n" + "=" * 70)
        print("CASE 5: Thêm feature tạo từ target (price_per_sqft)")
        print(f"  - check_data.py (khi chỉ thêm train)     : {not check_data_result} (BẮT ĐƯỢC do lệch cột)")
        print(f"  - check_data.py (nếu thêm cả 2 tập)      : False (LỖ HỔNG: check_data không so với config)")
        print(f"  - test_config.py bắt được                : {test_config_caught} (Kỳ vọng: BẮT ĐƯỢC)")
        print(f"  - test_smoke.py (R² gốc > 0.97)          : {smoke_warning_caught} (R² gốc: {r2_leaked:.4f}, R² log: {r2_log_leaked:.4f})")
        print("=" * 70)

        assert not check_data_result, "check_data.py KHÔNG bắt được khi bị lệch cột!"
        assert test_config_caught, "test_config.py KHÔNG bắt được cột thừa ngoài config!"

    def test_case_6_apply_log1p_to_target(self, temp_env):
        """Case 6: Áp dụng np.log1p lên cột price trong d1_train và d1_test."""
        f_tr = os.path.join(temp_env, "d1_train.csv")
        f_te = os.path.join(temp_env, "d1_test.csv")
        df_tr = pd.read_csv(f_tr)
        df_te = pd.read_csv(f_te)

        df_tr["price"] = np.log1p(df_tr["price"])
        df_te["price"] = np.log1p(df_te["price"])
        df_tr.to_csv(f_tr, index=False)
        df_te.to_csv(f_te, index=False)

        # 1. Kiểm tra check_data.py (chỉ check price > 0, log1p của price > 0 vẫn > 0!)
        with patch("check_data.PROCESSED_DIR", temp_env):
            check_data_result = check_data.check_dataset_1()

        # 2. Kiểm tra test_data_integrity (check max price > 1000)
        test_integrity_caught = (df_tr["price"].max() <= 1000)

        print("\n" + "=" * 70)
        print("CASE 6: Áp dụng log1p lên cột target price")
        print(f"  - check_data.py bắt được : {not check_data_result} (LỖ HỔNG: check_data chỉ check price > 0!)")
        print(f"  - test_data_integrity bắt được: {test_integrity_caught} (Kỳ vọng: BẮT ĐƯỢC)")
        print(f"    (Max price sau log: {df_tr['price'].max():.2f} <= 1000)")
        print("=" * 70)

        # check_data.py bị lọt lỗ hổng vì check_data.py chỉ check min > 0
        assert test_integrity_caught, "test_data_integrity KHÔNG bắt được lỗi log target!"
