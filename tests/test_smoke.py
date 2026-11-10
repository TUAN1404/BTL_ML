"""Kiểm thử khói (Smoke Test) độc lập cho quy trình Baseline Pipeline tối thiểu.
File: tests/test_smoke.py

Mục tiêu:
Xây dựng pipeline scikit-learn tối thiểu sử dụng ColumnTransformer + Ridge,
kiểm tra xem dữ liệu processed (d1_train/d1_test, d2_train/d2_test) và config tương ứng
có thể đưa vào huấn luyện mô hình thực tế trơn tru hay không.

Kiểm tra:
1. Pipeline fit và predict không phát sinh lỗi trên cả 2 bộ dữ liệu.
2. Target dùng np.log1p khi fit, np.expm1 khi dự đoán / đánh giá.
3. R² của Ridge lớn hơn DummyRegressor(strategy="median") rõ rệt.
4. Cảnh báo nếu R² test > 0.97 (nghi ngờ rò rỉ) hoặc R² test < 0.30 (nghi ngờ lỗi tiền xử lý).
5. Dự đoán không có NaN, inf và không có giá trị âm sau expm1.
6. Pipeline chống chịu tốt với nhãn phân loại chưa từng thấy (unseen categories) ở tập test mà không crash.
7. So sánh R² train và test trên thang log và thang đo gốc.
"""

import os
import warnings
import numpy as np
import pandas as pd
import pytest

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.metrics import r2_score, root_mean_squared_error

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


def build_pipeline(num_cols, cat_cols, ord_dict):
    """Xây dựng ColumnTransformer + Ridge chuẩn mực theo config."""
    transformers = [
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            num_cols,
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            cat_cols,
        ),
    ]

    if ord_dict:
        ord_cols = list(ord_dict.keys())
        ord_categories = [ord_dict[c] for c in ord_cols]
        transformers.append((
            "ord",
            OrdinalEncoder(
                categories=ord_categories,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
            ord_cols,
        ))

    preprocessor = ColumnTransformer(transformers=transformers)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", Ridge(alpha=1.0, random_state=42)),
    ])
    return pipeline


class TestSmokePipelineDataset1:
    """Smoke test trên Bộ 1: King County House Sales."""

    @pytest.fixture(autouse=True)
    def setup_data(self):
        self.df_train = pd.read_csv(D1_TRAIN_PATH)
        self.df_test = pd.read_csv(D1_TEST_PATH)

        self.features = NUM_D1 + CAT_D1
        self.X_train = self.df_train[self.features]
        self.y_train = self.df_train[TARGET_D1]
        self.X_test = self.df_test[self.features]
        self.y_test = self.df_test[TARGET_D1]

    def test_d1_pipeline_execution_and_metrics(self):
        pipeline = build_pipeline(NUM_D1, CAT_D1, ORD_D1)

        # 1. Fit trên log1p của target
        y_train_log = np.log1p(self.y_train)
        pipeline.fit(self.X_train, y_train_log)

        # 2. Predict và expm1
        y_train_pred_log = pipeline.predict(self.X_train)
        y_test_pred_log = pipeline.predict(self.X_test)

        y_train_pred = np.expm1(y_train_pred_log)
        y_test_pred = np.expm1(y_test_pred_log)

        # 3. Kiểm tra tính hợp lệ của dự đoán
        assert not np.isnan(y_test_pred).any(), "[NGHIÊM TRỌNG] Dự đoán Bộ 1 chứa NaN"
        assert not np.isinf(y_test_pred).any(), "[NGHIÊM TRỌNG] Dự đoán Bộ 1 chứa Inf"
        assert (y_test_pred >= 0).all(), "[NGHIÊM TRỌNG] Dự đoán Bộ 1 có giá trị âm sau expm1"

        # 4. Tính toán số liệu thực tế
        # Thang đo gốc
        r2_orig_train = r2_score(self.y_train, y_train_pred)
        r2_orig_test = r2_score(self.y_test, y_test_pred)
        # Thang log
        r2_log_train = r2_score(y_train_log, y_train_pred_log)
        r2_log_test = r2_score(np.log1p(self.y_test), y_test_pred_log)

        # RMSLE
        rmsle_test = root_mean_squared_error(
            np.log1p(self.y_test), np.log1p(np.clip(y_test_pred, 0, None))
        )

        # 5. So sánh với DummyRegressor (strategy="median")
        dummy = DummyRegressor(strategy="median")
        dummy.fit(self.X_train, self.y_train)
        y_dummy_test = dummy.predict(self.X_test)
        r2_dummy = r2_score(self.y_test, y_dummy_test)
        rmsle_dummy = root_mean_squared_error(np.log1p(self.y_test), np.log1p(y_dummy_test))

        print("\n" + "=" * 70)
        print("KẾT QUẢ THỰC NGHIỆM SMOKE TEST BỘ 1 (King County):")
        print(f"  + Ridge R² test (thang gốc USD) : {r2_orig_test:.4f}")
        print(f"  + Ridge R² test (thang log)     : {r2_log_test:.4f}")
        print(f"  + Ridge R² train (thang gốc USD): {r2_orig_train:.4f}")
        print(f"  + Ridge R² train (thang log)    : {r2_log_train:.4f}")
        print(f"  + Ridge RMSLE test              : {rmsle_test:.4f}")
        print(f"  + Dummy R² test (median)        : {r2_dummy:.4f}")
        print(f"  + Dummy RMSLE test (median)     : {rmsle_dummy:.4f}")
        print("=" * 70)

        # R² của Ridge phải vượt trội Dummy
        assert r2_orig_test > r2_dummy, (
            f"[NGHIÊM TRỌNG] Ridge R² ({r2_orig_test:.4f}) không thắng được Dummy ({r2_dummy:.4f})"
        )

        # Cảnh báo khoảng tin cậy R² test
        if r2_orig_test > 0.97:
            warnings.warn(f"[CẢNH BÁO] R² test quá cao ({r2_orig_test:.4f} > 0.97) - Nghi ngờ rò rỉ dữ liệu!")
        if r2_orig_test < 0.30:
            warnings.warn(f"[CẢNH BÁO] R² test quá thấp ({r2_orig_test:.4f} < 0.30) - Nghi ngờ lỗi tiền xử lý!")

        # Khoảng chấp nhận bình thường cho Ridge trên King County
        assert 0.30 <= r2_orig_test <= 0.97

    def test_d1_unseen_category_robustness(self):
        """Pipeline phải dự đoán bình thường khi gặp mã zipcode chưa từng xuất hiện ở train."""
        pipeline = build_pipeline(NUM_D1, CAT_D1, ORD_D1)
        pipeline.fit(self.X_train, np.log1p(self.y_train))

        unseen_sample = self.X_test.iloc[0:1].copy()
        unseen_sample["zipcode"] = 99999  # Mã zipcode hoàn toàn mới

        pred_log = pipeline.predict(unseen_sample)
        pred = np.expm1(pred_log)

        assert not np.isnan(pred[0]), "Dự đoán trả về NaN khi gặp unseen category"
        assert pred[0] > 0, f"Dự đoán không hợp lệ: {pred[0]}"


class TestSmokePipelineDataset2:
    """Smoke test trên Bộ 2: Indian Urban Housing."""

    @pytest.fixture(autouse=True)
    def setup_data(self):
        self.df_train = pd.read_csv(D2_TRAIN_PATH)
        self.df_test = pd.read_csv(D2_TEST_PATH)

        self.features = NUM_D2 + CAT_D2 + list(ORD_D2.keys())
        self.X_train = self.df_train[self.features]
        self.y_train = self.df_train[TARGET_D2]
        self.X_test = self.df_test[self.features]
        self.y_test = self.df_test[TARGET_D2]

    def test_d2_pipeline_execution_and_metrics(self):
        pipeline = build_pipeline(NUM_D2, CAT_D2, ORD_D2)

        # 1. Fit trên log1p của target
        y_train_log = np.log1p(self.y_train)
        pipeline.fit(self.X_train, y_train_log)

        # 2. Predict và expm1
        y_train_pred_log = pipeline.predict(self.X_train)
        y_test_pred_log = pipeline.predict(self.X_test)

        y_train_pred = np.expm1(y_train_pred_log)
        y_test_pred = np.expm1(y_test_pred_log)

        # 3. Kiểm tra tính hợp lệ của dự đoán
        assert not np.isnan(y_test_pred).any(), "[NGHIÊM TRỌNG] Dự đoán Bộ 2 chứa NaN"
        assert not np.isinf(y_test_pred).any(), "[NGHIÊM TRỌNG] Dự đoán Bộ 2 chứa Inf"
        assert (y_test_pred >= 0).all(), "[NGHIÊM TRỌNG] Dự đoán Bộ 2 có giá trị âm sau expm1"

        # 4. Tính toán số liệu thực tế
        # Thang đo gốc
        r2_orig_train = r2_score(self.y_train, y_train_pred)
        r2_orig_test = r2_score(self.y_test, y_test_pred)
        # Thang log
        r2_log_train = r2_score(y_train_log, y_train_pred_log)
        r2_log_test = r2_score(np.log1p(self.y_test), y_test_pred_log)

        # RMSLE
        rmsle_test = root_mean_squared_error(
            np.log1p(self.y_test), np.log1p(np.clip(y_test_pred, 0, None))
        )

        # 5. So sánh với DummyRegressor (strategy="median")
        dummy = DummyRegressor(strategy="median")
        dummy.fit(self.X_train, self.y_train)
        y_dummy_test = dummy.predict(self.X_test)
        r2_dummy = r2_score(self.y_test, y_dummy_test)
        rmsle_dummy = root_mean_squared_error(np.log1p(self.y_test), np.log1p(y_dummy_test))

        print("\n" + "=" * 70)
        print("KẾT QUẢ THỰC NGHIỆM SMOKE TEST BỘ 2 (Indian Urban):")
        print(f"  + Ridge R² test (thang gốc Lakhs): {r2_orig_test:.4f}")
        print(f"  + Ridge R² test (thang log)      : {r2_log_test:.4f}")
        print(f"  + Ridge R² train (thang gốc Lakhs): {r2_orig_train:.4f}")
        print(f"  + Ridge R² train (thang log)     : {r2_log_train:.4f}")
        print(f"  + Ridge RMSLE test               : {rmsle_test:.4f}")
        print(f"  + Dummy R² test (median)         : {r2_dummy:.4f}")
        print(f"  + Dummy RMSLE test (median)      : {rmsle_dummy:.4f}")
        print("=" * 70)

        # R² của Ridge phải vượt trội Dummy
        assert r2_orig_test > r2_dummy, (
            f"[NGHIÊM TRỌNG] Ridge R² ({r2_orig_test:.4f}) không thắng được Dummy ({r2_dummy:.4f})"
        )

        # Cảnh báo khoảng tin cậy R² test
        if r2_orig_test > 0.97:
            warnings.warn(f"[CẢNH BÁO] R² test quá cao ({r2_orig_test:.4f} > 0.97) - Nghi ngờ rò rỉ dữ liệu!")
        if r2_orig_test < 0.30:
            warnings.warn(f"[CẢNH BÁO] R² test quá thấp ({r2_orig_test:.4f} < 0.30) - Nghi ngờ lỗi tiền xử lý!")

        # Khoảng chấp nhận bình thường cho Ridge trên Indian Urban
        assert 0.30 <= r2_orig_test <= 0.97

    def test_d2_unseen_category_robustness(self):
        """Pipeline phải dự đoán bình thường khi gặp City hoặc Property_Type chưa từng thấy."""
        pipeline = build_pipeline(NUM_D2, CAT_D2, ORD_D2)
        pipeline.fit(self.X_train, np.log1p(self.y_train))

        unseen_sample = self.X_test.iloc[0:1].copy()
        unseen_sample["City"] = "Atlantis_Fantasy_City"
        unseen_sample["Property_Type"] = "Floating_Sky_Palace"

        pred_log = pipeline.predict(unseen_sample)
        pred = np.expm1(pred_log)

        assert not np.isnan(pred[0]), "Dự đoán trả về NaN khi gặp City/Property_Type lạ"
        assert pred[0] > 0, f"Dự đoán không hợp lệ: {pred[0]}"

