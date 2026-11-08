"""Script tự động tải và kiểm tra tính toàn vẹn của 2 bộ dữ liệu:
1. King County House Sales (21,613 dòng, 21 cột)
2. Indian Urban Housing (80,000 dòng train, 20,000 dòng test, 18 cột)

Yêu cầu:
- Đã cài thư viện: pip install kaggle kagglehub pandas
- Đã có file kaggle.json tại ~/.kaggle/kaggle.json (Tải từ Kaggle -> Settings -> Create New Token)
"""

import os
import shutil
import sys
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
D1_DIR = os.path.join(DATA_DIR, "d1_king_county")
D2_DIR = os.path.join(DATA_DIR, "d2_indian_urban")


def ensure_dirs():
    os.makedirs(D1_DIR, exist_ok=True)
    os.makedirs(D2_DIR, exist_ok=True)


def download_dataset_1():
    print("\n--- ĐANG TẢI BỘ 1: King County House Sales ---")
    dest_file = os.path.join(D1_DIR, "kc_house_data.csv")
    if os.path.exists(dest_file):
        print(f"-> Đã tồn tại: {dest_file}")
    else:
        try:
            import kagglehub

            path = kagglehub.dataset_download("harlfoxem/housesalesprediction")
            src_file = os.path.join(path, "kc_house_data.csv")
            if os.path.exists(src_file):
                shutil.copy(src_file, dest_file)
                print(f"-> Đã tải thành công về: {dest_file}")
            else:
                for f in os.listdir(path):
                    if f.endswith(".csv"):
                        shutil.copy(os.path.join(path, f), dest_file)
                        break
        except Exception as e:
            print(f"[!] Không thể tải tự động qua kagglehub: {e}")
            print(
                "    Vui lòng tải thủ công từ: https://www.kaggle.com/datasets/harlfoxem/housesalesprediction"
            )
            print(f"    và đặt file kc_house_data.csv vào {D1_DIR}")

    # Verify
    if os.path.exists(dest_file):
        df1 = pd.read_csv(dest_file)
        print(f"   [V] Kiểm tra: Shape = {df1.shape} (Kỳ vọng: (21613, 21))")


def download_dataset_2():
    print("\n--- ĐANG TẢI BỘ 2: Indian Urban Housing ---")
    train_dest = os.path.join(D2_DIR, "train.csv")
    test_dest = os.path.join(D2_DIR, "test.csv")

    if os.path.exists(train_dest):
        print(f"-> Đã tồn tại: {train_dest}")
    else:
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()
            print("-> Đang tải competition: indian-urban-house-price-prediction...")
            api.competition_download_files(
                "indian-urban-house-price-prediction", path=D2_DIR, quiet=False
            )
            import zipfile

            zip_path = os.path.join(
                D2_DIR, "indian-urban-house-price-prediction.zip"
            )
            if os.path.exists(zip_path):
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(D2_DIR)
                os.remove(zip_path)
                print("-> Đã giải nén thành công.")
        except Exception as e:
            print(f"[!] Không thể tải tự động qua Kaggle API: {e}")
            print(
                "    Vui lòng tải thủ công từ: https://www.kaggle.com/competitions/indian-urban-house-price-prediction/data"
            )
            print(f"    và giải nén train.csv, test.csv vào {D2_DIR}")

    # Verify
    if os.path.exists(train_dest):
        df2_train = pd.read_csv(train_dest)
        print(
            f"   [V] Kiểm tra Train: Shape = {df2_train.shape} (Kỳ vọng: (80000, 18))"
        )
    if os.path.exists(test_dest):
        df2_test = pd.read_csv(test_dest)
        print(
            f"   [V] Kiểm tra Test: Shape = {df2_test.shape} (Kỳ vọng: (20000, 17 hoặc 18))"
        )


if __name__ == "__main__":
    ensure_dirs()
    download_dataset_1()
    download_dataset_2()
    print("\nHoàn tất kiểm tra dữ liệu!")

