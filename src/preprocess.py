#Tiền xử lý dùng chung cho Linear, Random Forest, XGBoost.
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
TARGET = "price"

# 10 cột gốc + 3 cột tạo thêm = 13 feature (v1)
RAW_NUMERIC = ["area", "floors", "bedrooms", "bathrooms", "frontage", "access_road"]
RAW_CATEGORICAL = ["district", "legal_status", "furniture_state", "house_direction"]
ENGINEERED_NUMERIC = ["avg_area_per_floor", "total_rooms"]
ENGINEERED_BINARY = ["has_frontage"]          # 0/1, KHÔNG đưa vào nhóm categorical

FEATURE_SETS = {
    "v1": {
        "numeric": RAW_NUMERIC + ENGINEERED_NUMERIC,
        "binary": ENGINEERED_BINARY,
        "categorical": RAW_CATEGORICAL,
    },
    # "v2": thêm ở task T35 (giá/m² theo quận, cờ hẻm nhỏ, ...)
}


def get_features(version="v1"):
    """Trả về danh sách 13 tên cột feature theo thứ tự cố định."""
    fs = FEATURE_SETS[version]
    return fs["numeric"] + fs["binary"] + fs["categorical"]


def get_raw_columns(version="v1"):
    """10 cột gốc mà người dùng/predict.py phải cung cấp."""
    return RAW_NUMERIC + RAW_CATEGORICAL


def add_features(X):
    """Tạo 3 feature mới từ 10 cột gốc. Không sửa DataFrame đầu vào."""
    X = X.copy()
    X["avg_area_per_floor"] = X["area"] / X["floors"].replace(0, 1)  # tránh chia 0
    X["total_rooms"] = X["bedrooms"] + X["bathrooms"]                # NaN nếu thiếu 1 trong 2
    X["has_frontage"] = (X["frontage"].notna() & (X["frontage"] > 0)).astype(int)
    return X


def build_preprocessor(version="v1"):
    """Bước tiền xử lý nhận 10 cột GỐC, tự tạo feature, rồi impute/scale/one-hot.

    Vì add_features nằm trong pipeline nên predict.py chỉ cần đưa 10 cột gốc.
    """
    fs = FEATURE_SETS[version]
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    column_step = ColumnTransformer([
        ("num", numeric, fs["numeric"]),
        ("bin", "passthrough", fs["binary"]),
        ("cat", categorical, fs["categorical"]),
    ])
    return Pipeline([
        ("features", FunctionTransformer(add_features)),
        ("columns", column_step),
    ])


def load_split(name, version="v1", data_dir=DATA_DIR):
    if name not in ("train", "dev", "test"):
        raise ValueError("name phải là 'train', 'dev' hoặc 'test'")
    path = Path(data_dir) / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Chưa có {path}. Chạy bước chia dữ liệu (T12) trước.")
    df = pd.read_csv(path, encoding="utf-8-sig").dropna(subset=[TARGET])
    return df[get_raw_columns(version)], np.log1p(df[TARGET])

