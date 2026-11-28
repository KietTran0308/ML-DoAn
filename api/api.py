import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent  # thư mục dự án (chứa app.py)
sys.path.insert(0, str(ROOT / "src")) # chứa preprocess.py
from preprocess import add_features, get_raw_columns

# Tên mô hình -> file trong thư mục models/ (mô hình đầu tiên là mặc định)
MODEL_FILES = {
    "XGBoost": "xgboost.pkl",
    "XGBoost Optimized": "xgboost_optimized_model.pkl",
    "Random Forest": "random_forest.pkl",
    "Random Forest (model 2)": "random_forest_model.pkl",
    "Linear Regression": "linear.pkl",
    "Linear Regression (Liner_Model)": "Liner_Model.pkl",
    "Ridge Regression": "ridge.pkl",
    "Lasso Regression": "lasso.pkl",
}
# Các file dự đoán thẳng giá (tỷ đồng); file còn lại dự đoán log1p(price)
RAW_PRICE_FILES = {"Liner_Model.pkl"}


@lru_cache(maxsize=10)
def load_model(path, modified_time):
    """Chỉ nạp lại khi file model thay đổi."""
    return joblib.load(path)


def predict_price(model, file, df):
    """Chuẩn bị đúng cột mà từng model cần, trả về giá dự đoán (tỷ đồng)."""
    num_cols = ["area", "floors", "bedrooms", "bathrooms", "frontage", "access_road"]
    raw_cols = list(get_raw_columns())
    needed = list(getattr(model, "feature_names_in_", raw_cols))
    cols = list(dict.fromkeys(raw_cols + [c for c in needed if c in df.columns]))
    X = df.reindex(columns=cols)
    nums = [c for c in num_cols if c in X.columns]
    cats = [c for c in raw_cols if c not in num_cols]
    X[nums] = X[nums].astype(float)
    X[cats] = X[cats].astype(object).where(X[cats].notna(), np.nan)
    if any(c not in X.columns for c in needed):
        X = add_features(X.copy())   # tạo thêm các cột đặc trưng
    missing = [c for c in needed if c not in X.columns]
    if missing:
        raise ValueError(f"Thiếu cột: {', '.join(missing)}")
    pred = np.asarray(model.predict(X[needed]), dtype=float)
    return pred if file in RAW_PRICE_FILES else np.expm1(pred)


app = FastAPI(title="API dự đoán giá nhà TP.HCM")


class House(BaseModel):
    model: str = Field("XGBoost", examples=["XGBoost"])
    area: float = Field(gt=0, examples=[60])
    district: str = Field(min_length=1, examples=["Gò Vấp"])
    floors: Optional[int] = Field(None, ge=1, examples=[3])
    bedrooms: Optional[int] = Field(None, ge=1, examples=[3])
    bathrooms: Optional[int] = Field(None, ge=1, examples=[3])
    frontage: Optional[float] = Field(None, gt=0, examples=[4.5])
    access_road: Optional[float] = Field(None, gt=0, examples=[6])
    legal_status: Optional[str] = Field(None, examples=["Have certificate"])
    furniture_state: Optional[str] = Field(None, examples=["Full"])
    house_direction: Optional[str] = Field(None, examples=["Đông"])


@app.get("/models")
def list_models():
    return {"models": [n for n, f in MODEL_FILES.items() if (ROOT / "models" / f).exists()]}


@app.post("/predict")
def predict(house: House):
    if house.model not in MODEL_FILES:
        raise HTTPException(400, f"Không có mô hình tên '{house.model}'")
    path = ROOT / "models" / MODEL_FILES[house.model]
    if not path.exists():
        raise HTTPException(404, f"Không tìm thấy file {path.name}")
    try:
        model = load_model(str(path), path.stat().st_mtime)
    except Exception as e:
        raise HTTPException(500, f"Không nạp được model: {e}")

    data = house.model_dump(exclude={"model"})
    try:
        price = float(predict_price(model, path.name, pd.DataFrame([data]))[0])
    except Exception as e:
        raise HTTPException(500, f"Không dự đoán được: {e}")
    return {"model": house.model, "predicted_price": round(price, 2), "unit": "tỷ đồng"}