import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src")) # chứa preprocess.py
from preprocess import get_raw_columns

# Tên mô hình -> file trong thư mục models/ (mô hình đầu tiên là mặc định)
MODEL_FILES = {
    "XGBoost": "xgboost.pkl",
    "Random Forest": "random_forest.pkl",
    "Linear Regression": "linear.pkl",
    "Ridge Regression": "ridge.pkl",
    "Lasso Regression": "lasso.pkl",
}


@lru_cache(maxsize=10)
def load_model(path, modified_time):
    """Chỉ nạp lại khi file model thay đổi."""
    return joblib.load(path)


NUM_COLS = ["area", "floors", "bedrooms", "bathrooms", "frontage", "access_road"]

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
    row = pd.DataFrame([data]).reindex(columns=get_raw_columns())
    cat_cols = [c for c in row.columns if c not in NUM_COLS]
    row[NUM_COLS] = row[NUM_COLS].astype(float)
    row[cat_cols] = row[cat_cols].astype(object).where(row[cat_cols].notna(), np.nan)
    price = float(np.expm1(model.predict(row)[0]))   # đổi từ log về tỷ đồng
    return {"model": house.model, "predicted_price": round(price, 2), "unit": "tỷ đồng"}