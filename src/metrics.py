from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             median_absolute_error, r2_score)

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"

def evaluate(y_true_log, y_pred_log):
    y_true, y_pred = np.expm1(y_true_log), np.expm1(y_pred_log)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MedAE": median_absolute_error(y_true, y_pred),
        "MAPE_%": float(np.mean(np.abs(y_true - y_pred) / y_true) * 100),  # price >= 1 nên không chia 0
        "R2_log": r2_score(y_true_log, y_pred_log),
    }


def report(model_name, pipe, splits):
    rows = [{"model": model_name, "split": s, **evaluate(y, pipe.predict(X))}
            for s, (X, y) in splits.items()]
    return pd.DataFrame(rows)


def save_results(df, filename):
    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / filename
    df.to_csv(path, index=False)
    print(f"[Đã lưu kết quả] {path}")