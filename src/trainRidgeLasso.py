"""Ridge và Lasso với nhiều giá trị alpha, chọn alpha theo dev MAE (task T19).

Chạy:  python src/train_ridge_lasso.py
Dùng chung preprocess.py (13 feature, log1p(price)) với Linear/RF/XGB.
Chỉ dùng train và dev. Tập test bị khóa đến T40.
Kết quả: reports/results_ridge_lasso.csv, models/ridge.pkl, models/lasso.pkl (alpha tốt nhất của mỗi loại).
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline

from metrics import evaluate, save_results
from preprocess import ROOT, build_preprocessor, load_split

VERSION = "v1"
ALPHAS = [0.01, 0.1, 1, 10, 100]
MODELS_DIR = ROOT / "models"

REGRESSORS = {
    "ridge": lambda alpha: Ridge(alpha=alpha),
    "lasso": lambda alpha: Lasso(alpha=alpha, max_iter=10000),
}


def build_pipeline(regressor, version=VERSION):
    return Pipeline([("prep", build_preprocessor(version)), ("model", regressor)])


def run_one(name, alpha, regressor, data):
    """Fit trên train, đo train và dev. Trả về (dòng kết quả, pipeline)."""
    (X_train, y_train), (X_dev, y_dev) = data["train"], data["dev"]
    pipe = build_pipeline(regressor).fit(X_train, y_train)
    tr, dv = evaluate(y_train, pipe.predict(X_train)), evaluate(y_dev, pipe.predict(X_dev))
    coef = pipe.named_steps["model"].coef_
    row = {
        "model": name, "alpha": alpha,
        "train_MAE": tr["MAE"], "dev_MAE": dv["MAE"], "gap_MAE": dv["MAE"] - tr["MAE"],
        "train_MAPE_%": tr["MAPE_%"], "dev_MAPE_%": dv["MAPE_%"],
        "dev_R2_log": dv["R2_log"],
        "nonzero_coef": int(np.sum(coef != 0)), "total_coef": int(coef.size),
    }
    return row, pipe


def main():
    data = {s: load_split(s, VERSION) for s in ("train", "dev")}

    rows, pipes = [], {}
    row, _ = run_one("linear", 0, LinearRegression(), data)   # mốc tham chiếu: không regularization
    rows.append(row)
    for name, make in REGRESSORS.items():
        for alpha in ALPHAS:
            row, pipe = run_one(name, alpha, make(alpha), data)
            rows.append(row)
            pipes[(name, alpha)] = pipe

    res = pd.DataFrame(rows)
    print(res.round(3).to_string(index=False))

    MODELS_DIR.mkdir(exist_ok=True)
    for name in REGRESSORS:
        sub = res[res["model"] == name]
        best_alpha = sub.loc[sub["dev_MAE"].idxmin(), "alpha"]
        print(f"\n{name}: alpha tốt nhất theo dev MAE = {best_alpha}")
        joblib.dump(pipes[(name, best_alpha)], MODELS_DIR / f"{name}.pkl")

    save_results(res, "results_ridge_lasso.csv")


if __name__ == "__main__":
    main()