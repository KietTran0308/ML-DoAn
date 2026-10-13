import argparse
import json

import joblib
import pandas as pd
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from metrics import REPORTS_DIR, report, save_results
from preprocess import ROOT, build_preprocessor, load_split

VERSION = "v1"
MODEL_PATH = ROOT / "models" / "xgboost.pkl"

PARAM_DISTRIBUTIONS = {
    "model__n_estimators": [100, 300, 500],
    "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "model__max_depth": [5, 7, 9],
    "model__subsample": [0.7, 0.8, 1.0],
    "model__colsample_bytree": [0.7, 0.8, 1.0],
}


def build_pipeline(version=VERSION, **xgb_params):
    params = {"objective": "reg:squarederror", "random_state": 42}
    params.update(xgb_params)
    return Pipeline([
        ("prep", build_preprocessor(version)),
        ("model", XGBRegressor(**params)),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-iter", type=int, default=20, help="số tổ hợp tham số thử")
    parser.add_argument("--cv", type=int, default=5, help="số fold cross-validation")
    parser.add_argument("--skip-tuning", action="store_true")
    args = parser.parse_args()

    X_train, y_train = load_split("train", VERSION)
    X_dev, y_dev = load_split("dev", VERSION)
    splits = {"train": (X_train, y_train), "dev": (X_dev, y_dev)}

    print("--- 1. XGBoost tham số mặc định ---")
    default_pipe = build_pipeline().fit(X_train, y_train)
    results = [report("xgb_default", default_pipe, splits)]
    print(results[0].round(3).to_string(index=False))
    best_pipe = default_pipe

    if not args.skip_tuning:
        print(f"\n--- 2. Tuning: {args.n_iter} tổ hợp x {args.cv} fold (chỉ trên train) ---")
        # n_jobs=1 cho XGB để RandomizedSearchCV song song theo fold, tránh tranh CPU
        search = RandomizedSearchCV(
            build_pipeline(n_jobs=1),
            param_distributions=PARAM_DISTRIBUTIONS,
            n_iter=args.n_iter,
            cv=KFold(n_splits=args.cv, shuffle=True, random_state=42),
            scoring="neg_mean_absolute_error",  # đo trên thang log
            random_state=42, n_jobs=-1, verbose=1,
        ).fit(X_train, y_train)
        print("Tham số tốt nhất:", search.best_params_)
        best_pipe = search.best_estimator_

        tuned = report("xgb_tuned", best_pipe, splits)
        results.append(tuned)
        print(tuned.round(3).to_string(index=False))

        REPORTS_DIR.mkdir(exist_ok=True)
        with open(REPORTS_DIR / "xgb_best_params.json", "w", encoding="utf-8") as f:
            json.dump(search.best_params_, f, indent=2, default=int)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(best_pipe, MODEL_PATH)
    print(f"[Đã lưu mô hình] {MODEL_PATH}")
    save_results(pd.concat(results, ignore_index=True), "results_xgb.csv")

if __name__ == "__main__":
    main()