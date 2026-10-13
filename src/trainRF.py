import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from metrics import report, save_results
from preprocess import ROOT, build_preprocessor, load_split

VERSION = "v1"
MODEL_PATH = ROOT / "models" / "random_forest.pkl"


def build_pipeline(version=VERSION, **rf_params):
    params = {"n_estimators": 100, "random_state": 42, "n_jobs": -1}
    params.update(rf_params)  # T36 sẽ truyền max_depth, min_samples_leaf... vào đây
    return Pipeline([
        ("prep", build_preprocessor(version)),
        ("model", RandomForestRegressor(**params)),
    ])


def main():
    print("--- Random Forest ---")
    X_train, y_train = load_split("train", VERSION)
    X_dev, y_dev = load_split("dev", VERSION)

    pipe = build_pipeline().fit(X_train, y_train)

    res = report("random_forest", pipe, {"train": (X_train, y_train), "dev": (X_dev, y_dev)})
    print(res.round(3).to_string(index=False))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f"[Đã lưu mô hình] {MODEL_PATH}")
    save_results(res, "results_rf.csv")


if __name__ == "__main__":
    main()