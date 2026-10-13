import joblib
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from metrics import report, save_results
from preprocess import ROOT, build_preprocessor, load_split

VERSION = "v1"
MODEL_PATH = ROOT / "models" / "linear.pkl"


def build_pipeline(version=VERSION):
    return Pipeline([
        ("prep", build_preprocessor(version)),
        ("model", LinearRegression()),
    ])


def main():
    print("--- Linear Regression ---")
    X_train, y_train = load_split("train", VERSION)
    X_dev, y_dev = load_split("dev", VERSION)

    pipe = build_pipeline().fit(X_train, y_train)

    res = report("linear", pipe, {"train": (X_train, y_train), "dev": (X_dev, y_dev)})
    print(res.round(3).to_string(index=False))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f"[Đã lưu mô hình] {MODEL_PATH}")
    save_results(res, "results_linear.csv")


if __name__ == "__main__":
    main()