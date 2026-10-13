import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from metrics import report, save_results
from preprocess import load_split


class DistrictMedianBaseline(BaseEstimator, RegressorMixin):

    def fit(self, X, y):
        df = pd.DataFrame({"district": X["district"].to_numpy(), "y": np.asarray(y)})
        self.medians_ = df.groupby("district")["y"].median().to_dict()
        self.global_median_ = float(np.median(df["y"]))
        return self

    def predict(self, X):
        return X["district"].map(self.medians_).fillna(self.global_median_).to_numpy()


def main():
    print("--- Baseline: trung vị giá theo quận ---")
    X_train, y_train = load_split("train")
    X_dev, y_dev = load_split("dev")

    model = DistrictMedianBaseline().fit(X_train, y_train)

    n_unseen = int((~X_dev["district"].isin(model.medians_)).sum())
    print(f"Số quận trong train: {len(model.medians_)} | dòng dev có quận chưa thấy: {n_unseen}")

    res = report("baseline_district_median", model,
                 {"train": (X_train, y_train), "dev": (X_dev, y_dev)})
    print(res.round(3).to_string(index=False))
    save_results(res, "results_baseline.csv")

    # Bảng trung vị theo quận (đã đổi về tỷ VNĐ) để dùng cho phân tích lỗi / báo cáo
    med = pd.DataFrame({"district": list(model.medians_),
                        "median_price": np.expm1(list(model.medians_.values())),
                        "n_train": X_train["district"].value_counts().reindex(list(model.medians_)).to_numpy()})
    save_results(med.sort_values("median_price", ascending=False), "baseline_district_median.csv")


if __name__ == "__main__":
    main()