"""Loại outlier và ghi lại số dòng bị loại (task T11).

Chạy:  python src/clean_data.py
Đọc   data/processed/hcm_housing_clean.csv
Ghi   data/processed/hcm_housing_filtered.csv   (đầu vào cho bước chia train/dev/test, T12)
      reports/outlier_decision.csv              (bảng quy tắc -> số dòng, dùng cho báo cáo)
Chạy TRƯỚC khi chia train/dev/test để cả 3 tập được làm sạch giống nhau.
"""
import pandas as pd

from metrics import save_results
from preprocess import DATA_DIR

RAW_FILE = DATA_DIR / "hcm_housing_clean.csv"
OUT_FILE = DATA_DIR / "hcm_housing_filtered.csv"

# Ngưỡng: sửa ở đây rồi chạy lại, bảng kết quả tự cập nhật
AREA_MIN, AREA_MAX = 10, 200           # m2
FRONTAGE_MAX = 20                      # m
PPM_RATIO_LOW, PPM_RATIO_HIGH = 1 / 3, 3   # giá/m2 so với trung vị giá/m2 CỦA CÙNG QUẬN
MAX_DROP_PCT = 2.0                     # nguyên tắc nhóm: loại không quá ~2% dữ liệu


def outlier_rules(df):
    """Trả về dict {tên quy tắc: mask True = dòng vi phạm}."""
    ppm = df["price"] * 1000 / df["area"]                  # triệu đồng / m2
    ratio = ppm / ppm.groupby(df["district"]).transform("median")
    return {
        f"area < {AREA_MIN} m2": df["area"] < AREA_MIN,
        f"area > {AREA_MAX} m2": df["area"] > AREA_MAX,
        f"frontage > {FRONTAGE_MAX} m": df["frontage"] > FRONTAGE_MAX,
        f"giá/m2 < {PPM_RATIO_LOW:.2f} x trung vị quận": ratio < PPM_RATIO_LOW,
        f"giá/m2 > {PPM_RATIO_HIGH:g} x trung vị quận": ratio > PPM_RATIO_HIGH,
    }


def remove_outliers(df):
    """Trả về (df đã lọc, bảng thống kê từng quy tắc)."""
    rules = outlier_rules(df)
    hits = pd.concat(rules, axis=1)
    drop = hits.any(axis=1)
    only = hits[hits.sum(axis=1) == 1]                     # dòng chỉ vi phạm đúng 1 quy tắc
    table = pd.DataFrame({
        "rule": list(rules),
        "n_rows": [int(m.sum()) for m in rules.values()],
        "pct_of_total": [round(m.mean() * 100, 2) for m in rules.values()],
        "n_only_this_rule": [int(only[k].sum()) for k in rules],
    })
    total = pd.DataFrame([{"rule": "TỔNG (hợp các quy tắc)", "n_rows": int(drop.sum()),
                           "pct_of_total": round(drop.mean() * 100, 2),
                           "n_only_this_rule": ""}])
    return df[~drop].copy(), pd.concat([table, total], ignore_index=True)


def main():
    df = pd.read_csv(RAW_FILE, encoding="utf-8-sig").dropna(subset=["price"])
    print(f"Ban đầu: {len(df)} dòng")

    # Kiểm tra giá bị cắt trần: nếu nhiều dòng dồn đúng 1 giá = giá lớn nhất -> dữ liệu bị giới hạn
    top = df["price"].value_counts().head(3)
    print(f"Giá lớn nhất = {df['price'].max()}; dòng có giá = {df['price'].max()}: "
          f"{int((df['price'] == df['price'].max()).sum())}")
    print("3 mức giá phổ biến nhất:", top.to_dict())
    print(f"Dòng có giá = 10.0: {int((df['price'] == 10).sum())}, giá > 10: {int((df['price'] > 10).sum())}")

    clean, table = remove_outliers(df)
    print("\n" + table.to_string(index=False))
    pct = table.iloc[-1]["pct_of_total"]
    print(f"\nCòn lại: {len(clean)} dòng (loại {len(df) - len(clean)} = {pct}%)")
    if pct > MAX_DROP_PCT:
        print(f"CẢNH BÁO: loại {pct}% > {MAX_DROP_PCT}%. Cân nhắc nới ngưỡng.")

    clean.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    print(f"[Đã lưu dữ liệu sạch] {OUT_FILE}")
    save_results(table, "outlier_decision.csv")


if __name__ == "__main__":
    main()