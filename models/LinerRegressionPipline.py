import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

# Đường dẫn tới file dữ liệu và nơi lưu mô hình
DATA_PATH = '../data/processed/hcm_housing_clean.csv'
MODEL_PATH = 'linear_regression_model.pkl'


def train_and_save_model(data_path, model_path):
    print("--- 1. BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH ---")
    # Đọc và làm sạch dữ liệu
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file dữ liệu tại {data_path}. Vui lòng kiểm tra lại đường dẫn.")
        return None, None, None, None, None

    df = df.dropna(subset=['price'])

    # Khai báo Features (X) và Target (y)
    X = df[['area', 'floors', 'bedrooms', 'bathrooms', 'district']]
    y = df['price']

    # Cấu hình tiền xử lý
    numeric_features = ['area', 'floors', 'bedrooms', 'bathrooms']
    numeric_transformer = SimpleImputer(strategy='median')

    categorical_features = ['district']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Đóng gói mô hình Linear Regression vào Pipeline
    pipeline_lr = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    # Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Huấn luyện (Train) mô hình
    print("Đang huấn luyện mô hình Linear Regression...")
    pipeline_lr.fit(X_train, y_train)

    # Đánh giá sơ bộ
    y_pred = pipeline_lr.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"-> MAE: {mae:.2f}")
    print(f"-> RMSE: {rmse:.2f}")
    print(f"-> R2: {r2:.4f}")

    # Tách và lưu mô hình ra file
    joblib.dump(pipeline_lr, model_path)
    print(f"[Thành công] Đã tách và lưu mô hình ra file: {model_path}\n")

    return pipeline_lr, X_test, y_test, y_pred


def predict_new_house(model_path):
    print("--- 2. DỰ ĐOÁN GIÁ CĂN NHÀ MỚI ---")
    if not os.path.exists(model_path):
        print(f"Lỗi: Chưa có mô hình tại {model_path}. Vui lòng train mô hình trước.")
        return

    # Mở (Load) mô hình từ file .pkl
    loaded_model = joblib.load(model_path)

    # Tạo dữ liệu cho một căn nhà mới để dự đoán
    new_house = pd.DataFrame({
        'area': [60.0],
        'floors': [2.0],
        'bedrooms': [2.0],
        'bathrooms': [2.0],
        'district': ['Gò Vấp']
    })

    print("Thông tin căn nhà mới:")
    print(new_house)

    # Sử dụng mô hình đã mở để dự đoán
    predicted_price = loaded_model.predict(new_house)
    print(f"=> Giá nhà dự đoán là: {predicted_price[0]:.2f} (tỷ VNĐ)\n")


def visualize_results(y_test, y_pred):
    print("--- 3. HIỂN THỊ BIỂU ĐỒ ĐÁNH GIÁ ---")
    if y_test is None or y_pred is None:
        print("Không có dữ liệu để vẽ biểu đồ.")
        return

    # Cấu hình không gian vẽ biểu đồ (1 hàng, 2 cột)
    plt.figure(figsize=(14, 6))

    # --- BIỂU ĐỒ 1: Thực Tế vs Dự Đoán (Scatter Plot) ---
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.5, color='blue', edgecolors='k')

    # Vẽ đường chéo y = x (Đường dự đoán hoàn hảo)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2)

    plt.xlabel('Giá Thực Tế (Tỷ VNĐ)')
    plt.ylabel('Giá Dự Đoán (Tỷ VNĐ)')
    plt.title('Biểu Đồ Thực Tế vs Dự Đoán')
    plt.grid(True, linestyle=':', alpha=0.7)

    # --- BIỂU ĐỒ 2: Phân Phối Sai Số (Residual Histogram) ---
    plt.subplot(1, 2, 2)
    residuals = y_test - y_pred  # Sai số = Thực tế - Dự đoán
    sns.histplot(residuals, kde=True, color='green', bins=30)

    plt.xlabel('Sai Số (Tỷ VNĐ)')
    plt.ylabel('Tần suất (Số lượng nhà)')
    plt.title('Phân Phối Sai Số (Residuals)')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2)  # Đường chuẩn 0

    # Hiển thị biểu đồ
    plt.tight_layout()
    plt.show()
    print("Đã vẽ xong biểu đồ.")


if __name__ == "__main__":
    # Chạy quy trình từ A-Z
    # 1. Huấn luyện và lưu mô hình
    model, X_test, y_test, y_pred = train_and_save_model(DATA_PATH, MODEL_PATH)

    # 2. Sử dụng mô hình vừa lưu để dự đoán nhà mới
    predict_new_house(MODEL_PATH)

    # 3. Vẽ biểu đồ đánh giá
    visualize_results(y_test, y_pred)
