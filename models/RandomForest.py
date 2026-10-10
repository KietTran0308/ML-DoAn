import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

# Đường dẫn tới file dữ liệu và nơi lưu mô hình
DATA_PATH = '../data/processed/hcm_housing_clean.csv'
MODEL_PATH = 'random_forest_model.pkl'


def train_and_save_model(data_path, model_path):
    print("--- 1. BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH RANDOM FOREST ---")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file tại {data_path}.")
        return None, None, None, None

    df = df.dropna(subset=['price'])

    # 1. Bổ sung Features (X) và Log-transform Target (y)
    features = ['area', 'floors', 'bedrooms', 'bathrooms', 'frontage',
                'access_road', 'district', 'legal_status', 'furniture_state', 'house_direction']
    X = df[features]
    y = np.log1p(df['price'])  # Biến đổi Logarit cho giá nhà

    # 2. Cấu hình Preprocessing nâng cao
    numeric_features = ['area', 'floors', 'bedrooms', 'bathrooms', 'frontage', 'access_road']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())  # Chuẩn hóa dữ liệu
    ])

    categorical_features = ['district', 'legal_status', 'furniture_state', 'house_direction']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # 3. Tạo Pipeline với Random Forest
    pipeline_rf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # 4. Chia tập Train/Test
    X_train, X_test, y_train_log, y_test_log = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Huấn luyện mô hình
    print("Đang huấn luyện mô hình (sẽ mất vài giây)...")
    pipeline_rf.fit(X_train, y_train_log)

    # 6. Dự đoán và khôi phục giá trị thực (Expm1)
    y_test = np.expm1(y_test_log)
    y_pred_rf = np.expm1(pipeline_rf.predict(X_test))

    # 7. Đánh giá
    mae = mean_absolute_error(y_test, y_pred_rf)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2 = r2_score(y_test, y_pred_rf)

    print(f"-> RF MAE: {mae:.2f}")
    print(f"-> RF RMSE: {rmse:.2f}")
    print(f"-> RF R2: {r2:.4f}")

    # 8. Lưu mô hình
    joblib.dump(pipeline_rf, model_path)
    print(f"[Thành công] Đã lưu mô hình ra file: {model_path}\n")

    return pipeline_rf, X_test, y_test, y_pred_rf


def predict_new_house(model_path):
    print("--- 2. DỰ ĐOÁN GIÁ CĂN NHÀ MỚI ---")
    if not os.path.exists(model_path):
        print(f"Lỗi: Chưa có mô hình tại {model_path}.")
        return

    loaded_model = joblib.load(model_path)

    # Dữ liệu nhà mới phải có ĐẦY ĐỦ 10 cột như lúc Train
    new_house = pd.DataFrame({
        'area': [60.0],
        'floors': [2.0],
        'bedrooms': [2.0],
        'bathrooms': [2.0],
        'frontage': [4.0],  # Mặt tiền 4m
        'access_road': [5.0],  # Đường vào 5m
        'district': ['Gò Vấp'],
        'legal_status': ['Sổ hồng'],  # Hoặc np.nan nếu không biết
        'furniture_state': ['Cơ bản'],
        'house_direction': ['Đông Nam']
    })

    print("Thông tin căn nhà mới:")
    print(new_house.to_string(index=False))

    # Dự đoán (Kết quả đang ở dạng Log)
    predicted_log = loaded_model.predict(new_house)

    # Khôi phục giá trị thực bằng Expm1
    predicted_price = np.expm1(predicted_log)

    print(f"\n=> Giá nhà dự đoán là: {predicted_price[0]:.2f} (tỷ VNĐ)\n")


def visualize_results(y_test, y_pred):
    print("--- 3. HIỂN THỊ BIỂU ĐỒ ĐÁNH GIÁ ---")
    if y_test is None or y_pred is None:
        return

    plt.figure(figsize=(14, 6))

    # Biểu đồ 1: Thực Tế vs Dự Đoán
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.5, color='orange', edgecolors='k')
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2)
    plt.xlabel('Giá Thực Tế (Tỷ VNĐ)')
    plt.ylabel('Giá Dự Đoán (Tỷ VNĐ)')
    plt.title('Random Forest: Thực Tế vs Dự Đoán')
    plt.grid(True, linestyle=':', alpha=0.7)

    # Biểu đồ 2: Phân Phối Sai Số
    plt.subplot(1, 2, 2)
    residuals = y_test - y_pred
    sns.histplot(residuals, kde=True, color='purple', bins=30)
    plt.xlabel('Sai Số (Tỷ VNĐ)')
    plt.ylabel('Tần suất')
    plt.title('Random Forest: Phân Phối Sai Số')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2)

    plt.tight_layout()
    plt.show()
    print("Đã vẽ xong biểu đồ.")


if __name__ == "__main__":
    # 1. Huấn luyện và lưu mô hình
    model, X_test, y_test, y_pred = train_and_save_model(DATA_PATH, MODEL_PATH)

    # 2. Sử dụng mô hình vừa lưu để dự đoán
    predict_new_house(MODEL_PATH)

    # 3. Vẽ biểu đồ
    visualize_results(y_test, y_pred)