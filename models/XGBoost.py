import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

print("1. Đang đọc và tạo đặc trưng mới (Feature Engineering)...")
df = pd.read_csv('../data/processed/hcm_housing_clean.csv').dropna(subset=['price'])

# --- FEATURE ENGINEERING (KỸ NGHỆ ĐẶC TRƯNG) ---
# Tự tạo ra các biến mới giúp mô hình "thông minh" hơn
df['avg_area_per_floor'] = df['area'] / df['floors'].replace(0, 1) # Tránh chia cho 0
df['total_rooms'] = df['bedrooms'] + df['bathrooms']
df['has_frontage'] = df['frontage'].apply(lambda x: 1 if pd.notnull(x) and x > 0 else 0)

features = ['area', 'floors', 'bedrooms', 'bathrooms', 'frontage', 'access_road',
            'district', 'legal_status', 'furniture_state', 'house_direction',
            'avg_area_per_floor', 'total_rooms', 'has_frontage']

X = df[features]
y = np.log1p(df['price']) # Vẫn giữ Log-transform vì rất hiệu quả với giá nhà

print("2. Cấu hình Pipeline Tiền xử lý...")
numeric_features = ['area', 'floors', 'bedrooms', 'bathrooms', 'frontage', 'access_road', 'avg_area_per_floor', 'total_rooms']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_features = ['district', 'legal_status', 'furniture_state', 'house_direction', 'has_frontage']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Gắn XGBoost vào Pipeline
pipeline_xgb = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', XGBRegressor(random_state=42, objective='reg:squarederror'))
])

X_train, X_test, y_train_log, y_test_log = train_test_split(X, y, test_size=0.2, random_state=42)

print("3. Đang dò tìm siêu tham số tối ưu (Hyperparameter Tuning)...")
# Định nghĩa không gian tham số cho XGBoost để thuật toán tự thử nghiệm
param_distributions = {
    'model__n_estimators': [100, 300, 500],
    'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'model__max_depth': [5, 7, 9],
    'model__subsample': [0.7, 0.8, 1.0],
    'model__colsample_bytree': [0.7, 0.8, 1.0]
}

# Sử dụng K-Fold (cv=5) để đánh giá chéo, đảm bảo mô hình không bị Overfitting
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# RandomizedSearchCV sẽ thử ngẫu nhiên 20 tổ hợp tham số (n_iter=20) để tìm ra bộ tốt nhất
random_search = RandomizedSearchCV(
    pipeline_xgb,
    param_distributions=param_distributions,
    n_iter=20,
    cv=cv,
    scoring='neg_mean_absolute_error',
    verbose=1,
    random_state=42,
    n_jobs=-1 # Sử dụng tất cả nhân CPU để chạy nhanh hơn
)

random_search.fit(X_train, y_train_log)

print("\n--- KẾT QUẢ TỐI ƯU ---")
print(f"Bộ tham số tốt nhất tìm được: {random_search.best_params_}")

# Lấy mô hình tốt nhất để dự đoán
best_model = random_search.best_estimator_

print("\n4. Đánh giá trên tập Test...")
y_test = np.expm1(y_test_log)
y_pred = np.expm1(best_model.predict(X_test))

print(f"-> Tối ưu MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"-> Tối ưu RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"-> Tối ưu R2: {r2_score(y_test, y_pred):.4f}")

# Lưu mô hình tốt nhất lại
joblib.dump(best_model, 'xgboost_optimized_model.pkl')
print("\nĐã lưu mô hình XGBoost siêu tối ưu thành 'xgboost_optimized_model.pkl'")