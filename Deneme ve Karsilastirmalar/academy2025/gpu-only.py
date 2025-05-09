# %% Gerekli kütüphanelerin yüklenmesi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# %% Veri setlerinin okunması
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('testFeatures.csv')
sample_submission = pd.read_csv('sample_submission.csv')

# %% Tarih özelliklerinin ayrılması
for df in [train_df, test_df]:
    df['tarih'] = pd.to_datetime(df['tarih'])
    df['yıl'] = df['tarih'].dt.year
    df['ay'] = df['tarih'].dt.month
    df['gün'] = df['tarih'].dt.day
    df['hafta_günü'] = df['tarih'].dt.dayofweek

# %% Hedef ve özelliklerin belirlenmesi
target = '""'
if '""' not in train_df.columns:
    target = '"\u00fcr\u00fcn fiyat\u0131"'  # fallback

categorical_cols = train_df.select_dtypes(include=['object', 'category']).columns.tolist()
if 'tarih' in categorical_cols:
    categorical_cols.remove('tarih')
numerical_cols = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if target in numerical_cols:
    numerical_cols.remove(target)

# %% Preprocessing pipeline
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ('cat', categorical_transformer, categorical_cols),
    ('num', numerical_transformer, numerical_cols)
])

# %% Eğitim-veri ayrımı
X = train_df.drop([target, 'tarih'], axis=1)
y = train_df[target]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# %% Verinin preprocess edilmesi
X_train_proc = preprocessor.fit_transform(X_train)
X_val_proc = preprocessor.transform(X_val)
X_full_proc = preprocessor.fit_transform(X)
X_test_proc = preprocessor.transform(test_df.drop(['tarih'], axis=1))

# %% GPU destekli XGBoost modeli
xgb_model = xgb.XGBRegressor(
    tree_method='gpu_hist',
    predictor='gpu_predictor',
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    verbosity=1
)

xgb_model.fit(X_train_proc, y_train)
y_pred_xgb = xgb_model.predict(X_val_proc)
xgb_rmse = np.sqrt(mean_squared_error(y_val, y_pred_xgb))
print(f"XGBoost (GPU) Validation RMSE: {xgb_rmse:.4f}")

# %% GPU destekli LightGBM modeli
lgb_model = lgb.LGBMRegressor(
    device='gpu',
    boosting_type='gbdt',
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)

lgb_model.fit(X_train_proc, y_train)
y_pred_lgb = lgb_model.predict(X_val_proc)
lgb_rmse = np.sqrt(mean_squared_error(y_val, y_pred_lgb))
print(f"LightGBM (GPU) Validation RMSE: {lgb_rmse:.4f}")

# %% Tahmin ve submission (XGBoost seçilirse)
test_predictions = xgb_model.predict(X_test_proc)
submission = sample_submission.copy()
submission[target] = test_predictions
submission.to_csv('submission_gpu_xgb.csv', index=False)
print("GPU destekli XGBoost submission dosyası kaydedildi.")
