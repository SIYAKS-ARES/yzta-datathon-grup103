# %% Gerekli kütüphanelerin yüklenmesi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# %% Veri setlerinin okunması
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('testFeatures.csv')
sample_submission = pd.read_csv('sample_submission.csv')

# %% Tarih kolonu işlenmesi
if 'tarih' in train_df.columns:
    train_df['tarih'] = pd.to_datetime(train_df['tarih'])
    train_df['yıl'] = train_df['tarih'].dt.year
    train_df['ay'] = train_df['tarih'].dt.month
    train_df['gün'] = train_df['tarih'].dt.day
    train_df['hafta_günü'] = train_df['tarih'].dt.dayofweek

if 'tarih' in test_df.columns:
    test_df['tarih'] = pd.to_datetime(test_df['tarih'])
    test_df['yıl'] = test_df['tarih'].dt.year
    test_df['ay'] = test_df['tarih'].dt.month
    test_df['gün'] = test_df['tarih'].dt.day
    test_df['hafta_günü'] = test_df['tarih'].dt.dayofweek

# %% Değii\u015kenlerin ayrılması
target = 'ürün fiyatı'
categorical_cols = train_df.select_dtypes(include=['object', 'category']).columns.tolist()
if 'tarih' in categorical_cols:
    categorical_cols.remove('tarih')
numerical_cols = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if target in numerical_cols:
    numerical_cols.remove(target)

# %% Pipeline tanımı
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

numerical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer([
    ('cat', categorical_transformer, categorical_cols),
    ('num', numerical_transformer, numerical_cols)
])

# %% Model ayrımı
X = train_df.drop([target, 'tarih'], axis=1)
y = train_df[target]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Ridge': Pipeline([('preprocessor', preprocessor), ('model', Ridge(alpha=1.0))]),
    'Lasso': Pipeline([('preprocessor', preprocessor), ('model', Lasso(alpha=0.001))]),
    'RandomForest': Pipeline([('preprocessor', preprocessor), ('model', RandomForestRegressor(n_estimators=100, random_state=42))]),
    'GradientBoosting': Pipeline([('preprocessor', preprocessor), ('model', GradientBoostingRegressor(n_estimators=100, random_state=42))])
}

# %% Model karşılaştırma
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    cv_rmse = -np.mean(cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=5))
    results[name] = {'Validation RMSE': val_rmse, 'CV RMSE': cv_rmse}
    print(f"{name} - Validation RMSE: {val_rmse:.4f}, CV RMSE: {cv_rmse:.4f}")

# %% En iyi model seçimi
best_model_name = min(results, key=lambda x: results[x]['CV RMSE'])
best_model = models[best_model_name]
print(f"En iyi model: {best_model_name} (CV RMSE: {results[best_model_name]['CV RMSE']:.4f})")

# %% Model tipine göre parametre gridleri
if best_model_name == 'Ridge':
    param_grid_light = {'model__alpha': [0.1, 1.0, 10.0], 'model__solver': ['auto', 'svd']}
elif best_model_name == 'Lasso':
    param_grid_light = {'model__alpha': [0.0001, 0.001, 0.01], 'model__max_iter': [1000, 2000]}
elif best_model_name in ['RandomForest', 'GradientBoosting']:
    param_grid_light = {'model__n_estimators': [50, 100], 'model__max_depth': [10, 20]}

# %% Daha hafif parametre grid
light_grid_search = GridSearchCV(best_model, param_grid_light, scoring='neg_root_mean_squared_error', cv=3, n_jobs=-1)
light_grid_search.fit(X, y)
print(f"Hafif GridSearch En iyi: {light_grid_search.best_params_}, RMSE: {-light_grid_search.best_score_:.4f}")

# %% RandomizedSearchCV - model tipine göre parametre dağılımı
if best_model_name == 'Ridge':
    param_dist = {'model__alpha': [0.01, 0.1, 1.0, 10.0, 100.0], 'model__solver': ['auto', 'svd', 'cholesky']}
elif best_model_name == 'Lasso':
    param_dist = {'model__alpha': [0.0001, 0.001, 0.01, 0.1], 'model__max_iter': [1000, 2000, 3000]}
elif best_model_name in ['RandomForest', 'GradientBoosting']:
    param_dist = {'model__n_estimators': [50, 100, 150], 'model__max_depth': [5, 10, 15, 20]}

random_search = RandomizedSearchCV(best_model, param_distributions=param_dist, n_iter=5, scoring='neg_root_mean_squared_error', cv=3, n_jobs=-1, random_state=42)
random_search.fit(X, y)
print(f"RandomizedSearchCV En iyi: {random_search.best_params_}, RMSE: {-random_search.best_score_:.4f}")

# %% XGBoost + GPU
X_preprocessed = preprocessor.fit_transform(X_train)
X_val_preprocessed = preprocessor.transform(X_val)
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
xgb_model.fit(X_preprocessed, y_train)
val_rmse_xgb = np.sqrt(mean_squared_error(y_val, xgb_model.predict(X_val_preprocessed)))
print(f"XGBoost - Validation RMSE: {val_rmse_xgb:.4f}")

# %% GridSearch sadece X_train ile
if 'param_grid_light' not in locals():
    # Model tipine göre parametre gridleri tanımlanmamışsa (yeniden tanımla)
    if best_model_name == 'Ridge':
        param_grid_light = {'model__alpha': [0.1, 1.0, 10.0], 'model__solver': ['auto', 'svd']}
    elif best_model_name == 'Lasso':
        param_grid_light = {'model__alpha': [0.0001, 0.001, 0.01], 'model__max_iter': [1000, 2000]}
    elif best_model_name in ['RandomForest', 'GradientBoosting']:
        param_grid_light = {'model__n_estimators': [50, 100], 'model__max_depth': [10, 20]}

grid_train_only = GridSearchCV(best_model, param_grid_light, scoring='neg_root_mean_squared_error', cv=3, n_jobs=-1)
grid_train_only.fit(X_train, y_train)
print(f"Sadece train ile GridSearch: {grid_train_only.best_params_}, RMSE: {-grid_train_only.best_score_:.4f}")

# %% X.sample(frac=0.3) GridSearch
X_sample = X.sample(frac=0.3, random_state=42)
y_sample = y.loc[X_sample.index]

if 'param_grid_light' not in locals():
    # Model tipine göre parametre gridleri tanımlanmamışsa (yeniden tanımla)
    if best_model_name == 'Ridge':
        param_grid_light = {'model__alpha': [0.1, 1.0, 10.0], 'model__solver': ['auto', 'svd']}
    elif best_model_name == 'Lasso':
        param_grid_light = {'model__alpha': [0.0001, 0.001, 0.01], 'model__max_iter': [1000, 2000]}
    elif best_model_name in ['RandomForest', 'GradientBoosting']:
        param_grid_light = {'model__n_estimators': [50, 100], 'model__max_depth': [10, 20]}

grid_small_sample = GridSearchCV(best_model, param_grid_light, scoring='neg_root_mean_squared_error', cv=3, n_jobs=-1)
grid_small_sample.fit(X_sample, y_sample)
print(f"Küçük örnek GridSearch: {grid_small_sample.best_params_}, RMSE: {-grid_small_sample.best_score_:.4f}")

# %% XGBoost + Basit Pipeline (CPU)
simple_preprocessor = ColumnTransformer([
    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols),
    ('num', StandardScaler(), numerical_cols)
])
X_simple = simple_preprocessor.fit_transform(X)

# Test verisi boyut kontrolü
print(f"Test verisi boyutu: {test_df.shape}")
print(f"Sample submission örnek boyutu: {sample_submission.shape}")

# TÜM test verisi için tahmin yap
X_test_simple = simple_preprocessor.transform(test_df.drop(['tarih'], axis=1))

xgb_model_final = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
xgb_model_final.fit(X_simple, y)
xgb_test_preds = xgb_model_final.predict(X_test_simple)

print(f"Tahmin sayısı: {len(xgb_test_preds)}")

# Yeni bir submission dosyası oluştur - sample_submission formatına benzer şekilde
# ID sütunu oluştur
if 'id' in test_df.columns:
    ids = test_df['id'].values
else:
    # ID yoksa indeksi kullan
    ids = np.arange(test_df.shape[0])

# Yeni DataFrame oluştur
submission_all = pd.DataFrame()
submission_all['id'] = ids
submission_all[target] = xgb_test_preds

print(f"Oluşturulan submission boyutu: {submission_all.shape}")
print(f"İlk 5 satır:\n{submission_all.head()}")

# Tüm tahminleri kaydet
submission_all.to_csv('submission_xgb_all.csv', index=False)
print(f"Tüm {len(xgb_test_preds)} ürün için XGBoost tahminleri kaydedildi: submission_xgb_all.csv")


# %%
