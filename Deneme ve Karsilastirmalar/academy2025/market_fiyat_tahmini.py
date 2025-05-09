# %% Gerekli kütüphanelerin yüklenmesi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# %% Veri setlerinin okunması
# Eğitim verisinin okunması
train_df = pd.read_csv('train.csv')
print(f"Eğitim veri seti boyutu: {train_df.shape}")

# Test verisinin okunması
test_df = pd.read_csv('testFeatures.csv')
print(f"Test veri seti boyutu: {test_df.shape}")

# Örnek submission formatının okunması
sample_submission = pd.read_csv('sample_submission.csv')
print(f"Örnek submission formatı: {sample_submission.shape}")

# %% Veri keşfi ve incelemesi
# Eğitim verisinin ilk satırlarını görüntüleme
print("Eğitim veri setinin ilk satırları:")
print(train_df.head())

# Veri setinin genel bilgilerinin görüntülenmesi
print("\nEğitim veri seti bilgileri:")
train_df.info()

# İstatistiksel özet
print("\nİstatistiksel özet:")
print(train_df.describe())

# Eksik değerlerin kontrolü
print("\nEksik değer sayıları:")
print(train_df.isnull().sum())

# Test verisinin incelenmesi
print("\nTest veri setinin ilk satırları:")
print(test_df.head())

# Test verisi eksik değer kontrolü
print("\nTest verisi eksik değer sayıları:")
print(test_df.isnull().sum())

# %% Veri temizleme ve ön işleme
# Tarih sütununun datetime formatına dönüştürülmesi
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

# %% Kategorik ve sayısal değişkenlerin ayrılması
# Hedef değişken
target = 'ürün fiyatı'

# Kategorik ve sayısal özelliklerin belirlenmesi
categorical_cols = train_df.select_dtypes(include=['object', 'category']).columns.tolist()
if 'tarih' in categorical_cols:
    categorical_cols.remove('tarih')  # Tarih zaten işlendi

numerical_cols = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if target in numerical_cols:
    numerical_cols.remove(target)  # Hedef değişkeni çıkaralım

print(f"Kategorik değişkenler: {categorical_cols}")
print(f"Sayısal değişkenler: {numerical_cols}")

# %% Görselleştirmeler
# Hedef değişkenin dağılımı
plt.figure(figsize=(10, 6))
sns.histplot(train_df[target], kde=True)
plt.title('Ürün Fiyatı Dağılımı')
plt.show()

# Kategorilere göre ortalama fiyatlar
if 'ürün kategorisi' in train_df.columns:
    plt.figure(figsize=(12, 8))
    train_df.groupby('ürün kategorisi')[target].mean().sort_values().plot(kind='barh')
    plt.title('Kategori Bazında Ortalama Ürün Fiyatları')
    plt.show()

# %% Veri hazırlama için pipeline oluşturma
# Eksik değerler için imputer, kategorik değişkenler için encoder
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_cols),
        ('num', numerical_transformer, numerical_cols)
    ]
)

# %% Modellerin hazırlanması ve eğitilmesi
# Eğitim ve doğrulama veri setlerinin ayrılması
X = train_df.drop([target, 'tarih'] if 'tarih' in train_df.columns else [target], axis=1)
y = train_df[target]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model listesi
models = {
    'Ridge': Pipeline([
        ('preprocessor', preprocessor),
        ('model', Ridge(alpha=1.0))
    ]),
    'Lasso': Pipeline([
        ('preprocessor', preprocessor),
        ('model', Lasso(alpha=0.001))
    ]),
    'RandomForest': Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
}

# Model değerlendirme
results = {}
for name, model in models.items():
    print(f"{name} modelinin eğitimi başlıyor...")
    model.fit(X_train, y_train)
    
    # Doğrulama seti üzerindeki performans
    val_pred = model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    
    # Cross-validation ile performans
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')
    cv_rmse = -np.mean(cv_scores)
    
    results[name] = {'Validation RMSE': val_rmse, 'CV RMSE': cv_rmse}
    print(f"{name} - Validation RMSE: {val_rmse:.4f}, CV RMSE: {cv_rmse:.4f}")

# %% En iyi modelin seçilmesi
best_model_name = min(results, key=lambda x: results[x]['CV RMSE'])
best_model = models[best_model_name]
print(f"En iyi model: {best_model_name} (CV RMSE: {results[best_model_name]['CV RMSE']:.4f})")

# %% Hyperparameter optimizasyonu
if best_model_name == 'Ridge':
    param_grid = {
        'model__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
    }
elif best_model_name == 'Lasso':
    param_grid = {
        'model__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0]
    }
elif best_model_name == 'RandomForest':
    param_grid = {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [None, 10, 20, 30]
    }

print(f"{best_model_name} için hyperparameter optimizasyonu yapılıyor...")
grid_search = GridSearchCV(
    best_model,
    param_grid,
    scoring='neg_root_mean_squared_error',
    cv=5,
    n_jobs=-1
)
grid_search.fit(X, y)

print(f"En iyi parametreler: {grid_search.best_params_}")
print(f"En iyi RMSE: {-grid_search.best_score_:.4f}")

# %% Tüm veri seti ile en iyi modelin eğitilmesi
optimized_model = grid_search.best_estimator_
optimized_model.fit(X, y)

# %% Test verisi için tahminlerin yapılması
# Test verisini hazırlama
X_test = test_df.drop(['tarih'] if 'tarih' in test_df.columns else [], axis=1)

# Tahminlerin elde edilmesi
test_predictions = optimized_model.predict(X_test)

# %% Submission dosyasının hazırlanması
submission = sample_submission.copy()
submission[target] = test_predictions

# Sonuçların kaydedilmesi
submission.to_csv('submission.csv', index=False)
print("Submission dosyası hazırlandı: submission.csv")

# %% Özellik önemlerinin incelenmesi (uygunsa)
if best_model_name in ['RandomForest']:
    if best_model_name == 'RandomForest':
        feature_importances = optimized_model.named_steps['model'].feature_importances_
    else:  # XGBoost
        feature_importances = optimized_model.named_steps['model'].feature_importances_
    
    # Preprocessing işlemi sonrası özellik isimlerini alıyoruz
    preprocessor = optimized_model.named_steps['preprocessor']
    
    # Kategorik kolonların one-hot encoding sonrası isimleri
    cat_features = preprocessor.transformers_[0][1].named_steps['encoder'].get_feature_names_out(categorical_cols)
    
    # Tüm özellik isimleri
    feature_names = np.concatenate([cat_features, np.array(numerical_cols)])
    
    # Özellik önemleri
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)
    
    # Görselleştirme
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20))
    plt.title(f'En Önemli 20 Özellik ({best_model_name})')
    plt.tight_layout()
    plt.show()

# %% XGBoost ve pipeline sorunu için çözüm seçenekleri

# Seçenek 1: XGBoost modelini pipeline dışında kullanmak
# models kısmını şu şekilde değiştirin:
models = {
    'Ridge': Pipeline([
        ('preprocessor', preprocessor),
        ('model', Ridge(alpha=1.0))
    ]),
    'Lasso': Pipeline([
        ('preprocessor', preprocessor),
        ('model', Lasso(alpha=0.001))
    ]),
    'RandomForest': Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
}

# XGBoost'u ayrı işleyin
X_preprocessed = preprocessor.fit_transform(X_train)
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_preprocessed, y_train)
X_val_preprocessed = preprocessor.transform(X_val)
val_pred_xgb = xgb_model.predict(X_val_preprocessed)
val_rmse_xgb = np.sqrt(mean_squared_error(y_val, val_pred_xgb))
print(f"XGBoost - Validation RMSE: {val_rmse_xgb:.4f}")

# Seçenek 2: sklearn'in kendi GradientBoostingRegressor'ını kullanmak
from sklearn.ensemble import GradientBoostingRegressor
models['GradientBoosting'] = Pipeline([
    ('preprocessor', preprocessor),
    ('model', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
])

# Seçenek 3: Kütüphane versiyonlarını güncellemek
# Terminal'de:
# pip install --upgrade scikit-learn xgboost
