# ============================================================================
# KÜTÜPHANE İMPORTLARI
# ============================================================================
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
import keras_tuner as kt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

# ============================================================================
# VERİ YÜKLEME VE ÖN İŞLEME
# ============================================================================
# Ana veri setlerini yükleme
train0 = pd.read_csv('/kaggle/input/academy2025/train.csv')
test0 = pd.read_csv('/kaggle/input/academy2025/testFeatures.csv')
# USD/TRY kur verilerini yükleme - fiyat tahminlerini iyileştirmek için ek özellik olarak kullanılacak
usd_df = pd.read_csv('/kaggle/input/afsfewwerwerwer/usd_clean.csv')

# ============================================================================
# ZAMAN ÖZELLİKLERİNİN ÇIKARILMASI
# ============================================================================
# Tarih sütunlarını datetime formatına dönüştürme
train0['tarih'] = pd.to_datetime(train0['tarih'])
test0['tarih'] = pd.to_datetime(test0['tarih'])
usd_df['tarih'] = pd.to_datetime(usd_df['tarih'], format='%Y-%m-%d')

# Tarihten yeni özellikler türetme (yıl, ay, haftanın günü)
train0["yıl"] = train0["tarih"].dt.year
train0["ay"] = train0["tarih"].dt.month
test0["yıl"] = test0["tarih"].dt.year
test0["ay"] = test0["tarih"].dt.month
train0["haftanın_günü"] = train0["tarih"].dt.dayofweek
test0["haftanın_günü"] = test0["tarih"].dt.dayofweek

# ============================================================================
# KATEGORİK VERİLERİN SAYISALLAŞTIRILMASI
# ============================================================================
# Kategorik değişkenleri LabelEncoder ile sayısal değerlere dönüştürme
# Not: Train ve test verisi birlikte encode ediliyor ki aynı kategoriler aynı değerleri alsın
categorical_cols = ["ürün", "ürün kategorisi", "ürün üretim yeri", "market", "şehir"]
for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([train0[col], test0[col]])
    le.fit(combined)
    train0[col] = le.transform(train0[col])
    test0[col] = le.transform(test0[col])

# ============================================================================
# VERİ DÖNÜŞÜMÜ VE ÖZELLİK MÜHENDİSLİĞİ
# ============================================================================
# Sayısal değerlere logaritmik dönüşüm uygulama - aşırı değerlerin etkisini azaltmak için
train0['ürün fiyatı'] = np.log1p(train0['ürün fiyatı'])
train0['ürün besin değeri'] = np.log1p(train0['ürün besin değeri'])
test0['ürün besin değeri'] = np.log1p(test0['ürün besin değeri'])

# USD kur verilerini ana veri setleriyle birleştirme
train_merged = pd.merge(train0, usd_df[['tarih', 'USD/TRY']], on='tarih', how='left')
test_merged = pd.merge(test0, usd_df[['tarih', 'USD/TRY']], on='tarih', how='left')

# USD/TRY verilerini sayısal formata dönüştürme ve temizleme
train_merged['USD/TRY'] = train_merged['USD/TRY'].str.replace(',', '.').astype(float)
test_merged['USD/TRY'] = test_merged['USD/TRY'].str.replace(',', '.').astype(float)
train_merged['USD/TRY'] = train_merged['USD/TRY'].fillna(0)
test_merged['USD/TRY'] = test_merged['USD/TRY'].fillna(0)

# ============================================================================
# MODEL İÇİN VERİ HAZIRLAMA
# ============================================================================
# Modelde kullanılacak özelliklerin seçimi
features_with_usd = ["ürün", "ürün besin değeri", "ürün kategorisi", "ürün üretim yeri",
                     "market", "şehir", "yıl", "ay", "haftanın_günü", "USD/TRY"]
X_usd = train_merged[features_with_usd]
y_usd = train_merged["ürün fiyatı"]
X_test_usd = test_merged[features_with_usd]

# Veri setini eğitim ve doğrulama olarak bölme
X_train_usd, X_val_usd, y_train_usd, y_val_usd = train_test_split(X_usd, y_usd, test_size=0.2, random_state=42)

# Özellikleri ölçeklendirme - model performansını artırmak için
scaler_usd = StandardScaler()
X_train_scaled_usd = scaler_usd.fit_transform(X_train_usd)
X_val_scaled_usd = scaler_usd.transform(X_val_usd)
X_test_scaled_usd = scaler_usd.transform(X_test_usd)

# ============================================================================
# KERAS TUNER İLE MODEL OLUŞTURMA VE HİPERPARAMETRE OPTİMİZASYONU
# ============================================================================
def build_model_with_usd(hp):
    # Dinamik model mimarisi - Keras Tuner ile optimize edilecek
    model = keras.Sequential([
        layers.Dense(hp.Int('units_1', min_value=32, max_value=128, step=32), activation='relu', input_shape=[X_train_scaled_usd.shape[1]]),
        layers.Dense(hp.Int('units_2', min_value=32, max_value=128, step=32), activation='relu'),
        layers.Dense(1)
    ])
    optimizer = hp.Choice('optimizer', values=['adam', 'rmsprop'])
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
    return model

# Keras Tuner ile hiperparametre optimizasyonu
tuner_usd = kt.RandomSearch(
    build_model_with_usd,
    objective='val_loss',
    max_trials=3,
    executions_per_trial=3,
    directory='kt_dir_usd',
    project_name='academy2025_tuning_usd'
)

# Erken durdurma callback'i - overfitting'i önlemek için
early_stopping_usd = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Hiperparametre arama işlemi
tuner_usd.search(X_train_scaled_usd, y_train_usd,
                 epochs=100,
                 validation_data=(X_val_scaled_usd, y_val_usd),
                 callbacks=[early_stopping_usd],
                 verbose=1)

# En iyi modeli seç
best_model_usd = tuner_usd.get_best_models(num_models=1)[0]

# ============================================================================
# MODEL DEĞERLENDİRME VE GÖRSELLEŞTIRME
# ============================================================================
# Doğrulama seti üzerinde model performansını değerlendirme
loss_usd, mse_usd = best_model_usd.evaluate(X_val_scaled_usd, y_val_usd, verbose=0)
rmse_val_deep_learning_tuned_usd = np.sqrt(mse_usd)
print(f"\nDoğrulama Kümesi RMSE (USD Kuru Dahil): {rmse_val_deep_learning_tuned_usd}")

# Eğitim sürecini görselleştirme
history_usd = best_model_usd.fit(
    X_train_scaled_usd, y_train_usd,
    validation_data=(X_val_scaled_usd, y_val_usd),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping_usd],
    verbose=0
)
history_df_usd = pd.DataFrame(history_usd.history)
history_df_usd.loc[:, ['loss', 'val_loss']].plot()
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.title("Eğitim ve Doğrulama Kaybı (USD Kuru Dahil)")
plt.show()

# ============================================================================
# TAHMİN VE SONUÇLARIN KAYDI
# ============================================================================
# Test verisi üzerinde tahmin yapma
preds_log_deep_tuned_usd = best_model_usd.predict(X_test_scaled_usd).flatten()

# Tahminleri orijinal ölçeğe geri dönüştürme
preds_deep_tuned_usd = np.expm1(preds_log_deep_tuned_usd)
preds_deep_tuned_usd[preds_deep_tuned_usd < 0] = 0  # Negatif fiyatları sıfıra çevirme

# Sonuçları CSV dosyasına kaydetme
submission_df_deep_tuned_usd = pd.DataFrame({'id': test_merged['id'], 'ürün fiyatı': preds_deep_tuned_usd})
submission_df_deep_tuned_usd.to_csv('submission_tuned_deep_learning_usd.csv', index=False)

print("\nsubmission_tuned_deep_learning_usd.csv dosyası başarıyla oluşturuldu.")
print("\nSubmission Dosyasının İlk 10 Değeri (USD Kuru Dahil):")
print(submission_df_deep_tuned_usd.head(10))

# Oluşturulan dosyaları kontrol etme
for dirname, _, filenames in os.walk('/kaggle/working'):
    for filename in filenames:
        print(os.path.join(dirname, filename))