# YZTA Datathon 2024 - Grup 103 Projesi

## Proje Özeti

Bu proje, YZTA Datathon 2024 yarışması kapsamında Grup 103 tarafından geliştirilen ürün fiyat tahmin modelini içermektedir. Modelimiz, çeşitli ürünlerin fiyatlarını tahmin etmek için derin öğrenme teknikleri kullanmakta ve USD/TRY döviz kuru verileri gibi dış etkenleri de hesaba katmaktadır.

## Veri Setleri

Projede kullanılan temel veri setleri:

- **train.csv**: Eğitim için kullanılan ana veri seti. Ürün özellikleri, kategorileri, market bilgileri ve fiyat bilgilerini içerir.
- **testFeatures.csv**: Test veri seti, eğitilen modelin tahmin yapması için kullanılır.
- **usd_clean.csv**: 2019-2025 yılları arasındaki USD/TRY günlük döviz kuru verilerini içerir. Bu veri seti, ürün fiyatlarını tahmin ederken önemli bir dış faktör olarak kullanılmıştır.
- **sample_submission.csv**: Yarışma tarafından sağlanan örnek gönderim formatını gösterir.

## Metodoloji

### Veri Ön İşleme

- Tarih bilgilerinden yeni özellikler türetildi (yıl, ay, haftanın günü)
- Kategorik değişkenler (ürün, ürün kategorisi, market, şehir vb.) sayısallaştırıldı
- Sayısal değerlere logaritmik dönüşüm uygulandı
- USD/TRY döviz kuru verileri ana veri setiyle birleştirildi
- Veriler, eğitim ve doğrulama setlerine bölündü
- Özellikler standartlaştırıldı (StandardScaler)

### Model Mimarisi

Derin öğrenme modeli olarak Keras/TensorFlow ile yapay sinir ağı kullanıldı:

- Giriş katmanı: Ürün özellikleri, kategorik bilgiler ve döviz kuru verilerini alır
- İki gizli katman: Dinamik boyutlarda (32-128 nöron arasında)
- Aktivasyon fonksiyonu: ReLU
- Çıkış katmanı: Tek nöron (ürün fiyatı tahmini)

### Hiperparametre Optimizasyonu

Model performansını artırmak için Keras Tuner kullanıldı:

- Gizli katmanlardaki nöron sayısı
- Optimizer seçimi (adam, rmsprop)
- Erken durdurma mekanizması (overfitting önleme)

## Sonuçlar

En iyi modelimiz, USD/TRY döviz kuru verilerini de dahil ederek oluşturulmuş ve bunu sonuç dosyasına yansıtılmıştır. Elde edilen sonuçlar `en-iyi-sonuc-4.6025400670.csv` dosyasında saklanmaktadır. Dosya adından anlaşılacağı üzere, modelin hata skoru (RMSE) 4.60254 olarak hesaplanmıştır.

## Kurulum ve Kullanım

### Gereksinimler

Projeyi çalıştırmak için gerekli kütüphaneler `requirements.txt` dosyasında belirtilmiştir. Gerekli kütüphaneleri yüklemek için:

```bash
pip install -r requirements.txt
```

### Kullanım Talimatları

Projeyi kendi ortamınızda çalıştırmak için:

1. Bu repository'yi klonlayın:
   ```bash
   git clone https://github.com/kullanici-adi/yzta-datathon-grup103.git
   cd yzta-datathon-grup103
   ```

2. Gereksinimleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```

3. Veri setlerini temin edin:
   - train.csv
   - testFeatures.csv
   - usd_clean.csv

4. Jupyter Notebook dosyasını (`grup-103.ipynb`) çalıştırın:
   ```bash
   jupyter notebook grup-103.ipynb
   ```

5. Notebook içerisindeki hücreleri sırasıyla çalıştırarak modelin eğitimini ve tahminleri gerçekleştirin.

## Model Özellikleri

- **Özellik Mühendisliği**: Tarih verileri işlenerek yeni özellikler yaratıldı
- **Dış Faktörler**: USD/TRY döviz kuru verileri modele dahil edildi
- **Log Dönüşümü**: Fiyat verileri için logaritmik dönüşüm uygulandı
- **Hiperparametre Optimizasyonu**: Keras Tuner ile optimal model mimarisi bulundu
- **Erken Durdurma**: Aşırı uyumu (overfitting) önlemek için EarlyStopping kullanıldı

## Dosya Yapısı

```
yzta-datathon-grup103/
├── README.md               # Proje açıklaması
├── LICENSE                 # MIT Lisansı
├── requirements.txt        # Gerekli kütüphaneler
├── grup-103.ipynb          # Ana notebook dosyası
├── usd_clean.csv           # USD/TRY kur verileri
├── sample_submission.csv   # Örnek submission dosyası
└── en-iyi-sonuc-4.6025400670.csv # En iyi tahmin sonuçları
```

## Grup Üyeleri

YZTA Datathon 2024 Grup 103 ekibi üyeleri:

- Mehmet Said Hüseyinoğlu
- Kadir Akyürek
- Eda Nur Işık
- Medine

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Daha fazla bilgi için [LICENSE](LICENSE) dosyasına bakınız.

---

**Not**: Bu README dosyası, YZTA Datathon 2024 kapsamında geliştirilen modelin GitHub'da paylaşılması için oluşturulmuştur. Projenin tüm hakları ve sorumlulukları Grup 103'e aittir.