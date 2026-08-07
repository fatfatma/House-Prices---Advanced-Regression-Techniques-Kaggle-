# House Prices - Advanced Regression Techniques

Kaggle'ın [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) yarışması için uçtan uca bir makine öğrenmesi çözümü. Iowa, Ames şehrindeki konutların 79 açıklayıcı değişkenine dayanarak satış fiyatı tahmini yapılmaktadır.

## Sonuçlar

| Versiyon | Yöntem | CV RMSE (log) | Kaggle Public Skor |
|----------|--------|----------------|---------------------|
| v1 | Gradient Boosting (elle ayarlanmış parametreler) | 0.1345 | 0.13619 |
| v2 | v1 + Feature Engineering + Outlier temizliği | 0.1185 | 0.13499 |
| v3 | Optuna ile hiperparametre optimizasyonu (5 model) + Ağırlıklı Ensemble | 0.1093 | 0.12777 |
| v4 | Stacking (Ridge + Gradient Boosting + XGBoost → Linear meta-model) | 0.1099 | **0.12619** |

Değerlendirme metriği: tahmin ve gerçek `SalePrice` değerlerinin logaritmaları arasındaki RMSE (Root Mean Squared Error).

## Proje Yol Haritası

Proje şu adımlarla ilerlemiştir:

1. **Veri Keşfi (EDA)** — `train.csv` / `test.csv` yapısının incelenmesi, `data_description.txt` ile değişkenlerin (numeric / nominal / ordinal) sınıflandırılması.
2. **Eksik Veri Analizi ve Temizliği** — Eksik değerlerin "özellik yok" (`PoolQC`, `GarageType` vb.) ve "gerçek eksik veri" (`LotFrontage`, `Electrical` vb.) olarak ayrıştırılması; mahalle bazlı medyan, mod ve `0` ile doldurma stratejileri. Data leakage'dan kaçınmak için tüm istatistikler yalnızca train setinden hesaplanıp hem train hem test'e uygulanmıştır.
3. **Ordinal Encoding** — Kalite/durum bildiren 17 kategorik değişken (`ExterQual`, `BsmtExposure`, `Functional` vb.) sıralı sayısal değerlere dönüştürülmüştür.
4. **Nominal Encoding** — Kalan 26 kategorik değişken One-Hot Encoding ile dönüştürülmüş, train/test arasındaki kategori uyuşmazlıkları `reindex` ile hizalanmıştır.
5. **Hedef Değişken Dönüşümü** — `SalePrice` sağa çarpık dağılım gösterdiği için `log1p` dönüşümü uygulanmıştır (yarışmanın metriğiyle de örtüşür).
6. **Outlier Analizi** — `GrLivArea` çok yüksek olduğu halde `SalePrice`'ı düşük olan 2 aykırı ev (bilinen Ames veri seti outlier'ları) train setinden çıkarılmıştır.
7. **Feature Engineering** — `TotalSF`, `TotalBath`, `HouseAge`, `IsRemodeled` gibi türetilmiş özellikler eklenmiştir.
8. **Baseline Modeller** — Linear Regression, Random Forest, Gradient Boosting ile ilk karşılaştırmalar yapılmıştır.
9. **Hiperparametre Optimizasyonu** — Optuna ile 5 model (Ridge, Random Forest, Gradient Boosting, XGBoost, LightGBM) üzerinde 5-fold cross-validation ile sistematik arama yapılmıştır.
10. **Ensemble & Stacking** — En iyi modellerin ağırlıklı ortalaması ve Out-of-Fold (OOF) tahminlerine dayalı stacking (meta-model) denenmiş, stacking en iyi genelleme performansını göstermiştir.

## Proje Yapısı

```
house-prices-advanced-regression-techniques/
│
├── data/
│   ├── raw/                          # Orijinal Kaggle verileri (değiştirilmez)
│   │   ├── train.csv
│   │   ├── test.csv
│   │   ├── sample_submission.csv
│   │   └── data_description.txt
│   └── processed/                    # Temizlenmiş / encode edilmiş veri
│       ├── train_processed.csv
│       └── test_processed.csv
│
├── notebooks/
│   └── notebooks/
│       ├── 01_eda.py                 # Veri keşfi, eksik veri temizliği, encoding, outlier temizliği
│       ├── 02_edaWithFE.py           # Feature engineering + baseline model karşılaştırmaları
│       ├── 03_edaWithOptuna.py       # 5 model için Optuna hiperparametre optimizasyonu + ensemble
│       └── 04_edaWithStacking.py     # Out-of-Fold tahminler + stacking (meta-model)
│
├── submissions/                      # Kaggle'a gönderilen tahmin dosyaları
│   ├── submission_v1_gb.csv
│   ├── submission_v2_gb_fe.csv
│   ├── submission_v3_optuna_ensemble.csv
│   └── submission_v4_stacking.csv
│
├── requirements.txt
└── README.md
```

## Kullanılan Yöntemler ve Kütüphaneler

- **Veri işleme:** pandas, numpy
- **Görselleştirme:** matplotlib, seaborn
- **Modelleme:** scikit-learn (Ridge, Random Forest, Gradient Boosting), XGBoost, LightGBM
- **Hiperparametre Optimizasyonu:** Optuna (Bayesian optimizasyon, 5-fold cross-validation ile)

## Nasıl Çalıştırılır

```bash
pip install -r requirements.txt
```

Ardından script'ler sırasıyla çalıştırılmalıdır (her biri bir öncekinin ürettiği veriye bağımlıdır):

```bash
python notebooks/notebooks/01_eda.py
python notebooks/notebooks/02_edaWithFE.py
python notebooks/notebooks/03_edaWithOptuna.py
python notebooks/notebooks/04_edaWithStacking.py
```

`01_eda.py` çalıştırıldığında `data/processed/` klasörüne temizlenmiş veri kaydedilir; sonraki script'ler bu işlenmiş veriyi okuyarak devam eder.

## Öne Çıkan Öğrenimler

- **Outlier temizliği**, model seçiminden bağımsız olarak genel veri kalitesini iyileştirdi; özellikle ağaç tabanlı modellerin performansında belirgin fark yarattı.
- **Basit + regularize edilmiş bir model (Ridge)**, karmaşık ağaç modelleriyle (Random Forest, Gradient Boosting) yarışacak kadar güçlü çıktı — yüksek boyutlu ama nispeten doğrusal ilişkiler taşıyan bu veri setinde regularization'ın gücünü gösterdi.
- **Ensemble/stacking'de çeşitlilik, model sayısından daha önemli**: Ridge (doğrusal) + XGBoost (ağaç tabanlı) kombinasyonu, benzer mantıkla çalışan Gradient Boosting'i de ekleyen kombinasyonlardan daha iyi performans gösterdi.
- **Cross-validation skoru ile Kaggle Public skoru arasında sistematik bir fark** gözlemlendi (~0.01-0.02), bu da public leaderboard'un test setinin yalnızca bir alt kümesini yansıttığının ve tek bir validation ölçümüne aşırı güvenilmemesi gerektiğinin bir hatırlatıcısıdır.

## Veri Seti Hakkında

[Ames Housing veri seti](https://jse.amstat.org/v19n3/decock.pdf), Dean De Cock tarafından veri bilimi eğitimi amacıyla derlenmiştir.
