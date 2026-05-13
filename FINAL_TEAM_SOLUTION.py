from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

DATA_DIR = Path("/content")
TARGET = "bilissel_performans_skoru"
RNG = 42

sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams["figure.figsize"] = (10, 4)

train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test_x.csv")
print(f"Train {train.shape} | Test {test.shape}\n")

df = train.copy()
feat = df.drop(columns=["id"], errors="ignore")
print(TARGET, "özeti:")
print(feat[[TARGET]].describe().T)
miss = feat.isna().mean().sort_values(ascending=False) * 100
miss = miss[miss > 0]
print("\n=== Eksiklik (ilk 15 sütun, %) ===")
print(miss.head(15).to_string())
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df[TARGET], kde=True, ax=ax[0], color="steelblue")
ax[0].set_title("Hedef histogram + KDE")
sns.boxplot(x=df[TARGET], ax=ax[1], color="lightyellow")
ax[1].set_title("Hedef kutu grafiği")
plt.tight_layout()
plt.show()

ys = df[TARGET]
print(f"\nUç dilim yüzdesi (y≤1 veya y≥9): {( (ys<=1)|(ys>=9) ).mean()*100:.2f}%")
num_cols = feat.select_dtypes(include=[np.number]).columns.drop(TARGET, errors="ignore")
pearson = feat[num_cols.tolist() + [TARGET]].corr(numeric_only=True)[TARGET].drop(TARGET).sort_values(
    ascending=False
)

pd.set_option("display.max_rows", 30)
print("\n=== Pearson korelasyonu (hedef ile, tüm sayısallar) — üst ve alt ===")
print("En pozitif 15:")
print(pearson.head(15).round(4).to_string())
print("\nEn negatif 15:")
print(pearson.tail(15).round(4).to_string())

plt.figure(figsize=(10, max(5, len(pearson.head(22)) * 0.28)))
sns.barplot(x=pearson.head(22).values, y=pearson.head(22).index, color="#2ecc71")
plt.axvline(0, color="gray", lw=1)
plt.title("Hedefle Pearson | en güçlü 22 özellik (pozitif uç)")
plt.xlabel("Pearson")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, max(5, len(pearson.tail(22)) * 0.28)))
sns.barplot(x=pearson.tail(22).values, y=pearson.tail(22).index, color="#e74c3c")
plt.axvline(0, color="gray", lw=1)
plt.title("Hedefle Pearson | en güçlü 22 özellik (negatif uç)")
plt.xlabel("Pearson")
plt.tight_layout()
plt.show()
heatmap_cols = [
    TARGET,
    "yas",
    "vucut_kitle_indeksi",
    "rem_yuzdesi",
    "derin_uyku_yuzdesi",
    "stres_skoru",
    "gunluk_calisma_saati",
    "uyku_oncesi_ekran_suresi_dk",
    "uyku_oncesi_kafein_mg",
    "gunluk_adim_sayisi",
    "dinlenik_nabiz_bpm",
]
heatmap_cols = [c for c in heatmap_cols if c in df.columns]

cm = df[heatmap_cols].corr(numeric_only=True)
plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=False)
plt.title("Seçilmiş sayısallar + hedef Pearson matrisi")
plt.tight_layout()
plt.show()
sample_n = min(15000, len(df))
spl = df.sample(sample_n, random_state=RNG)
spear = spl[num_cols.tolist() + [TARGET]].corr(method="spearman", numeric_only=True)[TARGET].drop(
    TARGET
).sort_values(ascending=False)

print("\n=== Spearman (hedef ile, n=", sample_n, ") — üst / alt 10 ===")
print(spear.head(10).round(4).to_string())
print()
print(spear.tail(10).round(4).to_string())
plot_xy = [(c,) for c in ["stres_skoru", "derin_uyku_yuzdesi", "rem_yuzdesi"]]
plot_xy = [c[0] for c in plot_xy if c[0] in df.columns][:3]

spl2 = df.sample(min(4000, len(df)), random_state=RNG + 1)
fig, axes = plt.subplots(1, len(plot_xy), figsize=(4 * len(plot_xy), 4))
if len(plot_xy) == 1:
    axes = [axes]
for ax, col in zip(axes, plot_xy):
    ax.scatter(spl2[col], spl2[TARGET], alpha=0.25, s=8)
    ax.set_xlabel(col)
    ax.set_ylabel(TARGET)
    ax.set_title(f"{col} vs hedef")
plt.tight_layout()
plt.show()
cats = ["meslek", "ulke", "kronotip", "ruh_sagligi_durumu", "cinsiyet", "mevsim", "gun_tipi"]
cats = [c for c in cats if c in df.columns]
card = pd.DataFrame(
    {"kolon": cats, "benzersiz": [df[c].fillna("__NA__").astype(str).nunique() for c in cats]}
).sort_values("benzersiz", ascending=False)
print("\n=== Kategori benzersiz sayısı ===")
print(card.to_string(index=False))

plt.figure(figsize=(7, 3.5))
sns.barplot(data=card, x="benzersiz", y="kolon", color="#9b59b6")
plt.title("Nominal kolon karmaşıklığı (TE / CatBoost için bağlam)")
plt.tight_layout()
plt.show()

print("\n--- EDA bitti ---")

# Colab equivalent: %pip install -q lightgbm catboost pandas scikit-learn
import subprocess
import sys
subprocess.check_call([
    sys.executable, '-m', 'pip', 'install', '-q',
    'lightgbm', 'catboost', 'pandas', 'scikit-learn',
])
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
import warnings
import gc

warnings.filterwarnings('ignore')

print("=== 🚀 NİHAİ TAKIM İTTİFAK SENTEZİ (FRANKENSTEIN: ORİJİNAL VERİ + OPTUNA PARAMETRELERİ) 🚀 ===")

print("[-] FAZ 1: Veriler okunuyor ve ayrıştırılıyor...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test_x.csv')

TARGET = 'bilissel_performans_skoru'
y_train_target = train[TARGET]
test_ids = test['id']

X_train_raw = train.drop(['id', TARGET], axis=1)
X_test_raw = test.drop(['id'], axis=1)

LOWER_BOUND = y_train_target.quantile(0.0005)
UPPER_BOUND = y_train_target.quantile(0.9995)
print("[-] FAZ 1.5: Metinler ASCII formatında temizleniyor ve kırpılıyor...")

tr_harfler = "çğıöşü"
en_harfler = "cgiosu"
ceviri_tablosu = str.maketrans(tr_harfler, en_harfler)

ulke_sozlugu = {
    'netherlands': 'hollanda', 'france': 'fransa', 'germany': 'almanya',
    'england': 'ingiltere', 'uk': 'ingiltere', 'united kingdom': 'ingiltere',
    'china': 'cin', 'usa': 'amerika', 'united states': 'amerika',
    'south korea': 'guney kore', 'spain': 'ispanya', 'sweden': 'isvec'
}

cat_cols_for_cleaning = ['meslek', 'kronotip', 'ruh_sagligi_durumu', 'cinsiyet', 'ulke', 'mevsim', 'gun_tipi']

for df in [X_train_raw, X_test_raw]:
    for col in cat_cols_for_cleaning:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()
            df[col] = df[col].str.translate(ceviri_tablosu)

    if 'ulke' in df.columns:
        df['ulke'] = df['ulke'].replace(ulke_sozlugu)

    if 'gunluk_calisma_saati' in df.columns: df['gunluk_calisma_saati'] = df['gunluk_calisma_saati'].clip(lower=0, upper=20)
    if 'dinlenik_nabiz_bpm' in df.columns: df['dinlenik_nabiz_bpm'] = df['dinlenik_nabiz_bpm'].clip(lower=30, upper=150)
    if 'yas' in df.columns: df['yas'] = df['yas'].clip(lower=16, upper=90)
print("[-] FAZ 2: Eksiklik Odaklı Hibrit MICE Imputation uygulanıyor...")

num_cols = X_train_raw.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train_raw.select_dtypes(exclude=[np.number]).columns.tolist()

missing_num_cols = [col for col in num_cols if X_train_raw[col].isna().any()]

mice_imputer = IterativeImputer(max_iter=10, random_state=42, add_indicator=True)
cat_imputer = SimpleImputer(strategy='constant', fill_value='unknown')

transformers = [('cat_simp', cat_imputer, cat_cols)]
if missing_num_cols:
    transformers.append(('num_mice', mice_imputer, missing_num_cols))

preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough', verbose_feature_names_out=False)
preprocessor.set_output(transform="pandas")

X_train_clean = preprocessor.fit_transform(X_train_raw)
X_test_clean = preprocessor.transform(X_test_raw)

for col in num_cols:
    X_train_clean[col] = X_train_clean[col].astype(float)
    X_test_clean[col] = X_test_clean[col].astype(float)

for col in cat_cols:
    X_train_clean[col] = X_train_clean[col].astype(str)
    X_test_clean[col] = X_test_clean[col].astype(str)
print("[-] FAZ 3: Üç Arkadaşın Tüm Altın Özellikleri Sentezleniyor...")

def add_validated_features(df_train, df_test):
    df = pd.concat([df_train, df_test], axis=0).copy()

    df['uyku_kalitesi_skoru_v2'] = df['rem_yuzdesi'] + df['derin_uyku_yuzdesi'] - (2 * df['gecelik_uyanma_sayisi']) - (0.2 * df['uykuya_dalma_suresi_dk'])
    df['is_kronik_uykusuz'] = ((df['gecelik_uyanma_sayisi'] > 3) & (df['derin_uyku_yuzdesi'] < 15)).astype(int)
    if 'uyku_oncesi_kafein_mg' in df.columns: df['kafein_log'] = np.log1p(df['uyku_oncesi_kafein_mg'])

    df["fiziksel_enerji"] = df["gunluk_adim_sayisi"] / (df["dinlenik_nabiz_bpm"] + 1)
    df['kardiyo_stres_endeksi'] = df['dinlenik_nabiz_bpm'] * df['stres_skoru']

    df["ekran_yas_orani"] = df["uyku_oncesi_ekran_suresi_dk"] / (df["yas"] + 1)
    df["uyku_parcalanma"] = df["gecelik_uyanma_sayisi"] * df["uykuya_dalma_suresi_dk"]
    df["hafta_sonu"] = (df["gun_tipi"] == "hafta sonu").astype(int)
    df["weekend_sleep_shift"] = df["hafta_sonu"] * df["hafta_sonu_uyku_farki_saat"].fillna(0)
    df["stres_x_ekran"] = df["stres_skoru"] * df["uyku_oncesi_ekran_suresi_dk"]
    df["stres_x_kafein"] = df["stres_skoru"] * df["uyku_oncesi_kafein_mg"]
    df["bmi_yas"] = df["vucut_kitle_indeksi"] * df["yas"]

    df["meslek_kronotip"] = df["meslek"].astype(str) + "_" + df["kronotip"].astype(str)

    res_train = df.iloc[:len(df_train)].copy()
    res_test = df.iloc[len(df_train):].copy()
    return res_train, res_test

X_train_feat, X_test_feat = add_validated_features(X_train_clean, X_test_clean)
print("[-] FAZ 4: Sızıntısız OOF Target Encoding ve Standartlaştırma...")

current_num_cols = X_train_feat.select_dtypes(include=[np.number]).columns.tolist()
current_cat_cols = X_train_feat.select_dtypes(exclude=[np.number]).columns.tolist()

scaler = StandardScaler()
X_train_feat[current_num_cols] = scaler.fit_transform(X_train_feat[current_num_cols])
X_test_feat[current_num_cols] = scaler.transform(X_test_feat[current_num_cols])

X_train_le = X_train_feat.copy()
X_test_le = X_test_feat.copy()

for col in current_cat_cols:
    le = LabelEncoder()
    le.fit(list(X_train_feat[col].astype(str)) + list(X_test_feat[col].astype(str)))
    X_train_le[col] = le.transform(X_train_feat[col].astype(str))
    X_test_le[col] = le.transform(X_test_feat[col].astype(str))

# Dinamik Kolon Kontrolü (Olası hataları engeller)
target_enc_cols = [c for c in ['meslek', 'ulke', 'kronotip', 'ruh_sagligi_durumu', 'meslek_kronotip'] if c in X_train_feat.columns]
X_train_encoded = X_train_le.copy()
X_test_encoded = X_test_le.copy()

kf = KFold(n_splits=10, shuffle=True, random_state=42)

for col in target_enc_cols:
    mapping = y_train_target.groupby(X_train_feat[col]).mean()
    X_test_encoded[f'{col}_target_enc'] = X_test_encoded[col].map(mapping).fillna(y_train_target.mean())
    X_train_encoded[f'{col}_target_enc'] = np.nan

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_feat, y_train_target)):
    tr_cats = X_train_feat.iloc[tr_idx]
    tr_y = y_train_target.iloc[tr_idx]
    val_cats = X_train_feat.iloc[val_idx]

    for col in target_enc_cols:
        fold_map = tr_y.groupby(tr_cats[col]).mean()
        encoded_vals = val_cats[col].map(fold_map).fillna(tr_y.mean())
        X_train_encoded.loc[val_idx, f'{col}_target_enc'] = encoded_vals

if target_enc_cols:
    new_enc_cols = [f'{col}_target_enc' for col in target_enc_cols]
    X_train_encoded[new_enc_cols] = scaler.fit_transform(X_train_encoded[new_enc_cols])
    X_test_encoded[new_enc_cols] = scaler.transform(X_test_encoded[new_enc_cols])

if "meslek_target_enc" in X_train_encoded.columns:
    X_train_encoded["te_meslek_x_stres"] = X_train_encoded["meslek_target_enc"] * X_train_encoded["stres_skoru"]
    X_test_encoded["te_meslek_x_stres"] = X_test_encoded["meslek_target_enc"] * X_test_encoded["stres_skoru"]
print("[-] FAZ 5: Pseudo-Labeling test cevapları tahmin ediliyor...")
cat_indices = [X_train_feat.columns.get_loc(col) for col in current_cat_cols]

m_pre = CatBoostRegressor(n_estimators=1000, cat_features=cat_indices, verbose=False, random_state=42)
m_pre.fit(X_train_feat, y_train_target)
pseudo_y = m_pre.predict(X_test_feat)
print("\n=== FAZ 6: 10-Fold Dörtlü Motor Çapraz Doğrulaması Başladı ===")

oof_cat, oof_xgb, oof_lgb, oof_knn = (np.zeros(len(X_train_feat)) for _ in range(4))
test_cat, test_xgb, test_lgb, test_knn = (np.zeros(len(X_test_feat)) for _ in range(4))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_feat, y_train_target)):
    X_fold_train_cat = pd.concat([X_train_feat.iloc[train_idx], X_test_feat], axis=0)
    X_fold_train_enc = pd.concat([X_train_encoded.iloc[train_idx], X_test_encoded], axis=0)
    y_fold_train = pd.concat([y_train_target.iloc[train_idx], pd.Series(pseudo_y)], axis=0)

    m_cat = CatBoostRegressor(
        iterations=1113,
        learning_rate=0.04832,
        depth=5,
        l2_leaf_reg=8.7638,
        cat_features=cat_indices,
        verbose=False,
        random_state=42
    )
    m_cat.fit(X_fold_train_cat, y_fold_train)
    oof_cat[val_idx] = m_cat.predict(X_train_feat.iloc[val_idx])
    test_cat += m_cat.predict(X_test_feat) / kf.n_splits

    m_xgb = XGBRegressor(n_estimators=1500, learning_rate=0.03, max_depth=6, random_state=42, n_jobs=-1)
    m_xgb.fit(X_fold_train_enc, y_fold_train)
    oof_xgb[val_idx] = m_xgb.predict(X_train_encoded.iloc[val_idx])
    test_xgb += m_xgb.predict(X_test_encoded) / kf.n_splits

    m_lgb = LGBMRegressor(
        n_estimators=1064,
        learning_rate=0.02164,
        num_leaves=30,
        max_depth=8,
        subsample=0.9845,
        verbose=-1,
        random_state=42,
        n_jobs=-1
    )
    m_lgb.fit(X_fold_train_enc, y_fold_train)
    oof_lgb[val_idx] = m_lgb.predict(X_train_encoded.iloc[val_idx])
    test_lgb += m_lgb.predict(X_test_encoded) / kf.n_splits

    m_knn = KNeighborsRegressor(n_neighbors=15, weights='distance', n_jobs=-1)
    m_knn.fit(X_train_encoded.iloc[train_idx], y_train_target.iloc[train_idx])
    oof_knn[val_idx] = m_knn.predict(X_train_encoded.iloc[val_idx])
    test_knn += m_knn.predict(X_test_encoded) / kf.n_splits

    print(f" ✔ Fold {fold + 1}/10 başarıyla tamamlandı.")

    del m_cat, m_xgb, m_lgb, m_knn
    gc.collect()
print("\n=== FAZ 7: Ridge Meta-Model Optimizasyonu ve Teslimat ===")

stacked_train = np.column_stack((oof_cat, oof_xgb, oof_lgb, oof_knn))
stacked_test = np.column_stack((test_cat, test_xgb, test_lgb, test_knn))

meta_model = Ridge(alpha=10.0, random_state=42)
meta_model.fit(stacked_train, y_train_target)

oof_final_preds = meta_model.predict(stacked_train)
final_rmse = np.sqrt(mean_squared_error(y_train_target, oof_final_preds))

print(f"\n>>> DÜRÜST, SIZINTISIZ VE OPTUNA NAKİLLİ NİHAİ OOF RMSE: {final_rmse:.6f}")
print("Meta-Model Motor Ağırlıkları (CatBoost, XGBoost, LightGBM, KNN):")
print(np.round(meta_model.coef_, 4))

final_sub_preds = meta_model.predict(stacked_test)
final_sub_preds = final_sub_preds - (final_sub_preds.mean() - y_train_target.mean())
final_sub_preds = np.clip(final_sub_preds, LOWER_BOUND, UPPER_BOUND)

# Submission CSV uses full float predictions (no rounding to 2 decimals).
submission = pd.DataFrame({'id': test_ids, TARGET: final_sub_preds})
submission.to_csv('TEAM_VOLTRAN_FRANKENSTEIN_OPTUNA.csv', index=False)

print("\n🏆 DOSYA HAZIR! 'TEAM_VOLTRAN_FRANKENSTEIN_OPTUNA.csv' diske yazıldı.")
print("=" * 55)
