import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# =============================================================================
# Paths
# NOTE: The script looks for full_dataset_cleaned.csv in the workspace folder.
#       Update INPUT_CSV if your file has a different name or location.
# =============================================================================
INPUT_CSV          = "/Users/boyuedong/Desktop/new3:11/full_dataset_cleaned.csv"
AMAZON_ONLY_CSV    = "/Users/boyuedong/Desktop/new3:11/amazon_only.csv"
AMAZON_ENRICHED_CSV = "/Users/boyuedong/Desktop/new3:11/amazon_only_enriched.csv"
RESULTS_CSV        = "/Users/boyuedong/Desktop/new3:11/amazon_ensemble_results.csv"

# =============================================================================
# 1. Load cleaned dataset and filter Amazon rows
# =============================================================================
print("=" * 60)
print("Step 1: Loading dataset …")
df = pd.read_csv(INPUT_CSV)
print(f"  Full dataset: {len(df):,} rows, {df.shape[1]} columns")

amazon = df[df["STOCK"].astype(str).str.strip().str.lower() == "amazon"].copy()
print(f"  Amazon rows found: {len(amazon)}")

# Parse dates – handle both DD/MM/YYYY and YYYY-MM-DD gracefully
amazon["DATE"] = pd.to_datetime(amazon["DATE"], dayfirst=True, errors="coerce")
amazon = amazon.dropna(subset=["DATE"])

for col in ["LAST_PRICE", "PX_VOLUME", "VOLATILITY_10D", "VOLATILITY_30D",
            "LSTM_POLARITY", "TEXTBLOB_POLARITY"]:
    if col in amazon.columns:
        amazon[col] = pd.to_numeric(amazon[col], errors="coerce")

amazon = (
    amazon
    .sort_values("DATE")
    .drop_duplicates(subset=["DATE"], keep="last")
    .reset_index(drop=True)
)
print(f"  Amazon rows after dedup & sort: {len(amazon)}")
print(f"  Date range: {amazon['DATE'].min().date()} → {amazon['DATE'].max().date()}")

amazon.to_csv(AMAZON_ONLY_CSV, index=False)
print(f"  Saved → {AMAZON_ONLY_CSV}")

# =============================================================================
# 2. Download AMZN daily data from Yahoo Finance and merge
# =============================================================================
print()
print("=" * 60)
print("Step 2: Downloading AMZN data from Yahoo Finance …")

start_date = amazon["DATE"].min().strftime("%Y-%m-%d")
end_date   = (amazon["DATE"].max() + pd.Timedelta(days=5)).strftime("%Y-%m-%d")

yf_raw = yf.download("AMZN", start=start_date, end=end_date,
                     auto_adjust=False, progress=False)
yf_raw = yf_raw.reset_index()

# Flatten MultiIndex columns if present
if isinstance(yf_raw.columns, pd.MultiIndex):
    yf_raw.columns = [c[0] if isinstance(c, tuple) else c for c in yf_raw.columns]

yf_raw = yf_raw.rename(columns={"Date": "DATE"})
yf_raw["DATE"] = pd.to_datetime(yf_raw["DATE"])
yf_raw = yf_raw.sort_values("DATE").reset_index(drop=True)

yf_cols = {
    "Open":   "YF_OPEN",
    "High":   "YF_HIGH",
    "Low":    "YF_LOW",
    "Close":  "YF_CLOSE",
    "Volume": "YF_VOLUME",
}
yf_raw = yf_raw.rename(columns=yf_cols)
keep = ["DATE"] + [v for v in yf_cols.values() if v in yf_raw.columns]
yf_raw = yf_raw[keep]

print(f"  Downloaded {len(yf_raw)} YF trading days")

# Use merge_asof so sparse monthly dates match the nearest prior trading day
amazon_sorted = amazon.sort_values("DATE").reset_index(drop=True)
amazon = pd.merge_asof(amazon_sorted, yf_raw, on="DATE", direction="backward")
amazon = amazon.sort_values("DATE").reset_index(drop=True)

yf_available = amazon["YF_CLOSE"].notna().sum()
print(f"  YF data matched to {yf_available}/{len(amazon)} Amazon rows")

# =============================================================================
# 3. Engineer classification features
# =============================================================================
print()
print("=" * 60)
print("Step 3: Engineering features …")

def pct(a, b):
    """Safe percentage change (a - b) / b, returns NaN when b == 0."""
    return np.where(b != 0, (a - b) / b, np.nan)


df_feat = amazon.copy()

# --- Price returns from LAST_PRICE ---
lp = df_feat["LAST_PRICE"]
df_feat["price_return_1"] = lp.pct_change(1)
df_feat["price_return_3"] = lp.pct_change(3)
df_feat["price_return_5"] = lp.pct_change(5)

# --- YF close returns ---
yfc = df_feat["YF_CLOSE"]
df_feat["yf_return_1"] = yfc.pct_change(1)
df_feat["yf_return_3"] = yfc.pct_change(3)
df_feat["yf_return_5"] = yfc.pct_change(5)

# --- Moving averages ---
for w in [3, 5, 10]:
    df_feat[f"price_ma_{w}"]    = lp.rolling(w).mean()
    df_feat[f"yf_close_ma_{w}"] = yfc.rolling(w).mean()

# --- Price vs its moving averages ---
for w in [3, 5, 10]:
    df_feat[f"price_vs_ma_{w}"]    = pct(lp.values, df_feat[f"price_ma_{w}"].values)
    df_feat[f"yf_close_vs_ma_{w}"] = pct(yfc.values, df_feat[f"yf_close_ma_{w}"].values)

# --- Volume features ---
vol = df_feat["PX_VOLUME"]
df_feat["volume_change_1"] = vol.pct_change(1)

yfvol = df_feat["YF_VOLUME"]
df_feat["yf_volume_change_1"] = yfvol.pct_change(1)

vol_roll_mean = yfvol.rolling(10).mean()
df_feat["volume_shock"] = pct(yfvol.values, vol_roll_mean.values)

# --- Volatility spread ---
df_feat["vol_spread"] = df_feat["VOLATILITY_10D"] - df_feat["VOLATILITY_30D"]

# --- Sentiment features ---
df_feat["sentiment_avg"]  = (df_feat["LSTM_POLARITY"] + df_feat["TEXTBLOB_POLARITY"]) / 2
df_feat["sentiment_diff"] = df_feat["LSTM_POLARITY"] - df_feat["TEXTBLOB_POLARITY"]

# --- YF candle features ---
df_feat["hl_range"]  = pct(
    (df_feat["YF_HIGH"] - df_feat["YF_LOW"]).values,
    df_feat["YF_CLOSE"].values
)
df_feat["oc_change"] = pct(
    (df_feat["YF_CLOSE"] - df_feat["YF_OPEN"]).values,
    df_feat["YF_OPEN"].values
)

# --- Rolling volatility (std of YF close returns over 10 periods) ---
df_feat["rolling_vol_10"] = yfc.pct_change().rolling(10).std()

# --- RSI (14-period, using YF close) ---
def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

df_feat["rsi_14"] = compute_rsi(yfc)

print(f"  Engineered {df_feat.shape[1] - amazon.shape[1]} new feature columns")

# =============================================================================
# 4. Build classification target
#    Target for row i = return from row i → row i+1 (next Amazon observation)
# =============================================================================
print()
print("=" * 60)
print("Step 4: Building classification target …")

df_feat["NEXT_PRICE"]  = df_feat["LAST_PRICE"].shift(-1)
df_feat["NEXT_RETURN"] = pct(df_feat["NEXT_PRICE"].values, df_feat["LAST_PRICE"].values)

def classify_return(r):
    if pd.isna(r):
        return np.nan
    if r >= 0.02:
        return "must buy"
    elif r >= 0.0:
        return "maybe buy"
    elif r > -0.02:
        return "don't buy"
    else:
        return "definitely don't buy"

df_feat["TARGET_LABEL"] = df_feat["NEXT_RETURN"].apply(classify_return)

# Drop last row (no next price) and any row with NaN target
df_feat = df_feat.dropna(subset=["TARGET_LABEL"]).reset_index(drop=True)
print(f"  Label distribution:\n{df_feat['TARGET_LABEL'].value_counts().to_string()}")

# =============================================================================
# 5. Define feature columns and drop rows with NaN features
# =============================================================================
feature_cols = [
    "LAST_PRICE", "PX_VOLUME", "VOLATILITY_10D", "VOLATILITY_30D",
    "LSTM_POLARITY", "TEXTBLOB_POLARITY",
    "YF_OPEN", "YF_HIGH", "YF_LOW", "YF_CLOSE", "YF_VOLUME",
    "price_return_1", "price_return_3", "price_return_5",
    "yf_return_1", "yf_return_3", "yf_return_5",
    "price_ma_3", "price_ma_5", "price_ma_10",
    "yf_close_ma_3", "yf_close_ma_5", "yf_close_ma_10",
    "price_vs_ma_3", "price_vs_ma_5", "price_vs_ma_10",
    "yf_close_vs_ma_3", "yf_close_vs_ma_5", "yf_close_vs_ma_10",
    "volume_change_1", "yf_volume_change_1", "volume_shock",
    "vol_spread", "sentiment_avg", "sentiment_diff",
    "hl_range", "oc_change",
    "rolling_vol_10", "rsi_14",
]
# Keep only columns that actually exist in df_feat
feature_cols = [c for c in feature_cols if c in df_feat.columns]

df_model = df_feat.dropna(subset=feature_cols).reset_index(drop=True)
print(f"\n  Rows after dropping NaN features: {len(df_model)}")
print(f"  Features used: {len(feature_cols)}")

# Save enriched file (all rows including NaN-dropped ones)
df_feat.to_csv(AMAZON_ENRICHED_CSV, index=False)
print(f"  Saved enriched CSV → {AMAZON_ENRICHED_CSV}")

# =============================================================================
# 6. Train / test split by year  (2017 → train, 2018 → test)
# =============================================================================
print()
print("=" * 60)
print("Step 5: Train/test split by year …")

train_df = df_model[df_model["DATE"].dt.year == 2017].copy()
test_df  = df_model[df_model["DATE"].dt.year == 2018].copy()

print(f"  Train (2017): {len(train_df)} rows")
print(f"  Test  (2018): {len(test_df)} rows")

# Safety: if 2017 is empty, fall back to chronological 80/20 split
if len(train_df) == 0:
    print("  WARNING: No 2017 rows after feature engineering.")
    print("  Falling back to chronological 80/20 split.")
    split_idx = int(len(df_model) * 0.8)
    train_df = df_model.iloc[:split_idx].copy()
    test_df  = df_model.iloc[split_idx:].copy()
    print(f"  Train: {len(train_df)} rows  ({train_df['DATE'].min().date()} → {train_df['DATE'].max().date()})")
    print(f"  Test:  {len(test_df)} rows  ({test_df['DATE'].min().date()} → {test_df['DATE'].max().date()})")

if len(test_df) == 0:
    raise ValueError("Test set is empty after split.")

# =============================================================================
# 7. Scale features with RobustScaler
# =============================================================================
print()
print("=" * 60)
print("Step 6: Scaling with RobustScaler …")

scaler = RobustScaler()
X_train = scaler.fit_transform(train_df[feature_cols])
X_test  = scaler.transform(test_df[feature_cols])

# Encode string labels → integers for XGBoost
label_order   = ["definitely don't buy", "don't buy", "maybe buy", "must buy"]
label_to_int  = {lbl: i for i, lbl in enumerate(label_order)}
int_to_label  = {i: lbl for lbl, i in label_to_int.items()}

y_train_str = train_df["TARGET_LABEL"].values
y_test_str  = test_df["TARGET_LABEL"].values

y_train_int = np.array([label_to_int[l] for l in y_train_str], dtype=np.int32)
y_test_int  = np.array([label_to_int[l] for l in y_test_str],  dtype=np.int32)

# If any class is missing from training set, warn
present_classes = np.unique(y_train_int)
n_classes = len(present_classes)
print(f"  Classes present in train: {[int_to_label[i] for i in present_classes]}")

# =============================================================================
# 8. Train XGBoost and LightGBM
# =============================================================================
print()
print("=" * 60)
print("Step 7: Training models …")

# Re-encode labels to be contiguous from 0 if some classes are missing in train
le = LabelEncoder()
le.fit(y_train_int)
y_train_enc = le.transform(y_train_int)
n_enc_classes = len(le.classes_)

lgbm_model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=1,       # allow small leaf sizes (important with few rows)
    random_state=42,
    class_weight="balanced",
    objective="multiclass",
    verbosity=-1,
)

xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=42,
    verbosity=0,
)

print("  Training LightGBM …")
lgbm_model.fit(X_train, y_train_enc)

print("  Training XGBoost …")
xgb_model.fit(X_train, y_train_enc)

# =============================================================================
# 9. Manual soft voting (avoid VotingClassifier shape mismatches)
# =============================================================================
print()
print("=" * 60)
print("Step 8: Predicting with soft-vote ensemble …")

def safe_proba(model, X, n_total_classes, encoder):
    """Return probability matrix of shape (n_samples, n_total_classes).
    Handles the case where the model was only trained on a subset of classes."""
    raw  = model.predict_proba(X)           # shape: (n, n_seen_classes)
    out  = np.zeros((X.shape[0], n_total_classes))
    for enc_idx, orig_label in enumerate(encoder.classes_):
        out[:, orig_label] = raw[:, enc_idx]
    return out

n_all = len(label_order)  # 4

proba_lgbm = safe_proba(lgbm_model, X_test, n_all, le)
proba_xgb  = safe_proba(xgb_model,  X_test, n_all, le)

proba_avg    = (proba_lgbm + proba_xgb) / 2
y_pred_int   = np.argmax(proba_avg, axis=1)
y_pred_label = np.array([int_to_label[i] for i in y_pred_int])

# =============================================================================
# 10. Evaluation
# =============================================================================
print()
print("=" * 60)
print("Step 9: Evaluation results")
print("=" * 60)

accuracy     = accuracy_score(y_test_str, y_pred_label)
macro_f1     = f1_score(y_test_str, y_pred_label, average="macro", zero_division=0)
macro_recall = recall_score(y_test_str, y_pred_label, average="macro", zero_division=0)

print(f"\nAccuracy     : {accuracy:.4f}")
print(f"Macro F1     : {macro_f1:.4f}")
print(f"Macro Recall : {macro_recall:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_str, y_pred_label, digits=4, zero_division=0))

# =============================================================================
# 11. Feature importances from LightGBM
# =============================================================================
print("=" * 60)
print("LightGBM Feature Importances (top 20):")
importances = lgbm_model.feature_importances_
fi_df = pd.DataFrame({
    "feature":    feature_cols,
    "importance": importances,
}).sort_values("importance", ascending=False).head(20)
print(fi_df.to_string(index=False))

# =============================================================================
# 12. Save results CSV
# =============================================================================
print()
print("=" * 60)
print("Step 10: Saving results …")

results = test_df[["DATE", "LAST_PRICE", "NEXT_PRICE", "NEXT_RETURN", "TARGET_LABEL"]].copy()
results["PREDICTED_LABEL"] = y_pred_label
results["CORRECT"] = results["TARGET_LABEL"] == results["PREDICTED_LABEL"]

results.to_csv(RESULTS_CSV, index=False)
print(f"  Saved results → {RESULTS_CSV}")
print()
print("Done.")
