"""
Walk-forward 70/30 split on Amazon-only data.

Strategy:
  - Take all Amazon rows from amazon_only_enriched.csv (sorted chronologically)
  - First 70% → training
  - Last 30%  → test
  - Same stock, same frequency, no cross-stock leakage

This is the most defensible split because:
  1. Training and test data come from the same stock (no domain mismatch)
  2. Data is daily/consistent — no sparse monthly vs daily mismatch
  3. Respects time ordering — model never sees the future
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder, RobustScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_CSV   = "/Users/boyuedong/Desktop/new3:11/amazon_only_enriched.csv"
RESULTS_CSV = "/Users/boyuedong/Desktop/new3:11/amazon_walkforward_results.csv"

# ── Features ──────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "LAST_PRICE", "PX_VOLUME", "VOLATILITY_10D", "VOLATILITY_30D",
    "LSTM_POLARITY", "TEXTBLOB_POLARITY",
    "YF_OPEN", "YF_HIGH", "YF_LOW", "YF_CLOSE", "YF_VOLUME",
    "price_return_1", "price_return_3", "price_return_5",
    "yf_return_1", "yf_return_3", "yf_return_5",
    "price_ma_3", "yf_close_ma_3", "price_ma_5", "yf_close_ma_5",
    "price_ma_10", "yf_close_ma_10",
    "price_vs_ma_3", "yf_close_vs_ma_3",
    "price_vs_ma_5", "yf_close_vs_ma_5",
    "price_vs_ma_10", "yf_close_vs_ma_10",
    "volume_change_1", "yf_volume_change_1", "volume_shock",
    "vol_spread", "sentiment_avg", "sentiment_diff",
    "hl_range", "oc_change", "rolling_vol_10", "rsi_14",
]

TARGET_COL = "TARGET_LABEL"


def main():
    # ── Load data ─────────────────────────────────────────────────────────────
    df = pd.read_csv(INPUT_CSV, parse_dates=["DATE"])
    df = df.sort_values("DATE").reset_index(drop=True)

    # Keep only columns we need
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    df = df[["DATE", "LAST_PRICE", "NEXT_PRICE", "NEXT_RETURN", TARGET_COL] + available_features].copy()

    # Drop rows with missing target or all-NaN features
    df = df.dropna(subset=[TARGET_COL])
    for col in available_features:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    print(f"Total Amazon rows: {len(df)}")
    print(f"Date range: {df['DATE'].min().date()} → {df['DATE'].max().date()}")
    print(f"Label distribution:\n{df[TARGET_COL].value_counts()}\n")

    # ── 70 / 30 chronological split ───────────────────────────────────────────
    split_idx = int(len(df) * 0.70)
    train_df  = df.iloc[:split_idx].copy()
    test_df   = df.iloc[split_idx:].copy()

    print(f"Train: {len(train_df)} rows  ({train_df['DATE'].min().date()} → {train_df['DATE'].max().date()})")
    print(f"Test : {len(test_df)} rows   ({test_df['DATE'].min().date()} → {test_df['DATE'].max().date()})\n")

    # ── Encode labels ─────────────────────────────────────────────────────────
    le = LabelEncoder()
    le.fit(df[TARGET_COL])
    y_train = le.transform(train_df[TARGET_COL])
    y_test  = le.transform(test_df[TARGET_COL])

    # ── Scale features ────────────────────────────────────────────────────────
    scaler = RobustScaler()

    X_train = train_df[available_features].copy()
    X_test  = test_df[available_features].copy()

    # Fill NaN with column median from training set
    train_medians = X_train.median()
    X_train = X_train.fillna(train_medians)
    X_test  = X_test.fillna(train_medians)

    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    n_classes = len(le.classes_)
    print(f"Classes: {list(le.classes_)}\n")

    # ── Train models ──────────────────────────────────────────────────────────
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0,
    )

    lgbm = LGBMClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )

    xgb.fit(X_train, y_train)
    lgbm.fit(X_train, y_train)

    # ── Soft voting ───────────────────────────────────────────────────────────
    def safe_proba(model, X, n_classes):
        """Pad probability matrix to n_classes columns if needed."""
        p = model.predict_proba(X)
        if p.shape[1] == n_classes:
            return p
        full = np.zeros((len(X), n_classes))
        for i, cls in enumerate(model.classes_):
            full[:, cls] = p[:, i]
        return full

    proba = (safe_proba(xgb, X_test, n_classes) + safe_proba(lgbm, X_test, n_classes)) / 2
    y_pred = np.argmax(proba, axis=1)

    # ── Metrics ───────────────────────────────────────────────────────────────
    acc          = accuracy_score(y_test, y_pred)
    macro_f1     = f1_score(y_test, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_test, y_pred, average="macro", zero_division=0)

    pred_labels = le.inverse_transform(y_pred)
    true_labels = le.inverse_transform(y_test)

    print("=" * 55)
    print("  Walk-Forward Results (70 / 30 Amazon-only)")
    print("=" * 55)
    print(f"  Accuracy     : {acc:.1%}")
    print(f"  Macro F1     : {macro_f1:.1%}")
    print(f"  Macro Recall : {macro_recall:.1%}")
    print()
    print(classification_report(true_labels, pred_labels, zero_division=0))

    # ── Save results ──────────────────────────────────────────────────────────
    out = test_df[["DATE", "LAST_PRICE", "NEXT_PRICE", "NEXT_RETURN", TARGET_COL]].copy()
    out["PREDICTED_LABEL"] = pred_labels
    out["CORRECT"]         = out[TARGET_COL] == out["PREDICTED_LABEL"]
    out.to_csv(RESULTS_CSV, index=False)
    print(f"Results saved → {RESULTS_CSV}")


if __name__ == "__main__":
    main()
