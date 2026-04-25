"""
Train the classical XGB/LGBM walk-forward pipeline on a selected 20-stock basket.

Outputs:
  - selected_stocks_20.csv
  - selected_stocks_walkforward_results.csv
  - selected_stocks_model_summary.csv
  - selected_stocks_failures.csv
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.preprocessing import LabelEncoder, RobustScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "full_dataset_cleaned.csv"

TARGET_STOCK_COUNT = 20

# Candidate mapping. We auto-pick the strongest names by date coverage from this list.
CANDIDATE_TICKERS = {
    "adidas": "ADS.DE",
    "adobe": "ADBE",
    "amazon": "AMZN",
    "apple": "AAPL",
    "bp": "BP",
    "costco": "COST",
    "disney": "DIS",
    "ebay": "EBAY",
    "facebook": "META",
    "ford": "F",
    "google": "GOOGL",
    "honda": "HMC",
    "hp": "HPQ",
    "intel": "INTC",
    "mcdonald's": "MCD",
    "microsoft": "MSFT",
    "netflix": "NFLX",
    "starbucks": "SBUX",
    "toyota": "TM",
    "ups": "UPS",
    "verizon": "VZ",
    "visa": "V",
    "walmart": "WMT",
    "wells fargo": "WFC",
    "bank of america": "BAC",
    "boeing": "BA",
    "blackrock": "BLK",
    "american express": "AXP",
    "sony": "SONY",
    "sap": "SAP",
    "vodafone": "VOD",
    "santander": "SAN",
}

WINDOW_MIN_ROWS = 80
TRAIN_FRACTION = 0.70

RESULTS_ROWS_CSV = BASE_DIR / "selected_stocks_walkforward_results.csv"
SUMMARY_CSV = BASE_DIR / "selected_stocks_model_summary.csv"
FAILURES_CSV = BASE_DIR / "selected_stocks_failures.csv"
STOCK_LIST_CSV = BASE_DIR / "selected_stocks_20.csv"

BASE_NUMERIC_COLS = [
    "LAST_PRICE",
    "PX_VOLUME",
    "VOLATILITY_10D",
    "VOLATILITY_30D",
    "LSTM_POLARITY",
    "TEXTBLOB_POLARITY",
]


def classify_return(r: float) -> str:
    if pd.isna(r):
        return np.nan
    if r >= 0.02:
        return "must buy"
    if r >= 0.0:
        return "maybe buy"
    if r > -0.02:
        return "don't buy"
    return "definitely don't buy"


def pct(a, b):
    return np.where(b != 0, (a - b) / b, np.nan)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    lp = out["LAST_PRICE"]
    yfc = out["YF_CLOSE"]
    vol = out["PX_VOLUME"]
    yfvol = out["YF_VOLUME"]

    out["price_return_1"] = lp.pct_change(1)
    out["price_return_3"] = lp.pct_change(3)
    out["price_return_5"] = lp.pct_change(5)

    out["yf_return_1"] = yfc.pct_change(1)
    out["yf_return_3"] = yfc.pct_change(3)
    out["yf_return_5"] = yfc.pct_change(5)

    for w in [3, 5, 10]:
        out[f"price_ma_{w}"] = lp.rolling(w).mean()
        out[f"yf_close_ma_{w}"] = yfc.rolling(w).mean()
        out[f"price_vs_ma_{w}"] = pct(lp.values, out[f"price_ma_{w}"].values)
        out[f"yf_close_vs_ma_{w}"] = pct(yfc.values, out[f"yf_close_ma_{w}"].values)

    out["volume_change_1"] = vol.pct_change(1)
    out["yf_volume_change_1"] = yfvol.pct_change(1)
    out["volume_shock"] = pct(yfvol.values, yfvol.rolling(10).mean().values)

    out["vol_spread"] = out["VOLATILITY_10D"] - out["VOLATILITY_30D"]
    out["sentiment_avg"] = (out["LSTM_POLARITY"] + out["TEXTBLOB_POLARITY"]) / 2
    out["sentiment_diff"] = out["LSTM_POLARITY"] - out["TEXTBLOB_POLARITY"]

    out["hl_range"] = pct((out["YF_HIGH"] - out["YF_LOW"]).values, out["YF_CLOSE"].values)
    out["oc_change"] = pct((out["YF_CLOSE"] - out["YF_OPEN"]).values, out["YF_OPEN"].values)
    out["rolling_vol_10"] = yfc.pct_change().rolling(10).std()

    delta = yfc.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    out["NEXT_PRICE"] = out["LAST_PRICE"].shift(-1)
    out["NEXT_RETURN"] = pct(out["NEXT_PRICE"].values, out["LAST_PRICE"].values)
    out["TARGET_LABEL"] = out["NEXT_RETURN"].apply(classify_return)
    out = out.dropna(subset=["TARGET_LABEL"]).reset_index(drop=True)

    return out


def safe_proba(model, X: np.ndarray, n_classes: int, class_backmap: dict[int, int]) -> np.ndarray:
    p = model.predict_proba(X)
    if p.shape[1] == n_classes:
        return p
    full = np.zeros((len(X), n_classes))
    for i, cls in enumerate(model.classes_):
        full_idx = class_backmap.get(int(cls))
        if full_idx is not None:
            full[:, full_idx] = p[:, i]
    return full


def choose_stocks(df: pd.DataFrame) -> dict[str, str]:
    coverage = (
        df.groupby("STOCK")["DATE"]
        .nunique()
        .reset_index(name="unique_dates")
        .sort_values("unique_dates", ascending=False)
    )
    coverage["has_ticker"] = coverage["STOCK"].isin(CANDIDATE_TICKERS.keys())
    picked = coverage[
        (coverage["has_ticker"]) & (coverage["unique_dates"] >= WINDOW_MIN_ROWS)
    ].head(TARGET_STOCK_COUNT)

    selected = {stock: CANDIDATE_TICKERS[stock] for stock in picked["STOCK"].tolist()}
    if len(selected) < 10:
        raise ValueError(
            f"Only found {len(selected)} mapped stocks with >= {WINDOW_MIN_ROWS} dates. "
            "Expand CANDIDATE_TICKERS or lower WINDOW_MIN_ROWS."
        )
    return selected


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input CSV: {INPUT_CSV}")

    use_cols = ["STOCK", "DATE"] + BASE_NUMERIC_COLS
    df = pd.read_csv(INPUT_CSV, usecols=use_cols)
    df["STOCK"] = df["STOCK"].astype(str).str.strip().str.lower()
    df["DATE"] = pd.to_datetime(df["DATE"], dayfirst=True, errors="coerce")
    for c in BASE_NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["DATE"]).copy()

    selected_stocks = choose_stocks(df)
    pd.DataFrame(
        [{"stock": stock, "ticker": ticker} for stock, ticker in selected_stocks.items()]
    ).to_csv(STOCK_LIST_CSV, index=False)
    print(f"Selected {len(selected_stocks)} stocks. Saved list -> {STOCK_LIST_CSV}")

    all_results = []
    summary_rows = []
    failures = []

    for idx, (stock_name, ticker) in enumerate(selected_stocks.items(), start=1):
        print(f"\n[{idx}/{len(selected_stocks)}] {stock_name} ({ticker})")
        try:
            stock_df = df[df["STOCK"] == stock_name].copy()
            stock_df = (
                stock_df.sort_values("DATE")
                .drop_duplicates(subset=["DATE"], keep="last")
                .reset_index(drop=True)
            )
            stock_df["DATE"] = pd.to_datetime(stock_df["DATE"], errors="coerce").astype(
                "datetime64[ns]"
            )
            if len(stock_df) < WINDOW_MIN_ROWS:
                raise ValueError(f"not enough unique-date rows ({len(stock_df)})")

            start_date = stock_df["DATE"].min().strftime("%Y-%m-%d")
            end_date = (stock_df["DATE"].max() + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
            yf_raw = yf.download(
                ticker, start=start_date, end=end_date, auto_adjust=False, progress=False
            ).reset_index()
            if yf_raw.empty:
                raise ValueError("yfinance returned no rows")
            if isinstance(yf_raw.columns, pd.MultiIndex):
                yf_raw.columns = [c[0] if isinstance(c, tuple) else c for c in yf_raw.columns]

            yf_raw = yf_raw.rename(columns={"Date": "DATE"})
            yf_raw["DATE"] = pd.to_datetime(yf_raw["DATE"], errors="coerce").astype(
                "datetime64[ns]"
            )
            yf_raw = yf_raw.rename(
                columns={
                    "Open": "YF_OPEN",
                    "High": "YF_HIGH",
                    "Low": "YF_LOW",
                    "Close": "YF_CLOSE",
                    "Volume": "YF_VOLUME",
                }
            )
            keep = ["DATE", "YF_OPEN", "YF_HIGH", "YF_LOW", "YF_CLOSE", "YF_VOLUME"]
            yf_raw = yf_raw[keep].sort_values("DATE").reset_index(drop=True)

            stock_df = pd.merge_asof(
                stock_df.sort_values("DATE"), yf_raw, on="DATE", direction="backward"
            ).sort_values("DATE").reset_index(drop=True)

            feat_df = add_features(stock_df)

            feature_cols = [
                "LAST_PRICE",
                "PX_VOLUME",
                "VOLATILITY_10D",
                "VOLATILITY_30D",
                "LSTM_POLARITY",
                "TEXTBLOB_POLARITY",
                "YF_OPEN",
                "YF_HIGH",
                "YF_LOW",
                "YF_CLOSE",
                "YF_VOLUME",
                "price_return_1",
                "price_return_3",
                "price_return_5",
                "yf_return_1",
                "yf_return_3",
                "yf_return_5",
                "price_ma_3",
                "yf_close_ma_3",
                "price_ma_5",
                "yf_close_ma_5",
                "price_ma_10",
                "yf_close_ma_10",
                "price_vs_ma_3",
                "yf_close_vs_ma_3",
                "price_vs_ma_5",
                "yf_close_vs_ma_5",
                "price_vs_ma_10",
                "yf_close_vs_ma_10",
                "volume_change_1",
                "yf_volume_change_1",
                "volume_shock",
                "vol_spread",
                "sentiment_avg",
                "sentiment_diff",
                "hl_range",
                "oc_change",
                "rolling_vol_10",
                "rsi_14",
            ]
            feat_df = feat_df.dropna(subset=feature_cols).reset_index(drop=True)
            if len(feat_df) < WINDOW_MIN_ROWS:
                raise ValueError(f"not enough model-ready rows ({len(feat_df)})")

            split_idx = int(len(feat_df) * TRAIN_FRACTION)
            train_df = feat_df.iloc[:split_idx].copy()
            test_df = feat_df.iloc[split_idx:].copy()
            if len(train_df) < 20 or len(test_df) < 10:
                raise ValueError(f"split too small: train={len(train_df)}, test={len(test_df)}")

            le = LabelEncoder()
            le.fit(feat_df["TARGET_LABEL"])
            y_train_full = le.transform(train_df["TARGET_LABEL"])
            y_test = le.transform(test_df["TARGET_LABEL"])
            train_classes = sorted(np.unique(y_train_full).tolist())
            if len(train_classes) < 2:
                raise ValueError("training split has fewer than 2 classes")
            forward_map = {orig: i for i, orig in enumerate(train_classes)}
            back_map = {i: orig for orig, i in forward_map.items()}
            y_train = np.array([forward_map[v] for v in y_train_full], dtype=np.int32)

            X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan)
            X_test = test_df[feature_cols].replace([np.inf, -np.inf], np.nan)
            train_medians = X_train.median()
            X_train = X_train.fillna(train_medians)
            X_test = X_test.fillna(train_medians)

            scaler = RobustScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            xgb = XGBClassifier(
                n_estimators=250,
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
                n_estimators=250,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
            )

            xgb.fit(X_train, y_train)
            lgbm.fit(X_train, y_train)

            n_classes = len(le.classes_)
            proba = (
                safe_proba(xgb, X_test, n_classes, back_map)
                + safe_proba(lgbm, X_test, n_classes, back_map)
            ) / 2
            y_pred = np.argmax(proba, axis=1)

            pred_labels = le.inverse_transform(y_pred)
            true_labels = le.inverse_transform(y_test)

            acc = accuracy_score(true_labels, pred_labels)
            macro_f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
            macro_recall = recall_score(true_labels, pred_labels, average="macro", zero_division=0)

            out = test_df[["DATE", "LAST_PRICE", "NEXT_PRICE", "NEXT_RETURN", "TARGET_LABEL"]].copy()
            out["STOCK"] = stock_name
            out["TICKER"] = ticker
            out["PREDICTED_LABEL"] = pred_labels
            out["CORRECT"] = out["TARGET_LABEL"] == out["PREDICTED_LABEL"]
            all_results.append(out)

            summary_rows.append(
                {
                    "stock": stock_name,
                    "ticker": ticker,
                    "rows_raw": len(stock_df),
                    "rows_model": len(feat_df),
                    "rows_train": len(train_df),
                    "rows_test": len(test_df),
                    "accuracy": round(float(acc), 4),
                    "macro_f1": round(float(macro_f1), 4),
                    "macro_recall": round(float(macro_recall), 4),
                }
            )
            print(
                f"  done: model_rows={len(feat_df)} test={len(test_df)} "
                f"acc={acc:.3f} f1={macro_f1:.3f}"
            )

        except Exception as e:
            failures.append({"stock": stock_name, "ticker": ticker, "reason": str(e)})
            print(f"  skipped: {e}")

    if all_results:
        pd.concat(all_results, ignore_index=True).to_csv(RESULTS_ROWS_CSV, index=False)
    else:
        pd.DataFrame(
            columns=[
                "DATE",
                "LAST_PRICE",
                "NEXT_PRICE",
                "NEXT_RETURN",
                "TARGET_LABEL",
                "STOCK",
                "TICKER",
                "PREDICTED_LABEL",
                "CORRECT",
            ]
        ).to_csv(RESULTS_ROWS_CSV, index=False)

    if summary_rows:
        pd.DataFrame(summary_rows).sort_values("accuracy", ascending=False).to_csv(
            SUMMARY_CSV, index=False
        )
    else:
        pd.DataFrame(
            columns=[
                "stock",
                "ticker",
                "rows_raw",
                "rows_model",
                "rows_train",
                "rows_test",
                "accuracy",
                "macro_f1",
                "macro_recall",
            ]
        ).to_csv(SUMMARY_CSV, index=False)

    pd.DataFrame(failures).to_csv(FAILURES_CSV, index=False)

    print("\n" + "=" * 70)
    print(f"Completed: {len(summary_rows)} stocks")
    print(f"Failed   : {len(failures)} stocks")
    print(f"Saved rows     -> {RESULTS_ROWS_CSV}")
    print(f"Saved summary  -> {SUMMARY_CSV}")
    print(f"Saved failures -> {FAILURES_CSV}")
    print(f"Saved stock list -> {STOCK_LIST_CSV}")


if __name__ == "__main__":
    main()
