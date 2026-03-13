import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# =========================
# Load cleaned file
# =========================
def out(msg):
    print(msg, flush=True)

out("Loading CSV...")
file_path = "/Users/boyuedong/Desktop/new3:11/full_dataset_cleaned.csv"

use_cols = [
    "STOCK",
    "DATE",
    "LAST_PRICE",
    "PX_VOLUME",
    "VOLATILITY_10D",
    "VOLATILITY_30D",
    "LSTM_POLARITY",
    "TEXTBLOB_POLARITY"
]

df = pd.read_csv(file_path, usecols=use_cols)

# =========================
# Filter Amazon and clean
# =========================
amazon = df[df["STOCK"].astype(str).str.strip().str.lower() == "amazon"].copy()

amazon["DATE"] = pd.to_datetime(amazon["DATE"], errors="coerce")
for col in [
    "LAST_PRICE",
    "PX_VOLUME",
    "VOLATILITY_10D",
    "VOLATILITY_30D",
    "LSTM_POLARITY",
    "TEXTBLOB_POLARITY"
]:
    amazon[col] = pd.to_numeric(amazon[col], errors="coerce")

amazon = amazon.dropna(subset=["DATE", "LAST_PRICE"])
amazon = amazon.sort_values("DATE").drop_duplicates(subset=["DATE"], keep="last").reset_index(drop=True)

out("Amazon rows: " + str(len(amazon)))
out(str(amazon[["DATE", "LAST_PRICE"]].tail(10)))

# =========================
# Label function
# =========================
def classify_return(r):
    if r >= 0.02:
        return "must buy"
    elif r >= 0.0:
        return "maybe buy"
    elif r > -0.02:
        return "don't buy"
    else:
        return "definitely don't buy"

# =========================
# Feature columns
# =========================
feature_cols = [
    "LAST_PRICE",
    "PX_VOLUME",
    "VOLATILITY_10D",
    "VOLATILITY_30D",
    "LSTM_POLARITY",
    "TEXTBLOB_POLARITY"
]

window_size = 5

X = []
y = []
target_dates = []

# =========================
# Build rolling-window samples
# Only use rows before target row i
# =========================
for i in range(window_size, len(amazon)):
    window_df = amazon.iloc[i - window_size:i]

    # Skip if any feature in the input window is missing
    if window_df[feature_cols].isna().any().any():
        continue

    prev_price = amazon.iloc[i - 1]["LAST_PRICE"]
    curr_price = amazon.iloc[i]["LAST_PRICE"]

    actual_return = (curr_price - prev_price) / prev_price
    label = classify_return(actual_return)

    # Flatten window into one feature vector
    x_window = window_df[feature_cols].values.flatten()

    X.append(x_window)
    y.append(label)
    target_dates.append(amazon.iloc[i]["DATE"])

X = np.array(X)
y = np.array(y)
target_dates = np.array(target_dates)

out("Dataset shape: " + str(X.shape))
out("Number of labels: " + str(len(y)))

# =========================
# Train/test split by time
# =========================
if len(X) < 10:
    raise ValueError("Not enough Amazon samples after windowing. Try reducing window_size.")

split_idx = int(len(X) * 0.8)

X_train = X[:split_idx]
y_train = y[:split_idx]
X_test = X[split_idx:]
y_test = y[split_idx:]
dates_test = target_dates[split_idx:]

# =========================
# Train classifier
# =========================
clf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

out("\n=== Holdout Evaluation ===")
out("Accuracy: " + str(accuracy_score(y_test, y_pred)))
out(classification_report(y_test, y_pred))

# =========================
# Final prediction = last available Amazon date
# =========================
out("\n=== Final Amazon Prediction ===")
out("Target date: " + str(pd.Timestamp(dates_test[-1]).date()))
out("Predicted class: " + str(y_pred[-1]))
out("Actual class   : " + str(y_test[-1]))

# =========================
# Save prediction results
# =========================
results = pd.DataFrame({
    "DATE": pd.to_datetime(dates_test),
    "PREDICTED_LABEL": y_pred,
    "ACTUAL_LABEL": y_test,
    "CORRECT": (y_pred == y_test)
})

results_path = "/Users/boyuedong/Desktop/new3:11/amazon_classification_results.csv"
results.to_csv(results_path, index=False)

out("Saved results to: " + results_path)