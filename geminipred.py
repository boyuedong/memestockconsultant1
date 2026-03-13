import os
import json
import sys
import time
from typing import Literal

def out(msg: str) -> None:
    print(msg, flush=True)

# Check dependencies and API key before heavy imports
if not os.environ.get("GEMINI_API_KEY"):
    out("Error: GEMINI_API_KEY is not set.")
    out("Set it in your terminal: export GEMINI_API_KEY='your_key_here'")
    out("Get a key at: https://aistudio.google.com/apikey")
    sys.exit(1)

try:
    from google import genai
except ImportError:
    out("Error: google-genai package not installed.")
    out("Install with: pip install google-genai")
    sys.exit(1)

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.metrics import classification_report, accuracy_score

# =========================
# Config
# =========================
FILE_PATH = "/Users/boyuedong/Desktop/new3:11/full_dataset_cleaned.csv"
MODEL_NAME = "gemini-2.5-flash"
WINDOW_SIZE = 5
EVAL_LAST_N = 5   # free tier allows ~20 requests/day; use 5 to stay under + add delay
REQUEST_DELAY_SEC = 2   # delay between API calls to avoid rate limit

# Requires: export GEMINI_API_KEY='your_key_here'
# or set it in your shell before running
# pip install -U google-genai pydantic pandas numpy scikit-learn

# =========================
# Structured response schema
# =========================
class BuyDecision(BaseModel):
    label: Literal["must buy", "maybe buy", "don't buy", "definitely don't buy"]
    confidence: float
    reasoning: str

# =========================
# Label definition
# =========================
def classify_return(r: float) -> str:
    if r >= 0.02:
        return "must buy"
    elif r >= 0.0:
        return "maybe buy"
    elif r > -0.02:
        return "don't buy"
    else:
        return "definitely don't buy"

# =========================
# Load + clean Amazon data
# =========================
use_cols = [
    "STOCK",
    "DATE",
    "LAST_PRICE",
    "PX_VOLUME",
    "VOLATILITY_10D",
    "VOLATILITY_30D",
    "LSTM_POLARITY",
    "TEXTBLOB_POLARITY",
]

out("Loading CSV...")
df = pd.read_csv(FILE_PATH, usecols=use_cols)

amazon = df[df["STOCK"].astype(str).str.strip().str.lower() == "amazon"].copy()
amazon["DATE"] = pd.to_datetime(amazon["DATE"], format="%d/%m/%Y", errors="coerce")

for col in [
    "LAST_PRICE",
    "PX_VOLUME",
    "VOLATILITY_10D",
    "VOLATILITY_30D",
    "LSTM_POLARITY",
    "TEXTBLOB_POLARITY",
]:
    amazon[col] = pd.to_numeric(amazon[col], errors="coerce")

amazon = (
    amazon
    .dropna(subset=["DATE", "LAST_PRICE"])
    .sort_values("DATE")
    .drop_duplicates(subset=["DATE"], keep="last")
    .reset_index(drop=True)
)

# =========================
# Build rolling examples
# Predict class of row i using rows [i-WINDOW_SIZE, ..., i-1]
# =========================
feature_cols = [
    "LAST_PRICE",
    "PX_VOLUME",
    "VOLATILITY_10D",
    "VOLATILITY_30D",
    "LSTM_POLARITY",
    "TEXTBLOB_POLARITY",
]

examples = []

for i in range(WINDOW_SIZE, len(amazon)):
    window_df = amazon.iloc[i - WINDOW_SIZE:i].copy()

    # Skip if any feature in the historical window is missing
    if window_df[feature_cols].isna().any().any():
        continue

    prev_price = amazon.iloc[i - 1]["LAST_PRICE"]
    curr_price = amazon.iloc[i]["LAST_PRICE"]
    actual_return = (curr_price - prev_price) / prev_price
    actual_label = classify_return(actual_return)

    examples.append({
        "target_date": str(amazon.iloc[i]["DATE"].date()),
        "window_rows": window_df[["DATE"] + feature_cols].assign(
            DATE=lambda x: x["DATE"].dt.strftime("%Y-%m-%d")
        ).to_dict(orient="records"),
        "actual_label": actual_label,
        "actual_return": actual_return,
        "prev_price": prev_price,
        "curr_price": curr_price,
    })

if len(examples) == 0:
    raise ValueError("No usable Amazon examples found.")

# We'll evaluate on the last N examples
eval_examples = examples[-EVAL_LAST_N:] if len(examples) >= EVAL_LAST_N else examples

# =========================
# Gemini client
# =========================
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

SYSTEM_PROMPT = """
You are a cautious financial classification assistant.

You will receive only historical observations for Amazon up to but NOT including the target date.
Each observation contains:
- DATE
- LAST_PRICE
- PX_VOLUME
- VOLATILITY_10D
- VOLATILITY_30D
- LSTM_POLARITY
- TEXTBLOB_POLARITY

Your task:
Classify the NEXT available Amazon observation into exactly one label:
- must buy
- maybe buy
- don't buy
- definitely don't buy

Use this interpretation:
- must buy: likely large rise
- maybe buy: likely small rise or flat-to-slightly-up
- don't buy: likely small drop
- definitely don't buy: likely large drop

Be conservative. Return only structured JSON.
"""

def predict_with_gemini(target_date: str, window_rows: list[dict]) -> BuyDecision:
    from google.genai.errors import ClientError

    user_prompt = f"""
Target date to classify: {target_date}

Historical Amazon observations only:
{json.dumps(window_rows, indent=2)}

Return one label and a confidence score from 0 to 1.
"""

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=user_prompt,
                config={
                    "system_instruction": SYSTEM_PROMPT,
                    "response_mime_type": "application/json",
                    "response_json_schema": BuyDecision.model_json_schema(),
                },
            )
            parsed = BuyDecision.model_validate_json(response.text)
            return parsed
        except ClientError as e:
            is_429 = getattr(e, "status_code", None) == 429 or "429" in str(e)
            if is_429 and attempt < 2:
                wait = 35  # free tier: retry after ~35s
                out(f"  Rate limited (429). Waiting {wait}s before retry {attempt + 1}/3...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Still rate limited after 3 retries. Try again later or reduce EVAL_LAST_N.")

# =========================
# Run evaluation
# =========================
pred_rows = []

out(f"Evaluating on {len(eval_examples)} examples with Gemini ({MODEL_NAME})...")
for idx, ex in enumerate(eval_examples, start=1):
    if idx > 1:
        time.sleep(REQUEST_DELAY_SEC)
    out(f"Predicting {idx}/{len(eval_examples)} -> {ex['target_date']}")
    pred = predict_with_gemini(ex["target_date"], ex["window_rows"])

    pred_rows.append({
        "DATE": ex["target_date"],
        "PREDICTED_LABEL": pred.label,
        "CONFIDENCE": pred.confidence,
        "REASONING": pred.reasoning,
        "ACTUAL_LABEL": ex["actual_label"],
        "ACTUAL_RETURN": ex["actual_return"],
        "PREV_PRICE": ex["prev_price"],
        "CURR_PRICE": ex["curr_price"],
        "CORRECT": pred.label == ex["actual_label"],
    })

results = pd.DataFrame(pred_rows)

out("\n=== LLM Evaluation ===")
out("Accuracy: " + str(accuracy_score(results["ACTUAL_LABEL"], results["PREDICTED_LABEL"])))
out(classification_report(results["ACTUAL_LABEL"], results["PREDICTED_LABEL"]))

out("\n=== Final Prediction (most recent Amazon date) ===")
out(results.tail(1).to_string(index=False))

results_path = "/Users/boyuedong/Desktop/new3:11/amazon_gemini_results.csv"
results.to_csv(results_path, index=False)
out("Saved results to: " + results_path)