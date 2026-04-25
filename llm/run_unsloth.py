"""
run_unsloth.py — LLM SFT pipeline using HuggingFace + LoRA (Mac/CPU compatible)
=================================================================================
Replaces Unsloth (CUDA-only) with standard HuggingFace + peft so it runs on:
  • Mac Apple Silicon  (uses MPS Metal GPU if available)
  • Mac Intel          (CPU)
  • Linux / Colab      (CUDA if available, else CPU)

Model: Qwen/Qwen2.5-0.5B-Instruct  (~1 GB RAM, fast on CPU)
       Change BASE_MODEL below for a larger model if you have more RAM.

Install (one time):
    pip install transformers peft trl accelerate datasets scikit-learn

Run:
    python3 llm/run_unsloth.py
"""

import os
import json
import torch
import pandas as pd

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorForSeq2Seq
from datasets import Dataset

# =============================================================================
# Device detection  (MPS → CUDA → CPU)
# =============================================================================
if torch.backends.mps.is_available():
    DEVICE = "mps"
    DTYPE  = torch.float32   # MPS is most stable with float32 during training
elif torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE  = torch.float16
else:
    DEVICE = "cpu"
    DTYPE  = torch.float32

print(f"Device: {DEVICE}  |  dtype: {DTYPE}")

# =============================================================================
# Config
# =============================================================================
BASE_MODEL   = "Qwen/Qwen2.5-0.5B-Instruct"   # ~1 GB RAM — runs on any Mac
# Alternatives (better accuracy, more RAM):
#   "Qwen/Qwen2.5-1.5B-Instruct"              ~3 GB
#   "microsoft/Phi-3-mini-4k-instruct"         ~7.5 GB
#   "meta-llama/Llama-3.2-1B-Instruct"        ~2 GB (needs HF login)

MODEL_SAVE   = os.path.join(os.path.dirname(__file__), "model_out")
CORRECTIONS  = os.path.join(os.path.dirname(__file__), "corrections_amazon.json")
SFT_OUT_DIR  = os.path.join(os.path.dirname(__file__), "sft_outputs")

# Note: path uses  new3:11  (colon), NOT  new3/11  (slash)
INPUT_CSV    = "/Users/boyuedong/Desktop/new3:11/amazon_only_enriched.csv"
RESULTS_CSV  = "/Users/boyuedong/Desktop/new3:11/amazon_unsloth_results.csv"

WINDOW_SIZE   = 5
TRAIN_SPLIT   = 0.70   # walk-forward: first 70% of rows → train, last 30% → test
ITERATIONS    = 3

LORA_RANK    = 8      # small rank — we have very few training examples
LORA_ALPHA   = 16
MAX_SEQ_LEN  = 1024

VALID_LABELS = ["must buy", "maybe buy", "don't buy", "definitely don't buy"]

SYSTEM_PROMPT = (
    "You are a stock decision classifier. "
    "Given historical Amazon stock observations, classify the next available "
    "Amazon move into exactly one of: "
    "must buy, maybe buy, don't buy, definitely don't buy. "
    "Reply with ONLY one label — nothing else."
)

# =============================================================================
# Helpers
# =============================================================================
def normalize_label(text: str) -> str:
    """Map any model output to the nearest valid label."""
    if not text:
        return "maybe buy"
    t = text.strip().lower()
    if t in VALID_LABELS:
        return t
    # Longest match first (so "definitely don't buy" beats "don't buy")
    for lbl in sorted(VALID_LABELS, key=len, reverse=True):
        if lbl in t:
            return lbl
    if "definitely" in t:               return "definitely don't buy"
    if "must" in t:                     return "must buy"
    if "maybe" in t or "perhaps" in t:  return "maybe buy"
    if "don" in t or "not" in t:        return "don't buy"
    return "maybe buy"


def classify_return(r: float) -> str:
    if r >= 0.02:   return "must buy"
    if r >= 0.0:    return "maybe buy"
    if r > -0.02:   return "don't buy"
    return "definitely don't buy"


def _fmt(val, decimals=4):
    """Format a numeric value safely, returning 'N/A' for missing."""
    try:
        return f"{float(val):.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


def build_user_prompt(window_df: pd.DataFrame, target_date) -> str:
    lines = [
        f"Target date: {pd.Timestamp(target_date).strftime('%Y-%m-%d')}",
        "Historical Amazon observations (most recent last):",
    ]
    for _, row in window_df.iterrows():
        lines.append(
            f"\n  Date: {pd.Timestamp(row['DATE']).strftime('%Y-%m-%d')}"
            # ── Original dataset features ──────────────────────────────
            f"\n    price={_fmt(row.get('LAST_PRICE'), 2)}"
            f"  px_volume={_fmt(row.get('PX_VOLUME'), 0)}"
            f"  volatility_10d={_fmt(row.get('VOLATILITY_10D'), 3)}"
            f"  volatility_30d={_fmt(row.get('VOLATILITY_30D'), 3)}"
            f"\n    lstm_sentiment={_fmt(row.get('LSTM_POLARITY'), 4)}"
            f"  textblob_sentiment={_fmt(row.get('TEXTBLOB_POLARITY'), 4)}"
            f"  sentiment_avg={_fmt(row.get('sentiment_avg'), 4)}"
            f"  sentiment_diff={_fmt(row.get('sentiment_diff'), 4)}"
            # ── yfinance OHLCV ─────────────────────────────────────────
            f"\n    yf_open={_fmt(row.get('YF_OPEN'), 2)}"
            f"  yf_high={_fmt(row.get('YF_HIGH'), 2)}"
            f"  yf_low={_fmt(row.get('YF_LOW'), 2)}"
            f"  yf_close={_fmt(row.get('YF_CLOSE'), 2)}"
            f"  yf_volume={_fmt(row.get('YF_VOLUME'), 0)}"
            # ── Engineered return features ─────────────────────────────
            f"\n    ret_1d={_fmt(row.get('yf_return_1'), 4)}"
            f"  ret_3d={_fmt(row.get('yf_return_3'), 4)}"
            f"  ret_5d={_fmt(row.get('yf_return_5'), 4)}"
            f"  price_ret_1={_fmt(row.get('price_return_1'), 4)}"
            f"  price_ret_5={_fmt(row.get('price_return_5'), 4)}"
            # ── Moving averages & distance from MA ─────────────────────
            f"\n    ma_3={_fmt(row.get('yf_close_ma_3'), 2)}"
            f"  ma_5={_fmt(row.get('yf_close_ma_5'), 2)}"
            f"  ma_10={_fmt(row.get('yf_close_ma_10'), 2)}"
            f"  vs_ma3={_fmt(row.get('yf_close_vs_ma_3'), 4)}"
            f"  vs_ma5={_fmt(row.get('yf_close_vs_ma_5'), 4)}"
            f"  vs_ma10={_fmt(row.get('yf_close_vs_ma_10'), 4)}"
            # ── Volume features ────────────────────────────────────────
            f"\n    vol_change_1d={_fmt(row.get('yf_volume_change_1'), 4)}"
            f"  volume_shock={_fmt(row.get('volume_shock'), 4)}"
            # ── Intraday / momentum features ───────────────────────────
            f"\n    hl_range={_fmt(row.get('hl_range'), 4)}"
            f"  oc_change={_fmt(row.get('oc_change'), 4)}"
            f"  rolling_vol_10={_fmt(row.get('rolling_vol_10'), 4)}"
            f"  rsi_14={_fmt(row.get('rsi_14'), 2)}"
        )
    lines.append("\nClassify the next move.")
    return "\n".join(lines)


# ChatML format used by Qwen2.5 and other models
def apply_chat_template(tokenizer, user_input: str, assistant_output: str = None) -> str:
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": user_input},
    ]
    if assistant_output is not None:
        messages.append({"role": "assistant", "content": assistant_output})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=(assistant_output is None),
    )


# =============================================================================
# 1. Load data
# =============================================================================
def load_examples():
    print("=" * 60)
    print("Step 1: Loading data …")

    df = pd.read_csv(INPUT_CSV)
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

    for col in ["LAST_PRICE", "PX_VOLUME", "VOLATILITY_10D", "VOLATILITY_30D",
                "LSTM_POLARITY", "TEXTBLOB_POLARITY",
                "YF_OPEN", "YF_HIGH", "YF_LOW", "YF_CLOSE", "YF_VOLUME"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("DATE").reset_index(drop=True)

    # Compute TARGET_LABEL if missing
    if "TARGET_LABEL" not in df.columns:
        if "NEXT_RETURN" in df.columns:
            df["TARGET_LABEL"] = df["NEXT_RETURN"].apply(
                lambda x: classify_return(x) if pd.notna(x) else None)
        elif "NEXT_PRICE" in df.columns:
            df["NEXT_RETURN"] = (df["NEXT_PRICE"] - df["LAST_PRICE"]) / df["LAST_PRICE"]
            df["TARGET_LABEL"] = df["NEXT_RETURN"].apply(
                lambda x: classify_return(x) if pd.notna(x) else None)
        else:
            raise ValueError("CSV must contain TARGET_LABEL, NEXT_RETURN, or NEXT_PRICE.")

    examples = []
    for i in range(WINDOW_SIZE, len(df)):
        window     = df.iloc[i - WINDOW_SIZE:i].copy()
        target_row = df.iloc[i]
        if pd.isna(target_row["TARGET_LABEL"]):
            continue
        if window.drop(columns=["DATE"], errors="ignore").isna().any().any():
            continue
        examples.append({
            "date":  pd.Timestamp(target_row["DATE"]).strftime("%Y-%m-%d"),
            "year":  pd.Timestamp(target_row["DATE"]).year,
            "input": build_user_prompt(window, target_row["DATE"]),
            "label": normalize_label(str(target_row["TARGET_LABEL"])),
        })

    # Walk-forward 70/30 chronological split (same strategy as train_walkforward.py)
    split_idx = int(len(examples) * TRAIN_SPLIT)
    train_ex  = examples[:split_idx]
    test_ex   = examples[split_idx:]

    print(f"  Total examples : {len(examples)}")
    print(f"  Train (first {int(TRAIN_SPLIT*100)}%) : {len(train_ex)}  "
          f"({train_ex[0]['date'] if train_ex else '?'} → {train_ex[-1]['date'] if train_ex else '?'})")
    print(f"  Test  (last  {int((1-TRAIN_SPLIT)*100)}%) : {len(test_ex)}  "
          f"({test_ex[0]['date'] if test_ex else '?'} → {test_ex[-1]['date'] if test_ex else '?'})")

    if train_ex:
        print("\n  Sample training example (first):")
        print("  " + train_ex[0]["input"].replace("\n", "\n  ")[:600])
        print(f"  → LABEL: {train_ex[0]['label']}")

    return train_ex, test_ex


# =============================================================================
# 2. Load model + apply LoRA
# =============================================================================
def load_model_and_tokenizer(checkpoint: str = None):
    model_path = checkpoint or BASE_MODEL
    print(f"\n  Loading model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype    = DTYPE,
        trust_remote_code = True,
        device_map     = None,         # we move manually below
    )

    # Only apply LoRA if loading the base model (not a saved PEFT adapter)
    if checkpoint is None or not os.path.exists(
            os.path.join(checkpoint, "adapter_config.json")):
        lora_cfg = LoraConfig(
            r              = LORA_RANK,
            lora_alpha     = LORA_ALPHA,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"],
            lora_dropout   = 0.05,
            bias           = "none",
            task_type      = TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
    else:
        model = PeftModel.from_pretrained(model, checkpoint)
        print("  Loaded saved LoRA adapter.")

    model = model.to(DEVICE)
    return model, tokenizer


# =============================================================================
# 3. SFT training on corrections
# =============================================================================
def run_sft(model, tokenizer, corrections: list, iteration: int) -> object:
    if len(corrections) < 2:
        print(f"  Only {len(corrections)} corrections — need ≥2. Skipping SFT.")
        return model

    out_dir = os.path.join(SFT_OUT_DIR, f"iter_{iteration}")
    os.makedirs(out_dir, exist_ok=True)

    # Format each correction as a full chat string
    texts = [
        apply_chat_template(tokenizer, c["input"], c["label"])
        for c in corrections
    ]
    dataset = Dataset.from_dict({"text": texts})

    trainer = SFTTrainer(
        model             = model,
        processing_class  = tokenizer,
        train_dataset     = dataset,
        args          = SFTConfig(
            dataset_text_field          = "text",
            max_length                  = MAX_SEQ_LEN,
            output_dir                  = out_dir,
            num_train_epochs            = 3,
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 4,
            learning_rate               = 2e-4,
            fp16                        = False,    # not supported on MPS/CPU
            bf16                        = False,
            optim                       = "adamw_torch",
            logging_steps               = 1,
            save_strategy               = "no",
            report_to                   = "none",
        ),
    )

    print(f"  Fine-tuning on {len(corrections)} corrections …")
    model.train()
    trainer.train()
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"  Saved → {out_dir}")
    return model


# =============================================================================
# 4. Inference
# =============================================================================
@torch.no_grad()
def predict(model, tokenizer, user_input: str):
    model.eval()
    prompt    = apply_chat_template(tokenizer, user_input, assistant_output=None)
    inputs    = tokenizer(prompt, return_tensors="pt",
                          truncation=True, max_length=MAX_SEQ_LEN).to(DEVICE)
    in_len    = inputs["input_ids"].shape[1]

    out = model.generate(
        **inputs,
        max_new_tokens = 20,
        do_sample      = False,
        temperature    = 1.0,          # ignored when do_sample=False
        pad_token_id   = tokenizer.eos_token_id,
    )
    raw   = tokenizer.decode(out[0][in_len:], skip_special_tokens=True).strip()
    label = normalize_label(raw.splitlines()[0] if raw.splitlines() else raw)
    return label, raw


# =============================================================================
# 5. Corrections log
# =============================================================================
def load_corrections() -> list:
    if os.path.exists(CORRECTIONS):
        with open(CORRECTIONS, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_corrections(new_items: list):
    existing = load_corrections()
    combined = existing + new_items
    with open(CORRECTIONS, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"  Corrections log: {len(combined)} total.")


# =============================================================================
# 6. Test evaluation
# =============================================================================
def evaluate_on_test(model, tokenizer, test_examples):
    from sklearn.metrics import (
        accuracy_score, f1_score, recall_score, classification_report
    )
    print("\n" + "=" * 60)
    print(f"Evaluating on {len(test_examples)} test examples …")

    rows = []
    for i, sample in enumerate(test_examples, start=1):
        pred, raw = predict(model, tokenizer, sample["input"])
        actual    = sample["label"]
        correct   = pred == actual
        rows.append({
            "DATE":            sample["date"],
            "ACTUAL_LABEL":    actual,
            "PREDICTED_LABEL": pred,
            "CORRECT":         correct,
            "RAW_MODEL_OUTPUT":raw,
        })
        print(f"  [{i:3d}/{len(test_examples)}]  "
              f"actual={actual!r:30s}  pred={pred!r:30s}  {'✓' if correct else '✗'}")

    actual_l = [r["ACTUAL_LABEL"]    for r in rows]
    pred_l   = [r["PREDICTED_LABEL"] for r in rows]

    print("\n" + "=" * 60)
    print(f"Accuracy     : {accuracy_score(actual_l, pred_l):.4f}")
    print(f"Macro F1     : {f1_score(actual_l, pred_l, average='macro', zero_division=0):.4f}")
    print(f"Macro Recall : {recall_score(actual_l, pred_l, average='macro', zero_division=0):.4f}")
    print(classification_report(actual_l, pred_l,
                                 labels=VALID_LABELS, zero_division=0))

    pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)
    print(f"Results saved → {RESULTS_CSV}")


# =============================================================================
# Main — iterative correction loop
# =============================================================================
def run():
    train_ex, test_ex = load_examples()

    print("\n" + "=" * 60)
    print("Step 2: Loading model …")
    model, tokenizer = load_model_and_tokenizer()

    for it in range(1, ITERATIONS + 1):
        print(f"\n{'='*60}\nIteration {it}/{ITERATIONS}\n{'='*60}")
        new_corrections = []

        for i, sample in enumerate(train_ex, start=1):
            pred, raw = predict(model, tokenizer, sample["input"])
            ok        = pred == sample["label"]
            print(f"  [TRAIN {i}/{len(train_ex)}] {'✓' if ok else '✗'}  "
                  f"pred={pred!r}  expected={sample['label']!r}")
            if not ok:
                new_corrections.append({
                    "input": sample["input"],
                    "label": sample["label"],
                })

        print(f"\n  Corrections this iteration: {len(new_corrections)}")
        if new_corrections:
            save_corrections(new_corrections)

        all_c = load_corrections()
        model = run_sft(model, tokenizer, all_c, it)

    evaluate_on_test(model, tokenizer, test_ex)


if __name__ == "__main__":
    run()
