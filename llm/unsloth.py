# -*- coding: utf-8 -*-

# If running in Colab, uncomment:
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-deps trl peft accelerate bitsandbytes datasets pandas

import os
import json
import glob
import re
from pathlib import Path
import torch
import pandas as pd

from datasets import Dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer, SFTConfig


# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL      = "unsloth/Phi-3-mini-4k-instruct-bnb-4bit"
SCRIPT_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT    = SCRIPT_DIR.parent
SFT_MODEL_PATH  = str(SCRIPT_DIR / "outputs")
SFT_OUTPUT_DIR  = str(SCRIPT_DIR / "sft_outputs_amazon")
CORRECTIONS_LOG = str(SCRIPT_DIR / "corrections_amazon.json")
MAX_SEQ_LEN     = 2048

INPUT_CSV       = str(PROJECT_ROOT / "amazon_only_enriched.csv")
RESULTS_CSV     = str(PROJECT_ROOT / "amazon_unsloth_same_method_results.csv")

WINDOW_SIZE     = 5
TRAIN_YEAR      = 2017
TEST_YEAR       = 2018

SYSTEM_PROMPT = (
    "You are a stock decision classifier. "
    "Given historical Amazon stock observations, classify the next available Amazon move into exactly one of: "
    "must buy, maybe buy, don't buy, definitely don't buy. "
    "Reply with ONLY one label — nothing else."
)

VALID_LABELS = [
    "must buy",
    "maybe buy",
    "don't buy",
    "definitely don't buy",
]

if not torch.cuda.is_available():
    raise RuntimeError(
        "llm/unsloth.py requires a CUDA GPU environment (e.g. Colab/Linux GPU). "
        "For Mac/CPU workflows, use llm/run_unsloth.py instead."
    )


# ── Data Prep ─────────────────────────────────────────────────────────────────
def normalize_label(text: str) -> str:
    if not text:
        return "maybe buy"

    t = text.strip().lower()

    # Exact match first
    if t in VALID_LABELS:
        return t

    # Soft matching
    if "definitely" in t and "don't" in t:
        return "definitely don't buy"
    if "must" in t and "buy" in t:
        return "must buy"
    if "maybe" in t and "buy" in t:
        return "maybe buy"
    if "don't" in t and "buy" in t:
        return "don't buy"

    # fallback
    return "maybe buy"


def classify_return(r: float) -> str:
    if r >= 0.02:
        return "must buy"
    elif r >= 0.0:
        return "maybe buy"
    elif r > -0.02:
        return "don't buy"
    else:
        return "definitely don't buy"


def load_amazon_examples():
    df = pd.read_csv(INPUT_CSV)

    # Parse/clean
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

    numeric_cols = [
        "LAST_PRICE", "PX_VOLUME", "VOLATILITY_10D", "VOLATILITY_30D",
        "LSTM_POLARITY", "TEXTBLOB_POLARITY",
        "YF_OPEN", "YF_HIGH", "YF_LOW", "YF_CLOSE", "YF_VOLUME"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("DATE").reset_index(drop=True)

    # If TARGET_LABEL is missing, create it from NEXT_RETURN/NEXT_PRICE if possible
    if "TARGET_LABEL" not in df.columns:
        if "NEXT_RETURN" in df.columns:
            df["TARGET_LABEL"] = df["NEXT_RETURN"].apply(
                lambda x: classify_return(x) if pd.notna(x) else None
            )
        elif "NEXT_PRICE" in df.columns:
            df["NEXT_RETURN"] = (df["NEXT_PRICE"] - df["LAST_PRICE"]) / df["LAST_PRICE"]
            df["TARGET_LABEL"] = df["NEXT_RETURN"].apply(
                lambda x: classify_return(x) if pd.notna(x) else None
            )
        else:
            raise ValueError("amazon_only_enriched.csv must contain TARGET_LABEL, NEXT_RETURN, or NEXT_PRICE.")

    base_feature_cols = [
        "DATE",
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
    ]

    for col in base_feature_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    examples = []

    # Same rolling-window idea: previous 5 rows -> predict current row's TARGET_LABEL
    for i in range(WINDOW_SIZE, len(df)):
        window_df = df.iloc[i - WINDOW_SIZE:i].copy()
        target_row = df.iloc[i]

        if pd.isna(target_row["TARGET_LABEL"]):
            continue

        # Need complete history rows
        if window_df.drop(columns=["DATE"]).isna().any().any():
            continue

        user_prompt = build_user_prompt(window_df, target_row["DATE"])
        correct_label = normalize_label(str(target_row["TARGET_LABEL"]))

        examples.append({
            "date": pd.Timestamp(target_row["DATE"]).strftime("%Y-%m-%d"),
            "year": pd.Timestamp(target_row["DATE"]).year,
            "input": user_prompt,
            "label": correct_label,
        })

    train_examples = [e for e in examples if e["year"] == TRAIN_YEAR]
    test_examples  = [e for e in examples if e["year"] == TEST_YEAR]

    print(f"Total usable examples: {len(examples)}")
    print(f"Train examples ({TRAIN_YEAR}): {len(train_examples)}")
    print(f"Test examples  ({TEST_YEAR}): {len(test_examples)}")

    if train_examples:
        print("\nSample training example:")
        print(train_examples[0]["input"][:1200])
        print("LABEL:", train_examples[0]["label"])

    return train_examples, test_examples


def build_user_prompt(window_df: pd.DataFrame, target_date) -> str:
    lines = []
    lines.append(f"Target date to classify: {pd.Timestamp(target_date).strftime('%Y-%m-%d')}")
    lines.append("Use only the historical observations below.")
    lines.append("Historical Amazon observations:")

    for _, row in window_df.iterrows():
        lines.append(
            f"Date: {pd.Timestamp(row['DATE']).strftime('%Y-%m-%d')}, "
            f"LAST_PRICE: {row['LAST_PRICE']}, "
            f"PX_VOLUME: {row['PX_VOLUME']}, "
            f"VOLATILITY_10D: {row['VOLATILITY_10D']}, "
            f"VOLATILITY_30D: {row['VOLATILITY_30D']}, "
            f"LSTM_POLARITY: {row['LSTM_POLARITY']}, "
            f"TEXTBLOB_POLARITY: {row['TEXTBLOB_POLARITY']}, "
            f"YF_OPEN: {row['YF_OPEN']}, "
            f"YF_HIGH: {row['YF_HIGH']}, "
            f"YF_LOW: {row['YF_LOW']}, "
            f"YF_CLOSE: {row['YF_CLOSE']}, "
            f"YF_VOLUME: {row['YF_VOLUME']}"
        )

    lines.append("Classify the next available Amazon move.")
    return "\n".join(lines)


# ── Weight Resolution ─────────────────────────────────────────────────────────
def best_model_path() -> str:
    def has_weights(p):
        return os.path.exists(os.path.join(p, "adapter_config.json")) or \
               os.path.exists(os.path.join(p, "config.json"))

    sft_dirs = sorted(
        glob.glob(os.path.join(SFT_OUTPUT_DIR, "iteration_*")),
        key=lambda p: int(p.rsplit("_", 1)[-1])
    )
    for d in reversed(sft_dirs):
        if has_weights(d):
            print(f"Loading SFT weights: {d}")
            return d
    if has_weights(SFT_MODEL_PATH):
        print(f"Loading SFT weights: {SFT_MODEL_PATH}")
        return SFT_MODEL_PATH
    print(f"Loading base model: {BASE_MODEL}")
    return BASE_MODEL


# ── Model ─────────────────────────────────────────────────────────────────────
def load_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=best_model_path(),
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=64,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    return model, tokenizer


# ── Prompt / Training Text ────────────────────────────────────────────────────
def build_training_text(user_input: str, correct_answer: str) -> str:
    return (
        f"<|system|>\n{SYSTEM_PROMPT}\n"
        f"<|user|>\n{user_input}\n"
        f"<|assistant|>\n{correct_answer}<|endoftext|>"
    )


def build_prompt(user_input: str) -> str:
    return (
        f"<|system|>\n{SYSTEM_PROMPT}\n"
        f"<|user|>\n{user_input}\n"
        f"<|assistant|>\n"
    )


# ── Inference ─────────────────────────────────────────────────────────────────
def predict(model, tokenizer, user_input: str) -> str:
    FastLanguageModel.for_inference(model)
    inputs = tokenizer(build_prompt(user_input), return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]

    out = model.generate(
        **inputs,
        max_new_tokens=20,
        temperature=0.0,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    raw = tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()
    first_line = raw.splitlines()[0] if raw.splitlines() else raw
    label = normalize_label(first_line)
    return label, raw


# ── Validation ────────────────────────────────────────────────────────────────
def evaluate(predicted: str, expected: str):
    p = normalize_label(predicted)
    e = normalize_label(expected)
    if p == e:
        return True, "OK"
    return False, f"predicted={p} | expected={e}"


# ── Corrections Log ───────────────────────────────────────────────────────────
def load_corrections() -> list:
    if os.path.exists(CORRECTIONS_LOG):
        with open(CORRECTIONS_LOG, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def append_corrections(new: list):
    existing = load_corrections()
    combined = existing + new
    with open(CORRECTIONS_LOG, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"Corrections log: {len(combined)} total examples saved.")


# ── SFT Training ──────────────────────────────────────────────────────────────
def run_sft(model, tokenizer, corrections: list, iteration: int):
    if len(corrections) < 2:
        print(f"Only {len(corrections)} corrections — need at least 2. Skipping.")
        return model

    out_dir = os.path.join(SFT_OUTPUT_DIR, f"iteration_{iteration}")

    texts = [c["text"] for c in corrections]
    dataset = Dataset.from_dict({"text": texts})

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=MAX_SEQ_LEN,
            output_dir=out_dir,
            num_train_epochs=3,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            optim="adamw_8bit",
            logging_steps=1,
        ),
    )

    print("Training on corrections...")
    trainer.train()
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Weights saved → {out_dir}")
    return model


# ── Test Evaluation ───────────────────────────────────────────────────────────
def evaluate_on_test(model, tokenizer, test_examples):
    rows = []

    correct = 0
    for i, sample in enumerate(test_examples, start=1):
        pred_label, raw_output = predict(model, tokenizer, sample["input"])
        actual_label = normalize_label(sample["label"])
        is_correct = pred_label == actual_label
        correct += int(is_correct)

        rows.append({
            "DATE": sample["date"],
            "ACTUAL_LABEL": actual_label,
            "PREDICTED_LABEL": pred_label,
            "CORRECT": is_correct,
            "RAW_MODEL_OUTPUT": raw_output,
        })

        print(f"[TEST {i}/{len(test_examples)}] pred={pred_label} | actual={actual_label} | correct={is_correct}")

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_CSV, index=False)

    accuracy = correct / len(test_examples) if test_examples else 0.0
    print(f"\nTest accuracy: {accuracy:.4f}")
    print(f"Saved test results to: {RESULTS_CSV}")
    return results


# ── Main Loop ─────────────────────────────────────────────────────────────────
def run(iterations: int = 3, min_corrections: int = 2):
    train_examples, test_examples = load_amazon_examples()
    model, tokenizer = load_model()

    for iteration in range(1, iterations + 1):
        print(f"\n{'='*60}\nIteration {iteration}/{iterations}\n{'='*60}")
        new_corrections = []

        for i, sample in enumerate(train_examples, start=1):
            pred_label, raw_output = predict(model, tokenizer, sample["input"])
            ok, diff = evaluate(pred_label, sample["label"])
            status = "✅" if ok else "❌"

            print(
                f"[TRAIN {i}/{len(train_examples)}] {status} "
                f"predicted: {pred_label} | expected: {sample['label']}"
            )

            if not ok:
                print(f"  {diff}")
                new_corrections.append({
                    "text": build_training_text(sample["input"], sample["label"])
                })

        print(f"\nNew corrections this iteration: {len(new_corrections)}")
        if new_corrections:
            append_corrections(new_corrections)

        all_corrections = load_corrections()
        print(f"Total accumulated corrections: {len(all_corrections)}")

        if len(all_corrections) >= min_corrections:
            model = run_sft(model, tokenizer, all_corrections, iteration)
        else:
            print("Not enough corrections yet. Skipping training.")

    print(f"\nDone. Final weights in: {os.path.join(SFT_OUTPUT_DIR, f'iteration_{iterations}')}")
    print("\nRunning final evaluation on 2018 test set...")
    evaluate_on_test(model, tokenizer, test_examples)

    return model, tokenizer


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run(iterations=3, min_corrections=2)