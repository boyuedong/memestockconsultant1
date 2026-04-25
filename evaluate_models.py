"""
evaluate_models.py — compute and compare precision, recall, F1, accuracy
across all model result files in this project.

Usage:
    python3 evaluate_models.py

Add any new results CSV to RESULT_FILES below and it will be included automatically.
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

BASE = "/Users/boyuedong/Desktop/new3:11"

# ── Register result files here ─────────────────────────────────────────────────
RESULT_FILES = {
    "XGB/LGBM  year-split    ": os.path.join(BASE, "amazon_ensemble_results.csv"),
    "XGB/LGBM  all-stocks    ": os.path.join(BASE, "amazon_allstocks_results.csv"),
    "XGB/LGBM  walk-forward  ": os.path.join(BASE, "amazon_walkforward_results.csv"),
    "LLM  SFT  (Qwen2.5 LoRA)": os.path.join(BASE, "amazon_unsloth_results.csv"),
    "LLM  Gemini zero-shot   ": os.path.join(BASE, "amazon_gemini_results.csv"),
}

VALID_LABELS = ["must buy", "maybe buy", "don't buy", "definitely don't buy"]

LABEL_COL_OPTIONS     = ["TARGET_LABEL",    "ACTUAL_LABEL"]
PRED_COL_OPTIONS      = ["PREDICTED_LABEL", "PRED_LABEL"]


def load_result(path: str):
    df = pd.read_csv(path)
    true_col = next((c for c in LABEL_COL_OPTIONS if c in df.columns), None)
    pred_col = next((c for c in PRED_COL_OPTIONS  if c in df.columns), None)
    if not true_col or not pred_col:
        raise ValueError(f"Cannot find label/prediction columns in {path}. "
                         f"Columns: {list(df.columns)}")
    y_true = df[true_col].astype(str).str.strip().str.lower()
    y_pred = df[pred_col].astype(str).str.strip().str.lower()
    return y_true, y_pred


def metrics(y_true, y_pred):
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall":    recall_score(y_true, y_pred,    average="macro", zero_division=0),
        "f1":        f1_score(y_true, y_pred,        average="macro", zero_division=0),
    }


def print_separator(char="─", width=70):
    print(char * width)


def run():
    print("\n" + "=" * 70)
    print("  MODEL EVALUATION — Precision / Recall / F1 / Accuracy")
    print("  Amazon stock movement classification  (4-class)")
    print("=" * 70)

    summary_rows = []

    for model_name, path in RESULT_FILES.items():
        if not os.path.exists(path):
            print(f"\n  {model_name.strip():<30}  [results file not found — skipping]")
            print(f"  Expected: {path}")
            continue

        try:
            y_true, y_pred = load_result(path)
        except Exception as e:
            print(f"\n  {model_name.strip():<30}  [error loading: {e}]")
            continue

        m = metrics(y_true, y_pred)
        n_test = len(y_true)

        print(f"\n{'─'*70}")
        print(f"  Model : {model_name.strip()}")
        print(f"  Test rows : {n_test}")
        print(f"{'─'*70}")
        print(f"  {'Metric':<20} {'Score':>8}   {'Bar':}")
        print(f"  {'──────':<20} {'─────':>8}")

        for metric_name, value in m.items():
            bar = "█" * int(value * 30)
            print(f"  {metric_name:<20} {value:>7.1%}   {bar}")

        print()
        print("  Per-class breakdown:")
        print(classification_report(
            y_true, y_pred,
            labels=VALID_LABELS,
            zero_division=0,
            digits=3,
        ))

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=VALID_LABELS)
        labels_short = ["must buy", "maybe buy", "don't buy", "def. don't buy"]
        print("  Confusion matrix  (rows=actual, cols=predicted):")
        header = f"  {'':>17}" + "".join(f"{l[:10]:>12}" for l in labels_short)
        print(header)
        for i, row_label in enumerate(labels_short):
            row = "  " + f"{row_label[:17]:>17}" + "".join(f"{cm[i,j]:>12}" for j in range(4))
            print(row)

        summary_rows.append({
            "Model":     model_name.strip(),
            "N_test":    n_test,
            "Accuracy":  round(m["accuracy"],  4),
            "Precision": round(m["precision"], 4),
            "Recall":    round(m["recall"],    4),
            "F1":        round(m["f1"],        4),
        })

    # ── Summary table ──────────────────────────────────────────────────────────
    if len(summary_rows) > 1:
        print("\n" + "=" * 70)
        print("  SUMMARY TABLE")
        print("=" * 70)

        df_sum = pd.DataFrame(summary_rows).sort_values("Accuracy", ascending=False)

        header = (f"  {'Model':<30} {'N':>5}  {'Accuracy':>9}  "
                  f"{'Precision':>9}  {'Recall':>9}  {'F1':>9}")
        print(header)
        print("  " + "─" * 68)

        best_acc = df_sum["Accuracy"].max()
        for _, r in df_sum.iterrows():
            star = " ◄ best" if r["Accuracy"] == best_acc else ""
            print(f"  {r['Model']:<30} {r['N_test']:>5}  "
                  f"{r['Accuracy']:>8.1%}  {r['Precision']:>8.1%}  "
                  f"{r['Recall']:>8.1%}  {r['F1']:>8.1%}{star}")

        # Save summary CSV
        out_path = os.path.join(BASE, "model_comparison.csv")
        df_sum.to_csv(out_path, index=False)
        print(f"\n  Summary saved → {out_path}")

    # ── Random baseline ────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  Random baseline (4 equal classes) :  25.0% accuracy")
    print("─" * 70 + "\n")


if __name__ == "__main__":
    run()
