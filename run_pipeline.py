"""
Amazon Stock Classification — Pipeline
=======================================
Run all steps in order, or pass a step number:

    python3 run_pipeline.py          # run steps 1–4 (skips step 0 and 5)
    python3 run_pipeline.py 2        # run only step 2

─────────────────────────────────────────────────────────────────────────────
STEP 0  parsingcsv.py
        Raw CSV → full_dataset_cleaned.csv
        Only needed once to rebuild the cleaned dataset from the source file.

STEP 1  train_ensemble.py
        XGBoost + LightGBM — year-based split (baseline, weak)
        Train: Amazon 2017 (~10 rows)  |  Test: Amazon 2018
        Output: amazon_ensemble_results.csv

STEP 2  train_walkforward.py        ← BEST classical ML approach
        XGBoost + LightGBM — chronological 70/30 split on Amazon only
        Train: first 70% of Amazon rows  |  Test: last 30%
        Output: amazon_walkforward_results.csv

STEP 3  train_allstocks.py
        XGBoost + LightGBM — train on ALL 101 stocks 2017, test Amazon 2018
        Full yfinance + 39 engineered features. Cross-asset comparison.
        Output: amazon_allstocks_results.csv

STEP 4  backtest.py
        Loads all *_results.csv files and prints:
          • Accuracy, macro F1, confusion matrix vs 25% random baseline
          • Trading simulation ($10k) vs buy-and-hold
          • Signal quality: avg return per predicted label
        Output: printed report

STEP 5  geminipred.py               ← LLM zero-shot (needs API key)
        Sends numerical features to Gemini-2.5-Flash as a prompt.
        Requires: export GEMINI_API_KEY='your_key_here'
        Output: amazon_gemini_results.csv

─────────────────────────────────────────────────────────────────────────────
File map
─────────────────────────────────────────────────────────────────────────────
  Scripts (active)
    parsingcsv.py           Step 0 — parse raw CSV into cleaned dataset
    train_ensemble.py       Step 1 — XGB/LGBM, year split
    train_walkforward.py    Step 2 — XGB/LGBM, walk-forward 70/30
    train_allstocks.py      Step 3 — XGB/LGBM, all stocks
    backtest.py             Step 4 — evaluate + trading simulation
    geminipred.py           Step 5 — Gemini LLM zero-shot
    run_pipeline.py         This file — orchestrates all steps

  Data
    full_dataset_cleaned.csv        Source: 862k tweet rows, 101 stocks
    amazon_only.csv                 Filtered: Amazon rows only
    amazon_only_enriched.csv        Amazon + yfinance + 39 engineered features
    amazon_dates.csv                Unique Amazon trading dates

  Results
    amazon_ensemble_results.csv     Step 1 output
    amazon_walkforward_results.csv  Step 2 output
    amazon_allstocks_results.csv    Step 3 output
    amazon_gemini_results.csv       Step 5 output

  archive/
    amazon.py                       Old — simple Amazon filter (superseded)
    checkdates.py                   Old — date inspection utility (superseded)
    amazonpred.py                   Old — basic prediction script (superseded)
─────────────────────────────────────────────────────────────────────────────
"""

import subprocess
import sys
import os

BASE = "/Users/boyuedong/Desktop/new3:11"

STEPS = {
    0: ("parsingcsv.py",       "Parse raw CSV → full_dataset_cleaned.csv"),
    1: ("train_ensemble.py",   "XGB/LGBM year-split (baseline)"),
    2: ("train_walkforward.py","XGB/LGBM walk-forward 70/30 (best)"),
    3: ("train_allstocks.py",  "XGB/LGBM all-stocks → Amazon test"),
    4: ("backtest.py",         "Backtest + compare all results"),
    5: ("geminipred.py",       "Gemini LLM zero-shot (needs GEMINI_API_KEY)"),
}

def run_step(n):
    script, desc = STEPS[n]
    path = os.path.join(BASE, script)
    print()
    print("=" * 65)
    print(f"  STEP {n}: {desc}")
    print(f"  Script : {script}")
    print("=" * 65)
    if not os.path.exists(path):
        print(f"  SKIPPED — {script} not found")
        return
    result = subprocess.run(["python3", path], cwd=BASE)
    if result.returncode != 0:
        print(f"\n  Step {n} exited with code {result.returncode}")

def print_status():
    result_files = [
        ("amazon_ensemble_results.csv",    "Step 1"),
        ("amazon_walkforward_results.csv", "Step 2"),
        ("amazon_allstocks_results.csv",   "Step 3"),
        ("amazon_gemini_results.csv",      "Step 5"),
    ]
    print()
    print("=" * 65)
    print("  Results status")
    print("=" * 65)
    for fname, step in result_files:
        exists = "✓" if os.path.exists(os.path.join(BASE, fname)) else "✗"
        print(f"  {exists}  {fname:<40} ({step})")
    print()

if __name__ == "__main__":
    print(__doc__)
    print_status()

    if len(sys.argv) > 1:
        try:
            run_step(int(sys.argv[1]))
        except (ValueError, KeyError):
            print(f"Unknown step: {sys.argv[1]}. Choose 0–5.")
    else:
        for n in [1, 2, 3, 4]:   # skip step 0 (one-time) and step 5 (needs API key)
            run_step(n)
