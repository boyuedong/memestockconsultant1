# Deliverable: Multi-Stock Recommendation Pipeline + Chat Frontend

## 1) Introduction

This project builds a stock recommendation assistant that combines:

- classical ML classification (XGBoost + LightGBM soft-vote ensemble),
- multi-stock evaluation outputs,
- and a chatbot frontend that translates model outputs into user-facing recommendations.

The core goal is not to predict exact future prices, but to classify short-term movement into actionable labels and use those signals to support profile-aware recommendations.

## 2) Methodology

### Technique used

Primary training script:

- `train_selected_stocks.py`

Per-stock modeling pipeline:

1. Load cleaned records from `full_dataset_cleaned.csv`
2. Merge stock rows with yfinance OHLCV data
3. Engineer technical + sentiment features
4. Create 4-class target label based on next return:
   - `must buy`
   - `maybe buy`
   - `don't buy`
   - `definitely don't buy`
5. Train XGBoost and LightGBM
6. Blend probabilities with soft voting
7. Save per-row predictions and per-stock summary metrics

### Assumptions made

- Chronological split approximates forward testing:
  - first 70% train, last 30% test
- Stocks with stronger historical out-of-sample metrics are better recommendation candidates
- LLM is used to explain and format recommendations, not replace core model scoring logic
- Current selected-stock runs are based on available historical window in the cleaned dataset (mostly 2018-tail evaluation period in outputs)

## 3) Data Scope

- Source file: `full_dataset_cleaned.csv`
- Total stocks available: 101
- Production subset selected for stable runs: 20 stocks (`selected_stocks_20.csv`)

Typical per-stock run profile:

- raw rows: 121
- model-ready rows: ~105-106
- training rows: ~73-74
- testing rows: 32

## 4) Output Artifacts

- `selected_stocks_20.csv`: stock universe used
- `selected_stocks_model_summary.csv`: per-stock `accuracy`, `macro_f1`, `macro_recall`, row counts
- `selected_stocks_walkforward_results.csv`: row-level actual/predicted labels and correctness
- `selected_stocks_failures.csv`: failure log

## 5) Quantitative Results (Current 20-Stock Run)

Across all 20 selected stocks:

- Mean accuracy: **45.31%**
- Median accuracy: **42.18%**
- Mean macro F1: **0.2171**
- Mean macro recall: **0.2578**
- Total test rows: **640**
- Overall row-level accuracy: **45.31%**

Best and worst by accuracy:

- Best: **COST (0.6250)**
- Worst: **AAPL (0.3125)**

Top 5 by accuracy:

1. COST - 0.6250
2. MSFT - 0.5938
3. TM - 0.5938
4. VZ - 0.5625
5. ADBE - 0.5312

## 6) Before vs After Large-Dataset Expansion

This project started with narrower/single-stock style baselines and expanded toward a broader multi-stock run.

### Before (baseline comparisons from `model_comparison.csv`)

- XGB/LGBM year-split: accuracy **0.1979**
- XGB/LGBM all-stocks (older variant): accuracy **0.1146**
- XGB/LGBM walk-forward (small setup): accuracy **0.3611**

### After (current selected 20-stock production run)

- Mean per-stock accuracy: **0.4531**
- Stable 20-stock execution with no failures in final run
- Row-level prediction outputs available for all selected tickers

Interpretation:

- Expanding to a curated, quality-screened multi-stock set with updated pipeline logic improved practical performance and robustness relative to earlier baseline runs.

## 7) Chatbot Integration and Concerns

### Current chatbot behavior

- Existing frontend (`frontend/`) is reused.
- Backend endpoint `POST /api/chat/recommend` ranks stocks using saved ML metrics.
- LLM provider can be `ollama`, `qwen`, `auto`, or deterministic `fallback`.

### Important concern: time-period assumption

The chatbot can appear to assume a 2017-2018 style backtest context because recommendation evidence is derived from historical output files (and many test rows are from that period). This means:

- recommendations are informed by historical reliability,
- not by live, real-time market features at request time.

### Other concerns

- The recommendation endpoint uses persisted outputs, not online inference per user turn.
- Some chatbot prompts were initially repetitive due to fallback clarification logic (this was patched).
- LLM provider failures can silently degrade to fallback if no visibility flag is exposed.

## 8) What Did Not Work Initially and What Changed

### What did not work

- Path mismatch issues in older scripts (`new3/11` vs `new3:11`)
- Missing/blocked yfinance calls under restricted network mode
- Frontend `vite` command resolution issue (`vite: command not found`)
- Qwen API access failures (`AccessDenied.Unpurchased`)
- Backend process conflicts on port 8000 causing frontend connection errors
- Clarification reply loop in conversation flow

### What changed

- Refactored script paths to robust project-relative handling
- Built and ran `train_selected_stocks.py` for 20-stock execution
- Fixed class-encoding and datetime merge edge cases
- Added recommendation backend service with provider routing and fallback
- Integrated frontend with real recommendation endpoint
- Added local Ollama provider path and set `.env` for local free model usage
- Restarted server processes and validated health and endpoint responses
- Patched clarification behavior in chat manager/LLM service

## 9) Discussion: What Could Be Done With More Data

If larger and newer data windows are added, the following steps should improve quality:

1. Expand temporal coverage (more recent years) to reduce historical regime bias
2. Retrain per stock with rolling-origin cross-validation (multiple walk-forward folds)
3. Incorporate recency features into ranking:
   - recent hit-rate,
   - recent average return by predicted class,
   - signal stability
4. Add confidence calibration and abstention thresholds
5. Compare static historical scoring vs dynamic recency-aware scoring
6. Add periodic retraining pipeline (scheduled batch jobs)

Expected impact:

- better generalization,
- stronger alignment with current market conditions,
- and less risk of recommendations overfitting older periods.

## 10) Current Limitations

- Saved-output scoring is still historical, not fully live
- Macro F1 remains modest on several tickers (class imbalance and regime shocks)
- LLM prompt quality influences explanation quality, not core model skill
- Model-source visibility is not yet exposed to UI by default

## 11) Deliverable Artifacts Checklist

- `train_selected_stocks.py` (multi-stock training script)
- `selected_stocks_20.csv` (final 20 stock set)
- `selected_stocks_model_summary.csv` (per-stock performance)
- `selected_stocks_walkforward_results.csv` (row-level predictions)
- `selected_stocks_failures.csv` (run failure log)
- `backend/services/recommendation_service.py` (recommendation engine + provider routing)
- `frontend/` integration updated to call backend recommendation endpoint

