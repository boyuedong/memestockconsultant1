# Meme Stock Consultant

A multi-model stock classification and recommendation pipeline that combines traditional ML (XGBoost / LightGBM), LLM fine-tuning, and a conversational chatbot UI to generate personalized portfolio recommendations.

---

## Project Structure

```
new3:11/
│
├── README.md
│
├── ── ML Pipeline ──────────────────────────────────────────────────────────
│
├── parsingcsv.py               Parse raw CSV → full_dataset_cleaned.csv
├── train_ensemble.py           XGBoost + LightGBM soft-voting (year-split baseline)
├── train_walkforward.py        XGB/LGBM walk-forward 70/30 split (best model)
├── train_allstocks.py          Train on all 101 stocks, test on Amazon 2018
├── backtest.py                 Backtest + classification metrics across all result files
├── geminipred.py               Gemini LLM zero-shot predictor (needs GEMINI_API_KEY)
├── run_pipeline.py             Orchestrator — run the full pipeline step by step
│
├── ── Data Files ───────────────────────────────────────────────────────────
│
├── full_dataset_cleaned.csv    Cleaned dataset for all 101 stocks
├── amazon_only.csv             Amazon-only rows extracted from full dataset
├── amazon_only_enriched.csv    Amazon rows merged with yfinance + engineered features
├── amazon_dates.csv            List of Amazon trading dates
├── amazon_ensemble_results.csv Results from train_ensemble.py
├── amazon_allstocks_results.csv Results from train_allstocks.py
│
├── ── LLM Fine-Tuning ──────────────────────────────────────────────────────
│
├── llm/
│   └── run_unsloth.py          Mac-compatible LoRA SFT fine-tuning (HuggingFace + PEFT)
│                               Reads amazon_only_enriched.csv, uses rolling 5-row windows
│                               Outputs: amazon_unsloth_results.csv + llm/sft_outputs/
│
├── ── Chatbot App ──────────────────────────────────────────────────────────
│
├── backend/                    FastAPI backend
│   ├── main.py                 App entry point (run with uvicorn)
│   ├── requirements.txt        Python dependencies
│   ├── routes/
│   │   └── chat.py             POST /api/chat/{start, message, reset}
│   └── services/
│       ├── profile_extractor.py   Deterministic rule-based field extraction
│       ├── conversation_manager.py Step sequencing + conversation orchestration
│       └── llm_service.py         OpenAI wrapper + rule-based fallback replies
│
├── frontend/                   React + TypeScript + Tailwind chatbot UI
│   ├── src/
│   │   ├── App.tsx             Split-panel layout (40% chat / 60% recommendation)
│   │   ├── types/chat.ts       TypeScript types
│   │   ├── services/chatApi.ts API client
│   │   ├── hooks/useChat.ts    Chat state management
│   │   └── components/
│   │       ├── ChatWindow.tsx        Left chat panel
│   │       ├── MessageBubble.tsx     User / assistant message bubbles
│   │       ├── ChatInput.tsx         Textarea + send button
│   │       ├── TypingIndicator.tsx   Animated typing dots
│   │       ├── ProfileSidebar.tsx    Filled-field progress strip
│   │       └── RecommendationPanel.tsx  Right-side results panel
│
└── archive/                    Superseded scripts (not part of active pipeline)
    ├── amazon.py
    ├── amazonpred.py
    └── checkdates.py
```

---

## Quick Start

### 1 — Run the ML Pipeline

```bash
cd "/Users/boyuedong/Desktop/new3:11"

# Parse raw data
python3 parsingcsv.py

# Train models (pick one, or run all via orchestrator)
python3 train_ensemble.py        # baseline
python3 train_walkforward.py     # recommended
python3 train_allstocks.py       # cross-stock generalisation

# Backtest all results
python3 backtest.py

# Or run everything in order
python3 run_pipeline.py
```

### 2 — Run the LLM Fine-Tuning (Mac / CPU)

```bash
# One-time setup
pip3 install transformers peft trl accelerate datasets

python3 llm/run_unsloth.py
# Outputs: amazon_unsloth_results.csv
# Saved model: llm/sft_outputs/
```

> Uses `Qwen/Qwen2.5-0.5B-Instruct` with LoRA. Runs on Apple Silicon MPS or CPU.
> For GPU / Colab, swap the base model to an Unsloth 4-bit variant.

### 3 — Run the Chatbot App

**Backend** (Terminal 1):
```bash
cd "/Users/boyuedong/Desktop/new3:11"
pip3 install fastapi uvicorn python-dotenv

uvicorn backend.main:app --reload --port 8000
```

**Frontend** (Terminal 2):
```bash
cd "/Users/boyuedong/Desktop/new3:11/frontend"
npm install
npm run dev
# → http://localhost:5173
```

---

## Chatbot Features

The chatbot collects an investor profile through a guided multi-step conversation:

| Step | Field collected |
|------|----------------|
| 1 | Investment time horizon (1M / 3M / 6M / 1Y) |
| 2 | Risk tolerance (low / medium / high) |
| 3 | Objective (growth / stability / balanced) |
| 4 | Sector preferences (optional) |

Once all required fields are collected, it generates a personalized recommendation comparing **Social Buzz Stocks** vs **Magnificent 7**.

Use unsloth for GPU/google colab, retrieve API key for MAC

```bash
export key=
```

To swap for a different LLM, edit `backend/services/llm_service.py`

## Dependencies

### Python
```
pandas, numpy, scikit-learn, xgboost, lightgbm, yfinance
fastapi, uvicorn, pydantic, python-dotenv
transformers, peft, trl, accelerate, datasets
```

### Node.js (frontend)
```
react, react-dom, typescript, vite
tailwindcss, postcss, autoprefixer
uuid
```

## Streamlit Deployment (Optional)

Run a lightweight Streamlit UI backed by your pipeline outputs:

```bash
cd "/Users/boyuedong/Desktop/new3:11"
python3 -m pip install -r requirements_streamlit.txt
streamlit run streamlit_app.py
```

The app uses:

- `selected_stocks_model_summary.csv`
- `selected_stocks_walkforward_results.csv`
- `backend/services/recommendation_service.py`

If you use local Ollama, keep your `.env` values set (for example `LLM_PROVIDER=ollama`).
