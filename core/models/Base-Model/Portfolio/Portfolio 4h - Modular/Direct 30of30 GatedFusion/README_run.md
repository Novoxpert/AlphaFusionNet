# Joint Portfolio — Direct Weights

This repository provides an end‑to‑end pipeline that **predicts portfolio weights directly** (instead of predicting returns/vol/correlation first). It supports:
1) dataset preparation  
2) training a direct‑weights model  
3) inference to generate weights on the test set  
4) backtesting with stop‑loss / take‑profit  
5) (optional) **real‑time** weight generation from rolling OHLCV + news

---

## 📁 Repository Layout (exact)

```
joint_portfolio_direct_weights
└── joint_portfolio_direct_weights/
    ├── data/
    │   ├── ohlcv/
    │   ├── outputs/
    │   │   └── model_weights.pt        # example checkpoint
    │   ├── processed/                  # ← put processed dataset here (see links below)
    │   └── raw_news/
    ├── lib/
    │   ├── backtest.py
    │   ├── backtest_weights.py         # backtest helpers tailored for direct weights
    │   ├── dataset.py
    │   ├── features.py
    │   ├── loss_weights.py             # loss functions for direct weights
    │   ├── market.py
    │   ├── model.py
    │   ├── news.py
    │   ├── train.py
    │   └── utils.py
    ├── config.py
    ├── README.md
    ├── requirements.txt
    ├── run_backtest.py
    ├── run_infer_and_portfolio.py
    ├── run_prepare.py
    ├── run_realtime_weights.py
    └── run_train.py
```
> Any folders missing on your machine will be created by the scripts if needed.

---

## 🔗 Required Downloads

### Processed dataset (offline pipeline)
Place the downloaded file(s) under `joint_portfolio_direct_weights/joint_portfolio_direct_weights/data/processed/`:

- https://drive.google.com/file/d/1YQoIFOFVKHVrKT9MoBtPVKXU7J1toTZr/view?usp=sharing

### LLM models (for news embeddings, etc.)
Place downloaded models under a folder you will reference (e.g. `./llm_models/` at repo root):

- https://drive.google.com/drive/folders/1htASoZVoRYkjzl8Svsi8fxc-x7eUEtCO?usp=sharing

### Real‑time sample data (optional)
Download this folder and place it **in the repository root (same level as `run_*.py`)**. You can name it `realtime_samples/` (recommended) or keep the original name.

- https://drive.google.com/drive/folders/1KpuU3krGtRmwc3P5HP6BLoPtWB4SEn1x?usp=sharing

> Expected layout:
> ```
> realtime_samples/
> ├── ohlcv/     # minute bars
> └── news/      # minute‑aligned news
> ```

---

## ⚙️ Setup

```bash
cd joint_portfolio_direct_weights/joint_portfolio_direct_weights

# (optional) create a virtual environment
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows (PowerShell):
# .\.venv\Scripts\Activate.ps1

# install exact dependencies
pip install -r requirements.txt
```

---

## 🚀 Offline Pipeline (Direct Weights)

### 1) Prepare dataset
Build train/val/test splits and cache tensors/frames used by the training code.
```bash
python run_prepare.py
```
Outputs: `data/processed/train_*`, `val_*`, `test_*` (formats depend on your config).

### 2) Train model (direct weights)
Train the model that **outputs portfolio weights** directly and save checkpoints into `data/outputs/`.
```bash
python run_train.py
```
Result: a checkpoint such as `data/outputs/model_weights.pt`.

### 3) Inference → Weights (+ optional portfolio constraints)
Generate weights on the test set. Depending on config, weights may be long‑only, leverage‑bounded, top‑k, or unconstrained with later normalization.
```bash
python run_infer_and_portfolio.py
```

### 4) Backtest with Stop‑Loss / Take‑Profit
Apply execution rules over predicted weights (supports fixed or volatility‑adjusted thresholds).
```bash
python run_backtest.py
```
---

## ⚡ Real‑Time (Direct Weights)

1) Ensure the real‑time sample folder is at repo root (e.g., `./realtime_samples`).  
2) Run the real‑time script that **produces weights directly** from the latest OHLCV + news:
```bash
python run_realtime_weights.py
```
This reads the latest OHLCV + news, builds the most recent window, embeds news via your LLM models, runs a forward pass, and writes current portfolio weights.

---

## 🧩 Script Cheat‑Sheet

- **`run_prepare.py`** — dataset preparation utilities (splits, caching).  
- **`run_train.py`** — train the direct‑weights model; saves `model_weights.pt`.  
- **`run_infer_and_portfolio.py`** — generate weights for the test set (optionally apply constraints).  
- **`run_backtest.py`** — stop‑loss / take‑profit application and backtest metrics; uses helpers in `lib/backtest_weights.py`.  
- **`run_realtime_weights.py`** — rolling online **direct weight** inference.  
- **`lib/*.py`** — internal modules for datasets, models, features, news embeddings, training loops, utilities, and backtesting specialized for direct weights.  
- **`config.py`** — central configuration / argument helpers used by the scripts.

---

## ✅ Notes / Differences vs. Pointwise (μ/σ/ρ) Pipeline

- This repo **does not compute μ, σ, ρ first**; instead the model **outputs weights directly**.  
- Backtesting helpers include `backtest_weights.py` and loss functions are in `loss_weights.py`, both tailored for weight supervision.  
---
