# Joint Portfolio Pointwise

This repository provides an end‑to‑end pipeline to:
1) prepare datasets  
2) train forecasting models  
3) run inference and build portfolios on the test set  
4) backtest with stop‑loss / take‑profit  
5) (optional) run a **real‑time** pointwise portfolio from rolling OHLCV + news

---

## 📁 Repository Layout (exact)

```
joint_portfolio_pointwise2
├── data
│   ├── processed/                 # ← put processed dataset here (see links below)
│   ├── ohlcv/
│   ├── raw_news/
│   └── outputs/
│       └── model_pointwise.pt # example checkpoint
├── lib/
│   ├── backtest.py
│   ├── dataset.py
│   ├── features.py
│   ├── inference.py
│   ├── loss.py
│   ├── market.py
│   ├── model.py
│   ├── news.py
│   ├── portfolio.py
│   └── utils.py
├── config.py
├── requirements.txt
├── README.md                      # (you can replace with this improved one)
├── run_prepare.py
├── run_train.py
├── run_infer_and_portfolio.py
├── run_backtest.py
└── run_realtime_pointwise.py
```

> Any folders missing on your machine will be created by the scripts if needed.

---

## 🔗 Required Downloads

### Processed dataset (offline pipeline)
Place the downloaded file(s) under `joint_portfolio_pointwise2/data/processed/`:

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
cd joint_portfolio_pointwise2

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

## 🚀 Offline Pipeline

### 1) Prepare dataset
Build train/val/test splits and cache tensors/frames used by the training code.
```bash
python run_prepare.py   
```
Typical outputs: `data/processed/train_*`, `val_*`, `test_*` (formats depend on your config).

### 2) Train models
Train pointwise model(s) and save checkpoints into `data/outputs/` or `outputs/` depending on config.
```bash
python run_train.py   
```
Result: a checkpoint such as `data/outputs/model_pointwise.pt`.

### 3) Inference & Portfolio (test set)
Produce predictions (e.g., μ/σ/ρ or equivalent) and compute portfolio weights on the test window.
```bash
python run_infer_and_portfolio.py   

```

### 4) Backtest with Stop‑Loss / Take‑Profit
Apply execution rules on top of predicted weights.
```bash
python run_backtest.py  
```

---

## ⚡ Real‑Time (Pointwise)

1) Ensure the real‑time sample folder is at repo root (e.g., `./data_realtime`).  
2) Run the streaming/rolling inference script:
```bash
python run_realtime_pointwise.py  
```
This reads the latest OHLCV + news, builds the most recent window, embeds news via your LLM models, runs a forward pass, and writes current portfolio weights.

---

## 🧩 Script Cheat‑Sheet

- **`run_prepare.py`** — dataset preparation utilities (splits, caching).  
- **`run_train.py`** — model training loop, logging, checkpointing.  
- **`run_infer_and_portfolio.py`** — inference on test + portfolio optimization.  
- **`run_backtest.py`** — stop‑loss / take‑profit application and metrics.  
- **`run_realtime_pointwise.py`** — rolling online weights for real‑time portfolio.  
- **`lib/*.py`** — internal modules for datasets, models, losses, features, news embeddings, portfolio construction, backtesting, and utilities.  
- **`config.py`** — central configuration / arguments helpers used by the scripts.

---
