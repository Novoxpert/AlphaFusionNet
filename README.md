# AlphaFusionNet: LLM-Driven Neural–Graph Portfolio Engine

This repository provides an end-to-end pipeline for portfolio modeling that integrates three core components: **NeuralFusionCore**, which directly predicts portfolio weights; **ChronoBridge**, which generates time-aligned fused embeddings; **NetWeaver**, which leverages these embeddings for downstream portfolio optimization and analysis; and **AlphaFusionNet**, which is a Hybrid Portfolio Decision-Making Engine.

The NeuralFusionCore ingests the latest OHLCV data and financial news, constructs the most recent temporal window, encodes news using a large language model (LLM), performs a forward inference pass, and outputs the current portfolio weights. In addition, ChronoBridge extracts trained fused embeddings for each asset and stores them in MongoDB for subsequent retrieval and analysis. NetWeaver is a comprehensive stock prediction system leveraging Graph Attention Networks (GAT) to model relationships between stocks—both within industries (intra-sector) and across industries (inter-sector). It combines sequential encoding with graph-based learning to predict profitable stock movements and recommend top-K stocks. AlphaFusionNet provides the main service for AlphaFusionNet, a hybrid portfolio decision-making engine that fuses outputs from two intelligent portfolio modules — NeuralFusionCore and NetWeaver — and combines both quantitative and qualitative reasoning to produce final portfolio weights.

It supports:
1) Data ingesting from databases using redis
2) Dataset preparation  
3) Training a direct‑weights model  
4) Real‑time weight generation 
5) save predictions to MongoDB as NeuralFusionCore output
6) FastAPI service to serve latest portfolio predictions for dashboard
7) Generating realtime Fused Embedding of (ohlcv + news) as ChronoBridge output
8) FastAPI service to serve latest ChronoBridge outputs for Netweaver
9) Combining sequential encoding with graph-based learning and predicting profitable stock movements and recommending top-K stocks as NetWeaver output.
10) Fusing outputs from NeuralFusionCore and NetWeaver 
11) Combining both quantitative and qualitative reasoning and producing final portfolio weights
12) Full CI-quality testing and unit testing suite that validates API, model, data pipeline, and persistence.

## Table of Contents

1. [Overview](#overview)  
2. [Key Features](#key-features)  
3. [Repository Layout](#repository-layout)  
4. [Required Downloads](#required-downloads)  
5. [Setup](#setup)  
6. [Pipeline](#pipeline)  
   - [Run Scheduler](#1-run-scheduler)  
   - [Run NeuralFusionCore API Service](#2-run-neuralfusioncore-api-service)  
   - [Run ChronoBridge](#3-run-chronobridge)  
   - [Run ChronoBridge API Service](#4-run-chronobridge-api-service)  
   - [Run Tests](#5-run-tests)  
7. [Script Cheat-Sheet](#script-cheat-sheet)  
8. [Dependencies](#dependencies)  
9. [Appendix](#appendix)  
   - [Upstream Repositories](#upstream-repositories)   
10. [Authors & Citation](#authors--citation)  
11. [Support](#support)
---

## Repository Layout

```
AlphaFusionNet
│
│
├── apps/
│    ├── ChronoBridge/
│    │       ├── scripts/
│    │       │     ├── chronobridge_api_service.py 
│    │       │     └── chronobridge_service.py
│    │       └──apps/
│    │              └──NeuralFusionCore
│    │
│    ├── NeuralFusionCore/
│    │     ├── data/
│    │     │   ├── outputs/
│    │     │   │   └── model_weights.pt        
│    │     │   └── processed/
│    │     │       └── show_files.py                   
│    │     │   
│    │     ├── lib/
│    │     │   ├── backtest.py
│    │     │   ├── backtest_weights.py        
│    │     │   ├── dataset.py
│    │     │   ├── features.py
│    │     │   ├── loss_weights.py            
│    │     │   ├── market.py
│    │     │   ├── model.py
│    │     │   ├── news.py
│    │     │   ├── redis_utils.py
│    │     │   ├── train.py
│    │     │   └── utils.py
│    │     ├──_init__.py
│    │     ├── README.md
│    │     ├── requirements.txt
│    │     ├── config.py
│    │     └── scripts/
│    │           ├── data_ingest_service.py
│    │           ├── features_service.py
│    │           ├── train_service.py
│    │           ├── finetune_service.py
│    │           ├── prediction_service.py 
│    │           └── api_service.py
│    │
│    └── NetWeaver/
│          ├── src/
│          │   ├── data_pipeline.py         # Main data processing pipeline
│          │   ├── train.py                 # Model training with analysis
│          │   ├── run_analysis.py          # Standalone analysis script
│          │   ├── parse_arg.py             # Argument parsing utilities
│          │   └── utils/
│          │       ├── data_utils.py        # Data management utilities
│          │       ├── metrics_utils.py     # Evaluation metrics
│          │       ├── analysis_utils.py    # Analysis functions
│          │       └── training_logger.py   # Logging utilities
│          ├── data/
│          │   ├── raw/
│          │   │   ├── SP500_dataset/
│          │   │   └── *.npy (graph files)
│          │   └── processed/
│          │       └── model_data.pickle
│          └── results/
│              ├── saved_models/            # Model checkpoints
│              ├── train/                   # Training logs
│              └── analysis/                # Analysis outputs
│ 
│ 
│
├── scripts/
│    ├──_init__.py
│    ├──alphafusionnet_api_service.py
│    └── alphafusionnet_service.py
│
├── src/
│     ├── contoller.py
│     ├── llm_alphafusionnet.py
│     └──quant_alphafusionnet.py
│
├── config/
│     └──AFN_config.yml
│
├── logs/
│     └──alphafusionnet_service.log
│
├── tests/
│    ├──_init__.py
│    ├── test_alphafusionnet_api_service.py
│    ├── test_alphafusionnet_service.py
│    ├── test_api_service.py
│    ├── test_chronobridge_api_service.py
│    ├── test_chronobridge_service.py
│    ├── test_data_ingest_service.py
│    ├── test_features_service.py
│    ├── test_finetune_service.py
│    ├── test_health_clickhouse.py
│    ├── test_health_mongo.py
│    ├── test_health_redis.py
│    ├── test_model.py
│    ├── test_prediction_service.py
│    ├── test_redis_clickhouse_integration.py
│    ├── test_redis_mongo_integration.py
│    └── test_train_service.py
│
├── README.md
├── scheduler/
│     ├── schaduler.py
│     ├── tasks.py
│     ├── test_celery_func.py
│     └──trigger.py
├── requirements.txt
└── pytest.ini
```
> Any folders missing on your machine will be created by the scripts if needed.

---
## Required Downloads

### LLM models (for news embeddings, etc.)
Place downloaded models under a folder you will reference (e.g. `apps/NeuralFusionCore/models/` at repo root):

- https://drive.google.com/drive/folders/1htASoZVoRYkjzl8Svsi8fxc-x7eUEtCO?usp=sharing


## Setup

```bash

# Clone repository
git clone https://github.com/Novoxpert/AlphaFusionNet.git
cd AlphaFusionNet


# (optional) create a virtual environment
python -m venv .venv

# Linux/macOS:
source .venv/bin/activate

# Windows (PowerShell):
 .\.venv\Scripts\Activate.ps1

# install exact dependencies
pip install -r requirements.txt
```
---
## Pipeline 

### 1) run scheduler    

```bash

Celery -A tasks worker --loglevel=info
python run_triggers.py
celery -A scheduler beat --loglevel=info
```
Outputs:  a checkpoint such as `apps/NeuralFusionCore/data/outputs/model_weights.pt` and 
        `apps/NeuralFusionCore/data/processed/train.parquet`, `val.parquet` 
        ,`finetune_train.parquet`, `finetune_val.parquet`, `online_test.parquet`,
        `online_bridge.parquet`, `online_bridge_not_norm.parquet`,
        `meta.json`, `normalizer.pkl` and
        `NeuralFusionCore_predictions` , `chrono_bridge` , `AlphaFusionNet_predictions`collections created in `portfolio_db` database of MongoDB


### 2) run NeuralFusionCore API service
Create API for Get weights from Mongodb.

```bash
python -m apps.NeuralFusionCore.scripts.api_service
```

### 4) ChronoBridge
Extract trained fused embeddings per asset and store in MongoDB

### 3) run ChronoBridge API service
Create API for Get Fused embedding and ohlcv per asset from Mongodb.
```bash
python -m apps.ChronoBridge.scripts.chronobridge_api_ervice 
```
### 4) run tests
run Full CI-quality testing and unit testing suite.
```bash
pip install pytest
ptest -v tests
```
---
## Script Cheat‑Sheet

- **`apps/NeuralFusionCore/lib/*.py`** — internal modules for datasets, models, features, news embeddings,training loops, utilities, and backtesting specialized for direct weights.  
- **`apps/NeuralFusionCore/config.py`** — central configuration / argument helpers used by the scripts.
- **`apps/NeuralFusionCore/data_ingest_service.py`** — fetch OHLCV from ClickHouse and news from Mongo for the given interval, and push results (per-symbol ohlcv DataFrame pickles and news DataFrame) to Redis.

Usage examples:
one-shot latest 4h (use scheduler to run every 4h)
```bash
python -m apps.NeuralFusionCore.scripts.data_ingest_service --mode latest --hours 4
```
- **`apps/NeuralFusionCore/scripts/features_service.py`** — Builds features from Redis.

Modes:
  - train:     full rebuild (includes normalizer + meta)
  - finetune:  incremental build (reuse existing normalizer/meta)
  - inference: build features for inference only (produces online_test.parquet)
  - bridge:    build features for ChronobBridge only
  - time:      select data by start_time/end_time for any mode

Usage Examples:
```bash
python -m apps.NeuralFusionCore.scripts.features_service --mode finetune --latest_hours 24
```
- **`apps/NeuralFusionCore/scripts/train_service.py`** — Train from scratch on processed/train.parquet and processed/val.parquet
Usage Example:
```bash
python -m apps.NeuralFusionCore.scripts.train_service --epocha 50 
```
- **`apps/NeuralFusionCore/scripts/finetune_service.py`** —Fine-tune an existing saved model using the latest features. If validation loss improves, replace saved model and keep previous version with timestamp.

Usage Example:
```bash
python -m apps.NeuralFusionCore.scripts.finetune_service --epocha 10 --save_best
```
- **`apps/NeuralFusionCore/scripts/prediction_service.py`** —Scheduled inference: fetch latest data, compute features, infer model, transform logits into portfolio weights, and save predictions to MongoDB and Redis.

Usage Example:
```bash
python -m apps.NeuralFusionCore.scripts.prediction_service --hours 4 
```
- **`apps/NeuralFusionCore/scripts/api_service.py`** — create API for Get NeuralFusion weights from Mongodb.
- **`apps/ChronoBridge/scripts/chronobridge_service.py`** — extract trained fused embeddings per asset and store in MongoDB
- **`apps/ChronoBridge/scripts/chronobridge_api_service.py`** — create API for Get Fused embeddings from Mongodb.
- **`tests/*.py`** —  Full CI-quality testing and unit testing suite that validates API, model, data pipelines.
- **`scheduler/scheduler.py`** — Defines all Celery tasks for data ingestion, feature processing, model training, fine-tuning, and prediction 
- **`scheduler/tasks.py`** — Configures the Celery beat scheduler to automatically trigger periodic workflows(daily updates and 4-hourly predictions) at defined times.
- **`scheduler/trigger.py`** — Manually triggers the one-time initial workflow(historical or first-time pipeline run) by sending the initial_run task to Celery.
---
## Dependencies

* Python 3.12+
* PyTorch 2.x

---
## Appendix

### Upstream Repositories

Influential upstream repositories:

* [**NeuralFusionCore**](https://github.com/Novoxpert/NeuralFusionCore): Direct Portfolio Weight Forecasting with Cross‑Gated Attention Fusion
* [**ChronoBridge**](https://github.com/Novoxpert/ChronoBridge): Multi-Modal Embedding Fusion & Serving Pipeline
* [**NetWeaver**](https://github.com/Novoxpert/NetWeaver): Financial Graph Attention Network for Stock Prediction

---
## Authors & Citation

**Developed by the [Novoxpert Research Team](https://github.com/Novoxpert)**  
Lead Contributors:
 - [Elham Esmaeilnia](https://github.com/Elham-Esmaeilnia)
 

If you use this repository or build upon our work, please cite:

> Novoxpert Research (2025). *AlphaFusionNet: LLM-Driven Neural–Graph Portfolio Engine.*  
> GitHub: [https://github.com/Novoxpert/AlphaFusionNet](https://github.com/Novoxpert/AlphaFusionNet)

```bibtex
@software{novoxpert_neuralfusioncore_2025,
  author       = {Elham Esmaeilnia},
  title        = {AlphaFusionNet: LLM-Driven Neural–Graph Portfolio Engine},
  organization = {Novoxpert Research},
  year         = {2025},
  url          = {https://github.com/Novoxpert/AlphaFusionNet}
}
```
---
## Support

- **Issues & Bugs**: [Open on GitHub](https://github.com/Novoxpert/AlphaFusionNet/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Novoxpert/AlphaFusionNet/discussions)
- **Feature Requests**: Open a feature request issue
---
