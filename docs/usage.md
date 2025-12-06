# Usage Guide

This document explains how to run, train, deploy, and monitor the full **AlphaFusionNet** system.  
It covers the operational flow of ChronoBridge, NeuralFusionCore, NetWeaver, AlphaFusionNet Fusion Engine, and the Metric services.

---

# 1. Prerequisites

Before using the system, ensure you have the following:

### Required Infrastructure
- **ClickHouse** (OHLCV data)
- **MongoDB** (news, predictions, metrics, ..)
- **Redis** (streaming layer)
- **Celery + Redis** (scheduler + task queue)
- **Python 3.11+**
- **PyTorch** (GPU recommended)
- **poetry / pipenv / virtualenv** (optional)

### Environment Variables

Create a `.env` file form `.env_example` with:

~~~
# Local MongoDB
NOVO_MONGO_USER=
NOVO_MONGO_PASS=
NOVO_MONGO_HOST=
NOVO_MONGO_PORT=
NOVO_MONGO_AUTH_DB=
NOVO_MONGO_DB=

# Local Redis
REDIS_HOST=
REDIS_PORT=
REDIS_DB=

# LLM API Key
OPENAI_API_KEY=

# News Source (MongoDB)
NOVO_MONGO_USER=
NOVO_MONGO_PASS=
NOVO_MONGO_HOST=
NOVO_MONGO_PORT=
NOVO_MONGO_AUTH_DB=
NOVO_MONGO_DB=

# Price Source (Clickhouse DB)
CH_HOST=
CH_PORT=
CH_DB=
CH_TABLE=
CH_USER=
CH_PASS=
~~~

---

# 2. Running ChronoBridge

ChronoBridge has four main services:

- **data_ingest_service** – fetches OHLCV + news and pushes to Redis  
- **Features_service** – generates multimodal features (OHLCV + news embeddings)  
- **chronobridge_service** – synchronous features for inference/ bridge features as NetWeaver train/finetune data.
- **chronobridge_api_service** – synchronous feature API for inference/ bridge feature API for NetWeaver train/finetune data.
---

## 2.1 Run Data Ingest Service

Fetches OHLCV (1m) + news (1m), resamples OHLCV → 3m, pushes to Redis.

~~~bash
 python -m apps.ChronoBridge.scripts.data_ingest_service --mode historical --days 30
~~~

---

## 2.2 Run Feature Engineering Service

Consumes Redis streams, computes:

- OHLCV technical features 
- Time features for timesnet mask
- BigBird news embeddings  
- Symbol–news one-hot vectors  
- Merges everything into a final multimodal 3-minute dataframe  

Run:

~~~bash
python -m apps.ChronoBridge.scripts.features_service --mode train --history_days 30
~~~

Outputs:

- Real-time features for inference  
- Training datasets when run with `--mode train`

Outputs saved in `data/processed/`:

- `train.parquet`  
- `val.parquet`  
- `meta.json`

---
## 2.3 bridge Mode 
Provides input for NetWeaver training/finetuning.

~~~bash
python -m apps.ChronoBridge.scripts.chronobridge_service --mode bridge --history_days 30

~~~
---
## 2.4 synchronize Mode 

Provides synchronized input windows (4h) for NeuralFusionCore + NetWeaver.

~~~bash
python -m apps.ChronoBridge.scripts.chronobridge_service --hours 4 --mode synchronized
~~~

## 2.5 chronobridge api service 

Provides API for NetWeaver.

~~~bash
python -m apps.ChronoBridge.scripts.chronobridge_api_service
~~~
---

# 3. NeuralFusionCore

NeuralFusionCore is the multimodal fusion model combining:

- TimesNet (OHLCV sequences)  
- LSTM (news embeddings)  
- Gated Cross Fused attention  
- Sharpe-ratio-based optimization  

---

## 3.1 Training NeuralFusionCore

~~~bash
python -m apps.NeuralFusionCore.scripts.train_service  --epochs 50
~~~

Loads:

- `train.parquet`  
- `val.parquet`  
- `meta.json`

Model weight Output saved in:

data/processed/output/

---

## 3.2 Running NFC Predictions (Live)

~~~bash
python -m apps.NeuralFusionCore.scripts.prediction_service  --mode synchronize --hours 4
       
~~~

Writes to MongoDB:

- `NeuralFusionCore_predictions`

---

# 4. NetWeaver

Graph-based return prediction model.

---

## 4.1 Training NetWeaver

~~~bash
python -m apps.NetWeaver.src.services.netweaver_train_service  --latest_month 1 --no_analysis
~~~

Fetches from ChronoBridge (bridge mode):

- OHLCV windows  
- Fused embeddings from NeuralFusionCore  

Outputs: model weights + CSV logs.

---

## 4.2 Running NetWeaver Predictions (Live)

~~~bash
python -m apps.NetWeaver.src.services.netweaver_prediction_service --latest_hours 4 --future_steps 80 --no_timestamp
~~~

Exports:results/predict/selected_prediction.csv

---

# 5. AlphaFusionNet Fusion Engine

Fuses:

- NeuralFusionCore (risk-aware weights)  
- NetWeaver (return predictions)  
- LLM-generated policy (α, top-K, SL/TP, weighting strategy, reasoning)

---

## 5.1 Running the Fusion Engine (Live)

~~~bash
python -m scripts.alphafusionnet_service
~~~

Writes to MongoDB:

- `AlphaFusionNet_predictions`  

---

# 6. Metrics System

Metrics computed live, monthly, and during backtesting.

---

## 6.1 Live Metrics

~~~bash
python -m scripts.metric_live_service
~~~

---

## 6.2 Monthly Metrics

~~~bash
python -m scripts.metric_monthly_service
~~~

---

## 6.3 Backtesting Metrics

~~~bash
python -m scripts.metric_backtesting
~~~

Visualize:

~~~bash
python -m scripts.show_backtest_metrics
~~~

---

# 7. Celery Scheduler

Start worker:

~~~bash
celery -A app.celery_app worker --loglevel=INFO
~~~

Start training (just for initial run):
~~~bash
python run_triggers.py
~~~
Start scheduler:

~~~bash
celery -A app.celery_app beat --loglevel=INFO
~~~

Celery orchestrates:

- Daily update  
- Prediction at 14:30 PM 

---

# 8. Dashboard

Build and run the React dashboard:

~~~bash
npm install
npm run build
npm run start
~~~

---

# 9. Testing

## Unit Tests & integration tests

~~~bash
pytest -v tests
~~~

---

AlphaFusionNet is a modular, multimodal, LLM-guided portfolio engine built for production.