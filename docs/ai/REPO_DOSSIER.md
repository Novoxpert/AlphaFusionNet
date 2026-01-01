# REPO_DOSSIER — AlphaFusionNet

Repo root folder in this workspace: `AlphaFusionNet/`

This dossier is intended to be **merge-ready** for another AI agent doing architecture and integration work. It focuses on:
1) **what exists**, 2) **how it fits together**, 3) **the stable contracts** (APIs, queues, DB, file I/O), and 4) **risks/constraints**.

---

## 1) Executive Summary (product goal + merge goal 5–10 lines)

**Product goal:** AlphaFusionNet is a market monitoring and risk-awareness platform that coordinates a multimodal ingestion pipeline (prices + news), two modeling subsystems (temporal neural and graph dependency), and a fusion/reasoning layer to produce **monitoring-oriented outputs** (watchlists, diagnostics, governance artifacts). It explicitly avoids investment advice / execution instructions.

**Merge goal:** enable safe refactoring and cross-repo merging by clearly documenting module boundaries (ChronoBridge, NeuralFusionCore, NetWeaver, AlphaFusionNet), runtime entrypoints (FastAPI + Celery), data contracts (Mongo collections, Redis keys, ClickHouse table expectations, filesystem artifacts), and submodule topology. The outputs in `docs/ai/**` and `memory-bank/**` are meant to be commit-ready artifacts for future agents.

---

## 2) Repo Type & Layout

**Repo type:** multi-service Python platform orchestrator with nested submodules.

**Top-level layout (within `AlphaFusionNet/`):**
- `AlphaFusionNet/apps/` — component applications
  - `AlphaFusionNet/apps/ChronoBridge/` — ingestion + features + fused embeddings
  - `AlphaFusionNet/apps/NeuralFusionCore/` — temporal/news model + training + inference + API
  - `AlphaFusionNet/apps/NetWeaver/` — graph model training/prediction services
- `AlphaFusionNet/scripts/` — orchestrator scripts (fusion engine, metrics, APIs)
- `AlphaFusionNet/scheduler/` — Celery app + beat schedule + tasks
- `AlphaFusionNet/src/` — AlphaFusionNet fusion/controller cores
- `AlphaFusionNet/lib/` — shared utilities (DB init, trading calendar, metric utilities)
- `AlphaFusionNet/config/AFN_config.yaml` — primary YAML config (env-substituted secrets)
- `AlphaFusionNet/tests/` — unit + integration tests (ClickHouse/Mongo/Redis are mocked)

**Authoritative file inventories:**
- `docs/ai/REPO_TREE.txt` (top-level)
- `docs/ai/REPO_TREE_ALL.txt` (recursive, includes submodule working tree files)

---

## 3) Module Map (name, responsibility, key paths, entrypoints, dependencies)

### M0 — AlphaFusionNet Orchestrator / Fusion Layer
- **Responsibility:**
  - Fuse NeuralFusionCore signed weights with NetWeaver predicted-return scores.
  - Optionally consult an LLM to generate a “policy JSON” (alpha/method/top-k/overrides) and then compute final weights.
  - Persist outputs and reasoning to MongoDB.
- **Key paths:**
  - `AlphaFusionNet/src/controller.py` (LLM controller + policy validation)
  - `AlphaFusionNet/src/quant_alphafusionnet.py` (quant fusion math)
  - `AlphaFusionNet/src/llm_alphafusionnet.py` (OpenAI wrapper; legacy `openai.ChatCompletion`)
  - `AlphaFusionNet/src/TradingAgent.py` (LLM-generated human explanation)
  - `AlphaFusionNet/scripts/alphafusionnet_service.py` (main runtime script)
- **Entrypoints:**
  - `python -m scripts.alphafusionnet_service`
  - API: `python -m scripts.alphafusionnet_api_service`
- **Depends on:**
  - Files produced by component subsystems:
    - `AlphaFusionNet/apps/NeuralFusionCore/scripts/NeuralFusionCore_prediction.json` (NFC output)
    - `AlphaFusionNet/apps/NetWeaver/results/predict/selected_prediction.csv` (NetWeaver output)
  - MongoDB credentials (`NOVO_MONGO_*`), OpenAI key (`OPENAI_API_KEY`)

### M1 — ChronoBridge (Ingest + Features + Embeddings)
- **Responsibility:**
  - Fetch OHLCV from ClickHouse and news from MongoDB.
  - Resample OHLCV (1m → 3m) and build multimodal feature table.
  - Embed news content using a transformer (BigBird) and align to time grid.
  - In “bridge/synchronize” modes, produce datasets for NFC/NetWeaver and compute fused embeddings stored in Mongo.
- **Key paths:**
  - `AlphaFusionNet/apps/ChronoBridge/scripts/data_ingest_service.py`
  - `AlphaFusionNet/apps/ChronoBridge/scripts/features_service.py`
  - `AlphaFusionNet/apps/ChronoBridge/scripts/chronobridge_service.py`
  - `AlphaFusionNet/apps/ChronoBridge/scripts/chronobridge_api_service.py`
  - `AlphaFusionNet/apps/ChronoBridge/src/inference.py` (model-driven embedding extraction)
- **Entrypoints:**
  - `python -m apps.ChronoBridge.scripts.data_ingest_service ...`
  - `python -m apps.ChronoBridge.scripts.features_service ...`
  - `python -m apps.ChronoBridge.scripts.chronobridge_service ...`
  - API: `python -m apps.ChronoBridge.scripts.chronobridge_api_service`
- **Depends on:**
  - ClickHouse table configured via `AlphaFusionNet/config/AFN_config.yaml` + env
  - MongoDB (news source)
  - Redis (as intermediate store)

### M2 — NeuralFusionCore (Temporal + News Fusion Model)
- **Responsibility:**
  - Train and run inference for a model combining TimesNet-like temporal processing + news embeddings.
  - Produce **signed** portfolio weights (long/short) and persist them to MongoDB and JSON.
- **Key paths:**
  - `AlphaFusionNet/apps/NeuralFusionCore/scripts/train_service.py`
  - `AlphaFusionNet/apps/NeuralFusionCore/scripts/finetune_service.py`
  - `AlphaFusionNet/apps/NeuralFusionCore/scripts/prediction_service.py`
  - `AlphaFusionNet/apps/NeuralFusionCore/scripts/api_service.py`
  - Data artifacts in `AlphaFusionNet/apps/NeuralFusionCore/data/`.
- **Entrypoints:**
  - `python -m apps.NeuralFusionCore.scripts.train_service --epochs ...`
  - `python -m apps.NeuralFusionCore.scripts.prediction_service --mode synchronize|inference ...`
  - API: `python -m apps.NeuralFusionCore.scripts.api_service`
- **Depends on:**
  - Feature parquet outputs from ChronoBridge (`online_test.parquet`, `meta.json`, normalizers).
  - MongoDB + Redis (persistence/integration)

### M3 — NetWeaver (Graph Dependency Model)
- **Responsibility:**
  - Train and predict a graph model over cross-symbol relationships.
  - Emit a **Top-20 selection** CSV/JSON in `results/predict/`.
- **Key paths:**
  - `AlphaFusionNet/apps/NetWeaver/src/services/netweaver_train_service.py`
  - `AlphaFusionNet/apps/NetWeaver/src/services/netweaver_prediction_service.py`
  - `AlphaFusionNet/apps/NetWeaver/src/data_pipeline.py`
  - `AlphaFusionNet/apps/NetWeaver/src/predict.py`
- **Entrypoints:**
  - `python -m apps.NetWeaver.src.services.netweaver_train_service ...`
  - `python -m apps.NetWeaver.src.services.netweaver_prediction_service ...`
- **Depends on:**
  - ChronoBridge “bridge mode” outputs and/or its API (as described in `AlphaFusionNet/docs/usage.md`).
  - Local processed pickles under `AlphaFusionNet/apps/NetWeaver/data/`.

### M4 — Scheduler / Orchestration
- **Responsibility:**
  - Coordinate daily update, prediction window run, and metric computation.
  - Enforce trading-day gating.
- **Key paths:**
  - `AlphaFusionNet/scheduler/tasks.py`
  - `AlphaFusionNet/scheduler/scheduler.py`
- **Entrypoints:**
  - Worker: `celery -A scheduler.tasks worker --loglevel=info` (Windows uses `-P threads`)
  - Beat: `celery -A scheduler.scheduler beat --loglevel=info`
- **Depends on:**
  - Redis broker/backend
  - Python subprocess execution of the other module entrypoints

### M5 — Metrics
- **Responsibility:**
  - Live 4-hour valuation snapshots per minute and monthly performance aggregation.
- **Key paths:**
  - `AlphaFusionNet/scripts/metric_live_service.py`
  - `AlphaFusionNet/scripts/metric_monthly_service.py`
  - `AlphaFusionNet/scripts/metric_backtesting.py`
  - `AlphaFusionNet/lib/metric_utils.py`
  - `AlphaFusionNet/lib/db_utils.py`
- **Depends on:**
  - MongoDB collections: `windows`, `live_metrics`, `monthly`, `AlphaFusionNet_predictions`
  - ClickHouse table for prices (live & backtesting modes)

---

## 4) Key Data Flows (end-to-end narratives)

### Flow A — Initial end-to-end bootstrap (`tasks.initial_run`)
Defined in: `AlphaFusionNet/scheduler/tasks.py`
1) Compute trading-days cache (`python -m scripts.compute_trading_days_service`).
2) ChronoBridge ingest historical data → Redis.
3) ChronoBridge build features (train mode) → parquet + normalizer + meta.
4) NeuralFusionCore train → checkpoint weights.
5) ChronoBridge bridge mode → generate bridge artifacts + start ChronoBridge API in background.
6) NetWeaver train service.

### Flow B — Daily update (`tasks.daily_update`)
1) ChronoBridge ingest “latest” window (48h) to Redis.
2) ChronoBridge features finetune for last 48h.
3) NeuralFusionCore finetune.
4) ChronoBridge bridge mode (last 48h) for NetWeaver.
5) NetWeaver finetune.

### Flow C — Prediction window (`tasks.prediction_14_30PM`)
1) Determine previous common trading day (UTC) using trading-day cache.
2) Build [14:30, 18:30] UTC window.
3) ChronoBridge synchronize mode: run ingest + features for the window.
4) NeuralFusionCore prediction service (synchronize).
5) NetWeaver prediction service.
6) AlphaFusionNet fusion service (LLM policy + quant fusion) → MongoDB.
7) Schedule `metric_live_service` every minute in the window and then `metric_monthly_service`.

### Flow D — Live metrics
`AlphaFusionNet/scripts/metric_live_service.py`:
1) Fetch latest weights from `AlphaFusionNet_predictions`.
2) Initialize a window doc in `windows` with entry prices (ClickHouse) and computed positions.
3) Every minute compute NAV + contributions, store to:
   - `windows.live_history` (embedded)
   - `live_metrics` (flattened)

---

## 5) Inputs/Outputs (Contracts)

### 5.1 APIs (routes + request/response if discoverable)

Source of extracted routes: `docs/ai/_EXTRACTED_CONTRACTS.json`.

#### AlphaFusionNet API — `AlphaFusionNet/scripts/alphafusionnet_api_service.py`
- `GET /latest_alphafusionnet`
  - **Response:** latest Mongo doc from collection `AlphaFusionNet_predictions` (BSON types flattened).
- `GET /alphafusionnet_history?start=<ISO>&end=<ISO>&limit=<int>`
  - **Response:** list of prediction docs in time range.
- `GET /health`

#### ChronoBridge API — `AlphaFusionNet/apps/ChronoBridge/scripts/chronobridge_api_service.py`
- `GET /fused_embeddings?start_date=YYYY-MM-DD&end_date=YYYY-MM-DD&stocks=SYM&stocks=SYM...`
  - **Response:** `{ count: <int>, data: <list[doc]> }` from Mongo collection `chrono_bridge`.
- `GET /health`

#### NeuralFusionCore API — `AlphaFusionNet/apps/NeuralFusionCore/scripts/api_service.py`
- `GET /latest_prediction`
  - **Response:** latest doc from `NeuralFusionCore_predictions`.
- `GET /prediction_history?start=<ISO>&end=<ISO>&limit=<int>`
  - **Response:** list of predictions.
- `GET /health`

#### Future Testing API — `AlphaFusionNet/scripts/future_testing_api_service.py`
- `GET /future-testing/latest`
  - **Response:** latest reconstructed snapshot from `AlphaFusionNet_future_testing` (features list + weights).

### 5.2 Events/Queues (topics + payload fields if discoverable)

This repo uses **Celery** (task queue), not Kafka/Rabbit.

- Broker: `redis://localhost:6379/1` (defined in `AlphaFusionNet/scheduler/tasks.py`)
- Backend: `redis://localhost:6379/2`
- Task names (from `AlphaFusionNet/scheduler/tasks.py`):
  - `tasks.initial_run`
  - `tasks.daily_update`
  - `tasks.prediction_14_30PM`
  - `tasks.calculate_metric_live`
  - `tasks.calculate_metric_monthly`
  - `tasks.refresh_trading_days_cache`

### 5.3 DB (schema/models/migrations)

No explicit migrations were found; persistence is primarily MongoDB collections and a ClickHouse OHLCV table.

#### MongoDB collections (observed)
- `chrono_bridge`
  - Written by: `AlphaFusionNet/apps/ChronoBridge/scripts/chronobridge_service.py`
  - Read by: `AlphaFusionNet/apps/ChronoBridge/scripts/chronobridge_api_service.py`
  - Typical fields: `date`, `symbol`, fused embedding payloads (exact shape depends on model).

- `NeuralFusionCore_predictions`
  - Written by: `AlphaFusionNet/apps/NeuralFusionCore/scripts/prediction_service.py`
  - Typical doc: `{ ts, weights: [..], stocks: [..] }`

- `NetWeaver_predictions`
  - Written by: `AlphaFusionNet/scripts/alphafusionnet_service.py`
  - Typical doc: `{ timestamp, predicted_return: {symbol: float, ...} }`

- `AlphaFusionNet_predictions`
  - Written by: `AlphaFusionNet/scripts/alphafusionnet_service.py`
  - Typical doc keys:
    - `timestamp`
    - `policy` (alpha/method/topk/overrides/sector multipliers)
    - `final_weights` (dict)
    - `risk_controls` (dict)
    - `reasoning` (string)

- `TradingAgent_reasons`
  - Written by: `AlphaFusionNet/src/TradingAgent.py`
  - Typical doc: `{ timestamp, reasoning }`

- Metrics:
  - `windows` (one doc per live window; includes `live_history` array)
  - `live_metrics` (flattened per-minute snapshots)
  - `monthly` (month-to-date aggregate)

#### ClickHouse (assumed schema)
- Accessed by: `AlphaFusionNet/apps/ChronoBridge/scripts/data_ingest_service.py`, `AlphaFusionNet/lib/metric_utils.py`
- Expected columns include at least: `symbol`, `candle_time`, `close` (and likely OHLCV columns).

### 5.4 File-based I/O (important files)

Key artifacts (non-exhaustive):
- `AlphaFusionNet/config/AFN_config.yaml` — main config; env-substituted secrets.
- ChronoBridge processed datasets:
  - `AlphaFusionNet/apps/ChronoBridge/data/processed/train.parquet`
  - `AlphaFusionNet/apps/ChronoBridge/data/processed/val.parquet`
  - `AlphaFusionNet/apps/ChronoBridge/data/processed/online_test.parquet`
  - `AlphaFusionNet/apps/ChronoBridge/data/processed/online_bridge.parquet`
  - `AlphaFusionNet/apps/ChronoBridge/data/processed/meta.json`
  - `AlphaFusionNet/apps/ChronoBridge/data/processed/normalizer.pkl`
- NeuralFusionCore inference output:
  - `AlphaFusionNet/apps/NeuralFusionCore/scripts/NeuralFusionCore_prediction.json`
- NetWeaver inference output:
  - `AlphaFusionNet/apps/NetWeaver/results/predict/selected_prediction.csv`
- Logs:
  - `AlphaFusionNet/logs/alphafusionnet_service.log`

---

## 6) Tech Stack Inventory (language/framework/DB/queue/infra/CI)

- **Languages:** Python
- **Frameworks:** FastAPI, Uvicorn, Celery, PyTorch, HuggingFace Transformers
- **Databases:** MongoDB, ClickHouse
- **Queue / cache:** Redis (Celery broker/backend + feature cache)
- **Infra:** Local/dev oriented; no Kubernetes manifests found in the authoritative tree.
- **CI:** No GitHub Actions / CI config observed in the authoritative tree (unknown).

---

## 7) Top 20 Important Files (path + why)

1. `AlphaFusionNet/README.md` — repo overview and safe-use policy.
2. `AlphaFusionNet/docs/architecture.md` — architecture diagram + narrative.
3. `AlphaFusionNet/docs/usage.md` — operational runbook.
4. `AlphaFusionNet/config/AFN_config.yaml` — primary config + env var contract.
5. `AlphaFusionNet/scheduler/tasks.py` — orchestration graph (truth source of workflows).
6. `AlphaFusionNet/scheduler/scheduler.py` — schedule timings.
7. `AlphaFusionNet/scripts/alphafusionnet_service.py` — fusion runtime, Mongo persistence.
8. `AlphaFusionNet/src/controller.py` — LLM policy validation + fusion call chain.
9. `AlphaFusionNet/src/quant_alphafusionnet.py` — fusion math (stable contract for weights).
10. `AlphaFusionNet/src/llm_alphafusionnet.py` — OpenAI call wrapper and schema.
11. `AlphaFusionNet/src/TradingAgent.py` — post-hoc reasoning generator.
12. `AlphaFusionNet/scripts/alphafusionnet_api_service.py` — API exposing predictions.
13. `AlphaFusionNet/apps/ChronoBridge/scripts/data_ingest_service.py` — ClickHouse/Mongo → Redis contract.
14. `AlphaFusionNet/apps/ChronoBridge/scripts/features_service.py` — Redis → parquet features contract.
15. `AlphaFusionNet/apps/ChronoBridge/scripts/chronobridge_service.py` — feature pipeline + embedding persistence.
16. `AlphaFusionNet/apps/ChronoBridge/scripts/chronobridge_api_service.py` — embeddings API contract.
17. `AlphaFusionNet/apps/NeuralFusionCore/scripts/prediction_service.py` — NFC output contract.
18. `AlphaFusionNet/apps/NeuralFusionCore/scripts/api_service.py` — NFC API contract.
19. `AlphaFusionNet/apps/NetWeaver/src/services/netweaver_prediction_service.py` — NetWeaver output contract.
20. `AlphaFusionNet/lib/metric_utils.py` — metrics state model (`windows`, `live_metrics`, `monthly`).

---

## 8) Risks & Constraints (incl. submodule/private/auth, windows/linux issues)

### Submodules / nested repos
- There are multiple nested `.gitmodules`:
  - `AlphaFusionNet/.gitmodules` (ChronoBridge, NeuralFusionCore, NetWeaver)
  - `AlphaFusionNet/apps/NeuralFusionCore/.gitmodules` (TimesNet / Time-Series-Library)
  - `AlphaFusionNet/apps/NetWeaver/.gitmodules` (Financial-GraphAttention)
  - plus ChronoBridge contains its own nested NFC/TimesNet.
- This increases merge complexity: duplicated code copies exist (e.g., NFC appears in two places).

### Fork policy not yet configured
- `AlphaFusionNet/memory-bank/submodule_overrides.json` is currently a placeholder template.
- If you want ownership-safe merges, these URLs should be rewritten to your fork URLs.

### Windows vs Linux
- Windows shell differences matter (e.g., `&&` not valid in some shells). Scripts and docs sometimes assume bash.
- Celery on Windows uses `-P threads` and has limitations vs prefork.

### Secrets
- `.env` is expected but must never be committed; code raises if `OPENAI_API_KEY` missing.

### Large artifacts
- Model weights / safetensors and large ML codebases are present; consider Git LFS and avoid needless diffs.

### Compliance / product constraints
- Repo README explicitly forbids generating trade advice; downstream UIs must stay descriptive/diagnostic.

---

## 9) Suggested Read Order

1) `AlphaFusionNet/README.md`
2) `AlphaFusionNet/docs/architecture.md`
3) `AlphaFusionNet/docs/usage.md`
4) `AlphaFusionNet/config/AFN_config.yaml`
5) `AlphaFusionNet/scheduler/tasks.py` + `AlphaFusionNet/scheduler/scheduler.py`
6) `AlphaFusionNet/apps/ChronoBridge/scripts/*` (ingest/features/service)
7) `AlphaFusionNet/apps/NeuralFusionCore/scripts/prediction_service.py`
8) `AlphaFusionNet/apps/NetWeaver/src/services/*`
9) `AlphaFusionNet/scripts/alphafusionnet_service.py` + `AlphaFusionNet/src/*`
10) `AlphaFusionNet/lib/metric_utils.py` + `AlphaFusionNet/scripts/metric_*`
11) `AlphaFusionNet/tests/` for contract expectations

---

## 10) Open Questions / Unknowns

1) The architecture doc mentions a “React Dashboard”, but no frontend source tree was found in `docs/ai/REPO_TREE_ALL.txt` (may live elsewhere).
2) NetWeaver services print instructions assuming an external “Code directory” layout; integration expectations may differ when running inside `AlphaFusionNet/`.
3) MongoDB collection document schemas are implicit; no Pydantic models / schema versioning found.
4) The “future testing” pipeline details are only partially visible (API present; internal service exists but not fully analyzed here).

