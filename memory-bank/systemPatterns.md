# System Patterns — AlphaFusionNet

## Architectural patterns
- **Pipeline + orchestration**: services are composed into workflows executed by Celery tasks.
- **Data-contract integration**: components communicate via MongoDB collections and filesystem artifacts (CSV/JSON/parquet).
- **Synchronized windows**: ChronoBridge creates aligned windows to ensure NFC + NetWeaver see consistent context.
- **Monitoring metrics**: live window snapshots persisted per minute, plus monthly aggregation.

## Module boundaries (high-level)
- ChronoBridge: ingestion + feature generation + fused embeddings.
- NeuralFusionCore: temporal/news fusion model training + inference.
- NetWeaver: cross-symbol graph modeling.
- AlphaFusionNet: fusion + policy + reasoning.
- Scheduler: orchestration + trading-day gating.

## Data flow patterns
- ClickHouse/Mongo → Redis → parquet/meta → model inference → MongoDB outputs.
- MongoDB is the primary persistence layer; schemas are implicit.

## Conventions
- Services are usually invoked via `python -m ...`.
- Config is YAML + env vars.

