# Product Context — AlphaFusionNet

## Users / customers
- Internal research / monitoring teams who need **market structure** and **risk temperature** visibility.
- Operators who need data freshness/governance signals and auditability.

## Problem solved
Provide an end-to-end pipeline that:
1) ingests OHLCV + news,
2) constructs synchronized multimodal features,
3) runs temporal and graph models,
4) fuses results into **monitoring-oriented** outputs,
5) produces live and monthly metrics and exposes data via APIs.

## Success criteria
- Services can be run independently and orchestrated via Celery.
- Data contracts are stable: Mongo collections, Redis cache keys, and filesystem artifacts.
- Outputs are compliance-safe (descriptive/diagnostic; not advice).

## Non-goals / constraints
- Not an execution engine; must not output trade instructions or performance marketing.
- Requires external infra: ClickHouse, MongoDB, Redis, (optional) OpenAI API.

