# Project Brief — AlphaFusionNet

## What is this repository?
AlphaFusionNet is a multi-service Python platform for **market monitoring and risk awareness**.
It coordinates:
- **ChronoBridge**: OHLCV+news ingestion, time alignment, feature building, fused embeddings.
- **NeuralFusionCore**: multimodal temporal + news model that outputs signed weights.
- **NetWeaver**: graph model over cross-symbol relationships; outputs a Top-K ranking/score.
- **AlphaFusionNet**: fuses model outputs and optionally consults an LLM to produce a policy JSON + final weights.

The README emphasizes **non-advisory** use: outputs must be descriptive/diagnostic, not trading recommendations.

## Target architecture / merge goal (5–10 lines)
1) Produce a merge-ready inventory that maps **module boundaries, entrypoints, and contracts**.
2) Make submodule topology explicit (ChronoBridge / NeuralFusionCore / NetWeaver plus nested TimesNet + Financial-GraphAttention).
3) Document data contracts: Mongo collections, Redis usage (keys/dbs), ClickHouse expectations, and file artifacts.
4) Provide clear guidance for future agents doing merges/refactors: where each responsibility lives and how components interact.
5) Keep all documentation commit-ready under `docs/ai/**` and `memory-bank/**`.

