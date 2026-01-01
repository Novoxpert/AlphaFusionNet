# Tech Context — AlphaFusionNet

## Languages
- Python (primary)

## Frameworks / libraries
- FastAPI + Uvicorn (HTTP APIs)
- Celery (task queue / orchestration)
- Redis (Celery broker/backend; feature caching)
- PyTorch (+ torch-geometric) (ML modeling)
- HuggingFace Transformers (news embedding / LLM tokenization)
- Pydantic (config/validation in some places)

## Data stores
- MongoDB (predictions, embeddings, metrics, reasoning)
- ClickHouse (OHLCV source)

## Config
- `AlphaFusionNet/config/AFN_config.yaml` (env-substituted credentials like `${CH_HOST}`)
- `.env` is expected at `AlphaFusionNet/.env` but must never be committed.

## Local dev / run
- Submodules must be initialized recursively.
- Windows notes:
  - Celery worker typically needs `-P threads`.
  - Avoid bash-specific chaining in commands.

