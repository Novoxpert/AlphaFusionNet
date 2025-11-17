"""
Metric Monthly Service
=======================

This script computes **current month-to-date (MTD)** portfolio performance
metrics using the stored 4-hour trading windows and minute-level OHLCV data
from ClickHouse. It is intended to run **once per day**, immediately after
the daily trading window ends, so the dashboard remains updated with
fresh month-to-date analytics.

Overview
--------
The service aggregates all daily trading windows within the current month
and builds day-level portfolio performance metrics:

    • daily portfolio returns (using per-day weights and daily symbol returns)
    • winning percentage for the month
    • strongest consecutive positive-return streak (consistency score)
    • 5-day rolling return consistency
    • metadata including:
         number of trading days,
         first trading day observed,
         last trading day observed

These metrics are written into MongoDB and power the dashboard's
"Day-in-Month" / "Monthly Performance" visualizations.

Daily Execution Flow
--------------------
1. Determine the **current month** from the UTC date.
2. Read all window documents (`db["windows"]`) whose `t0` falls within
   this month. Each window contributes:
       • daily portfolio weights
       • a boundary for daily symbol pricing
3. Fetch minute-level OHLCV prices from ClickHouse for the full month
   and all trading symbols.
4. Convert minute-level prices into **daily open/close**, producing
   per-symbol daily returns.
5. Compute **daily portfolio returns**, combining daily symbol returns
   with the weights recorded in each day's window.
6. Calculate MTD performance metrics defined in the AlphaFusionNet
   specification.
7. Update the MongoDB `monthly` collection under:
       {
           "month_id": "YYYY-MM",
           "last_metrics": {...},
           "metrics_history": [...],
           "portfolio_daily_returns": {...}
       }

Persistence Model
-----------------
The script **does not overwrite history**.  
Instead, for each run during the month:

    • `last_metrics` is updated to the newest computation result
    • `metrics_history` receives a new snapshot entry
    • `portfolio_daily_returns` is refreshed

This allows the dashboard to visualize **how monthly metrics evolve day by day**.

Scheduling
----------
Run this script **once per trading day**, after the 4-hour window ends.
If run multiple times per day, it simply appends a new metrics snapshot
for that day and updates the latest values—safe and idempotent.

Inputs
------
• ClickHouse config from `AFN_config.yaml`  
• MongoDB config from `AFN_config.yaml`  
• Trading symbols & benchmark symbol from config  
• Window documents from MongoDB `windows` collection  
• OHLCV data from ClickHouse candle table  

Outputs
-------
A single document per month in MongoDB collection `monthly`:

    {
        "month_id": "YYYY-MM",
        "year": YYYY,
        "month": MM,
        "last_metrics": {...},
        "last_snapshot_date": "YYYY-MM-DD",
        "metrics_history": [
            {
                "date": "YYYY-MM-DD",
                "metrics": { ... }
            },
            ...
        ],
        "portfolio_daily_returns": {
            "YYYY-MM-DD": float,
            ...
        },
        "created_at": datetime
    }

This structure supports both real-time MTD metrics and a historical timeline
of daily changes for dashboard visualizations.
-------
Author: Elham Esmaeilnia(elham.e.shirvani@gmail.com)
Date: 2025 Nov 17
Version: 1.0.0 

"""

import logging
import os
from datetime import datetime

import yaml
from dotenv import load_dotenv

from lib.db_utils import init_clickhouse_client, init_mongo_client
from lib.metric_utils import compute_monthly_performance_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

CONFIG_FILE = os.environ.get("ALPHAFUSIONNET_CONFIG", "config/AFN_config.yaml")


def load_config(config_path: str = CONFIG_FILE):
    load_dotenv()
    with open(config_path, "r") as f:
        raw_yaml = f.read()
    expanded_yaml = os.path.expandvars(raw_yaml)
    cfg = yaml.safe_load(expanded_yaml)
    if not cfg:
        raise RuntimeError(f"Config file empty or invalid: {config_path}")
    return cfg


if __name__ == "__main__":
    cfg = load_config()

    # ClickHouse
    ch_cfg = cfg["clickhouse"]
    ch_client = init_clickhouse_client(
        host=ch_cfg["host"],
        port=int(ch_cfg["port"]),
        database=ch_cfg["database"],
        user=ch_cfg["user"],
        password=ch_cfg["password"],
    )

    # Mongo
    mongo_cfg = cfg["mongo"]
    mongo_client, db = init_mongo_client(mongo_cfg)

    # Decide which month to compute:
    # -> CURRENT month (month-to-date), not previous month
    today = datetime.utcnow().date()
    year = today.year
    month = today.month

    market_cfg = cfg.get("market", {})
    trading_symbols = market_cfg.get("symbols_usdt", [])
    benchmark_symbols = market_cfg.get("benchmark_symbol", ["SP:SPX"])
    benchmark_symbol = benchmark_symbols[0]

    logger.info("Computing MONTH-TO-DATE metrics for %04d-%02d", year, month)

    metrics = compute_monthly_performance_metrics(
        db=db,
        year=year,
        month=month,
        benchmark_symbol=benchmark_symbol,
        trading_symbols=trading_symbols,
    )

    logger.info("Done. MTD metrics for %04d-%02d: %s", year, month, metrics)
