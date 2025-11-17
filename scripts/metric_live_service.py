"""
Live Window Metric Snapshot Service
===================================

This script performs a **single-minute computation** of live portfolio
performance metrics for the current 4-hour trading window. It is designed
to run under a scheduler (Celery beat, cron, Kubernetes cronjob, etc.)
**every minute**, where each invocation processes exactly one snapshot.

Purpose
-------
The service computes real-time values required by the dashboard:

    • live NAV (V(t))
    • cumulative return Rp(t)
    • PnL(t)
    • benchmark return and alpha
    • per-symbol valuations and contributions
    • stale-price detection
    • current close price per symbol for charting

Each execution produces one new minute-level snapshot, which is appended to:

    • MongoDB `windows.live_history` (embedded history per window)
    • MongoDB `live_metrics` (flat, fast-to-query collection)

Lifecycle Logic
---------------
1. Identify the current 4-hour window (`window_id`, `t0`, `t1`).
2. Load or initialize the corresponding window document:
       - entry prices exactly at t0
       - fixed weights, quantities, and initial cash
3. Determine the current minute timestamp (`as_of`).
4. Conditions:
       • If `as_of < t0`: window not started → skip.
       • If `as_of > t1`: window ended → mark ENDED → stop.
5. Fetch the close price for each symbol and the benchmark at `as_of`
   using ClickHouse:
       - if missing, fallback to last snapshot price or entry price
       - track stale updates
6. Compute the full live metric snapshot and persist it.
7. Log stale symbol information or success messages.

Scheduler Usage
---------------
This script is intentionally **stateless between runs**.  
A scheduler should invoke it once per minute:

    - Celery beat task (recommended)
    - Cron job (`* * * * *`)
    - Serverless scheduled trigger

Each call computes exactly one minute's snapshot and exits immediately.

Inputs
------
• `AFN_config.yaml` (env-substituted settings)  
• MongoDB (`windows`, `live_metrics`, `AlphaFusionNet_predictions`)  
• ClickHouse OHLCV table for price lookups  
• Portfolio weights from `AlphaFusionNet_predictions`  
• Current UTC time (to infer t0, t1, and as_of)

Outputs
-------
Two MongoDB collections are updated:

1. **windows**  
    - `live_history`: appended snapshot  
    - updated stale count and timestamps  
    - marked ENDED once `t1` passes  

2. **live_metrics**  
    - flattened snapshot document for dashboard’s real-time charts  

This service is a core building block of the AlphaFusionNet dashboard’s
minute-by-minute live performance visualization.
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
from lib.metric_utils import (
    init_window_state,
    get_minute_close_price,
    compute_metrics_snapshot,
    persist_window_snapshot,
    mark_window_ended,
    NAV0_DEFAULT,
)

# ------------------------
# Basic logging
# ------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

CONFIG_FILE = os.environ.get("ALPHAFUSIONNET_CONFIG", "config/AFN_config.yaml")


# ------------------------
# Config loader
# ------------------------
def load_config(config_path: str = CONFIG_FILE):
    # Load environment variables from .env
    load_dotenv()

    # Read YAML as text
    with open(config_path, "r") as f:
        raw_yaml = f.read()

    # Substitute ${VAR} with environment values
    expanded_yaml = os.path.expandvars(raw_yaml)

    # Parse YAML
    cfg = yaml.safe_load(expanded_yaml)
    if not cfg:
        raise RuntimeError(f"Config file empty or invalid: {config_path}")

    return cfg


# ------------------------
# Single-shot live metric computation
# ------------------------
def run_live_metric_snapshot(
    db,
    ch_client,
    symbols,
    portfolio_weights,
    window_id: str,
    ch_table: str,
    start_time_utc: datetime,
    benchmark_symbol: str = "SP:SPX",
    nav0: float = NAV0_DEFAULT,
    window_hours: int = 4,
):
    """
    Single-shot computation for one minute.

    Intended to be triggered by a scheduler (Celery beat / cron) every minute.

    Steps:
      - Initialize or load window document for this window_id (t0).
      - Determine `as_of` = current minute (UTC).
      - If as_of < t0: do nothing (window not started yet).
      - If as_of > t1: mark window as ENDED (if not already) and exit.
      - Fetch live close prices for all symbols & benchmark at `as_of`,
        with fallback to last snapshot prices or entry prices.
      - Compute NAV, return, PnL, alpha, etc.
      - Persist snapshot into Mongo.
    """
    logger.info(
        "Running live metric snapshot for window %s (t0=%s)",
        window_id,
        start_time_utc,
    )

    # Initialize / load window state (entry prices exactly at t0)
    window_doc = init_window_state(
        db=db,
        ch_client=ch_client,
        ch_table=ch_table,
        window_id=window_id,
        symbols=symbols,
        portfolio_weights=portfolio_weights,
        benchmark_symbol=benchmark_symbol,
        start_time_utc=start_time_utc,
        nav0=nav0,
        window_hours=window_hours,
    )

    t0 = window_doc["t0"]
    t1 = window_doc["t1"]

    now = datetime.utcnow()
    as_of = now.replace(second=0, microsecond=0)

    # If we're before the window start, do nothing
    if as_of < t0:
        logger.info(
            "as_of=%s is before window start t0=%s, skipping computation",
            as_of,
            t0,
        )
        return

    # If we've passed the window end, mark ENDED and stop
    if as_of > t1:
        logger.info(
            "as_of=%s is after window end t1=%s for window %s",
            as_of,
            t1,
            window_id,
        )
        # Only mark ended if not already
        if window_doc.get("status") != "ENDED":
            mark_window_ended(db, window_id)
        return

    # ------------------------
    # Determine last known prices for fallback
    # ------------------------
    live_col = db["live_metrics"]
    last_snapshot = live_col.find_one(
        {"window_id": window_id},
        sort=[("as_of", -1)],
    )

    if last_snapshot:
        last_prices = last_snapshot.get("prices", {}) or {}
        last_benchmark_price = last_snapshot.get("benchmark_price")
    else:
        # First ever snapshot -> fallback to entry prices
        last_prices = dict(window_doc.get("entry_prices", {}))
        last_benchmark_price = window_doc.get("benchmark_entry")

    prices_t = {}
    stale_symbols = []

    # ------------------------
    # Fetch per-symbol prices for this minute
    # ------------------------
    for sym in symbols:
        price, is_stale = get_minute_close_price(
            ch_client=ch_client,
            ch_table=ch_table,
            symbol=sym,
            as_of=as_of,
        )
        if price is None:
            # fallback: last known OR entry price
            fallback = last_prices.get(sym)
            if fallback is not None:
                price = fallback
            else:
                # If we truly have no price, skip symbol
                logger.warning("No price available at all for %s at %s", sym, as_of)
                continue
            is_stale = True

        prices_t[sym] = price
        if is_stale:
            stale_symbols.append(sym)

    # ------------------------
    # Fetch benchmark price for this minute
    # ------------------------
    bench_price, bench_is_stale = get_minute_close_price(
        ch_client=ch_client,
        ch_table=ch_table,
        symbol=benchmark_symbol,
        as_of=as_of,
    )
    if bench_price is None:
        bench_price = last_benchmark_price
        bench_is_stale = True

    if bench_is_stale and benchmark_symbol not in stale_symbols:
        stale_symbols.append(benchmark_symbol)

    # ------------------------
    # Compute metrics snapshot
    # ------------------------
    snapshot = compute_metrics_snapshot(
        window_doc=window_doc,
        prices_t=prices_t,
        benchmark_price_t=bench_price,
        as_of=as_of,
    )

    # Persist snapshot & stale info
    persist_window_snapshot(
        db=db,
        window_id=window_id,
        snapshot=snapshot,
        stale_symbols=stale_symbols,
    )

    if stale_symbols:
        logger.warning(
            "Stale symbols for window %s at %s: %s",
            window_id,
            as_of,
            ", ".join(stale_symbols),
        )
    else:
        logger.info("Snapshot stored for window %s at %s", window_id, as_of)


# ------------------------
# Entrypoint
# ------------------------
if __name__ == "__main__":
    cfg = load_config()
    ch_cfg = cfg["clickhouse"]
    ch_client = init_clickhouse_client(
        host=ch_cfg["host"],
        port=int(ch_cfg["port"]),
        database=ch_cfg["database"],
        user=ch_cfg["user"],
        password=ch_cfg["password"],
    )

    mongo_cfg = cfg["novo_mongo"]
    mongo_client, db = init_mongo_client(mongo_cfg)

    # Default window start: current hour in UTC (e.g. 14:00 UTC for trading window)
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    start_time = now

    # Get latest portfolio weights from AlphaFusionNet_predictions
    weights_doc = db["AlphaFusionNet_predictions"].find_one(
        sort=[("timestamp", -1)]
    )
    if not weights_doc:
        raise RuntimeError("No weights found in AlphaFusionNet_predictions collection")

    portfolio_weights = weights_doc["final_weights"]
    symbols = list(portfolio_weights.keys())

    market_cfg = cfg.get("market", {})
    benchmark_symbols = market_cfg.get("benchmark_symbol", ["SP:SPX"])
    benchmark_symbol = benchmark_symbols[0]

    window_id = f"{start_time:%Y%m%d_%H%M}"

    run_live_metric_snapshot(
        db=db,
        ch_client=ch_client,
        symbols=symbols,
        portfolio_weights=portfolio_weights,
        window_id=window_id,
        ch_table=ch_cfg["table"],
        start_time_utc=start_time,
        benchmark_symbol=benchmark_symbol,
        nav0=NAV0_DEFAULT,
    )
