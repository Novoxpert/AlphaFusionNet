"""
Backtesting / Bulk Simulation Service
=====================================

This script back-fills MongoDB with **synthetic AlphaFusionNet predictions**
and simulated **live windows** over a historical period, using **real prices**
from ClickHouse.

Goals
-----
1. Populate:
     • AlphaFusionNet_predictions
     • windows
     • live_metrics
     • monthly

2. Use:
     • Real OHLCV prices from ClickHouse for the last N months.
     • Dummy but realistic portfolio weights (equal-weight across symbols).
     • The same computation logic as the live metric service:
           - init_window_state
           - get_minute_close_price
           - compute_metrics_snapshot
           - persist_window_snapshot
           - compute_monthly_performance_metrics (from live_history)

3. Allow you to:
     • Inspect per-minute NAV, PnL, alpha, and prices in `live_metrics`.
     • Inspect per-window history in `windows.live_history`.
     • Inspect month-to-date KPIs in `monthly`.

Simulation Model
----------------
For each calendar day in the backtest range:

    1. Create a dummy AlphaFusionNet_predictions document at t0-5min.
    2. Define a 4-hour window [t0, t1] (e.g., 14:00–18:00 UTC).
    3. Initialize a window document with entry prices at t0.
    4. For each minute in [t0, t1]:
           - Fetch or reconstruct prices via get_minute_close_price()
             (with fallback to last known or entry price).
           - Compute a metrics snapshot.
           - Persist it to `windows.live_history` and `live_metrics`.
    5. Mark the window as ENDED.

After all days are simulated, the script calls
`compute_monthly_performance_metrics` for each month covered by the
backtest period, which builds `monthly` documents using the *actual*
prices used in live_history.

Configuration
-------------
Reads config from `AFN_config.yaml` using the same pattern as other
services:

    • clickhouse: host, port, database, table, user, password
    • mongo: host, port, database, user, password, authSource
    • market.symbols_usdt: trading universe
    • market.benchmark_symbol: list with primary benchmark (e.g. SP:SPX)

Usage
-----
Run locally or in a test environment:

    python scripts/metric_backtesting.py

You can adjust:
    BACKTEST_MONTHS       : number of months to simulate backwards
    WINDOW_START_HOUR_UTC : window start hour (e.g. 14 for 14:00 UTC)
    WINDOW_HOURS          : window length (e.g. 4 hours)

Author: Elham Esmaeilnia (elham.e.shirvani@gmail.com)
Date: 2025-11-17
Version: 1.0.0
"""

import logging
import os
from datetime import datetime, timedelta, date

import yaml
from dotenv import load_dotenv

from lib.db_utils import init_clickhouse_client, init_mongo_client
from lib.metric_utils import (
    init_window_state,
    get_minute_close_price,
    compute_metrics_snapshot,
    persist_window_snapshot,
    mark_window_ended,
    compute_monthly_performance_metrics,
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

# How many months back to simulate (e.g., 3 = last 3 months)
BACKTEST_MONTHS = 3

# 4-hour window starting at 14:00 UTC by default
WINDOW_START_HOUR_UTC = 14
WINDOW_HOURS = 4


# ------------------------
# Config loader
# ------------------------
def load_config(config_path: str = CONFIG_FILE):
    load_dotenv()

    with open(config_path, "r") as f:
        raw_yaml = f.read()

    expanded_yaml = os.path.expandvars(raw_yaml)
    cfg = yaml.safe_load(expanded_yaml)
    if not cfg:
        raise RuntimeError(f"Config file empty or invalid: {config_path}")
    return cfg


# ------------------------
# Helpers
# ------------------------
def generate_equal_weights(symbols):
    """
    Simple dummy weights:
      - equal allocation across all provided symbols
      - sum of weights = 1.0
    """
    if not symbols:
        return {}
    w = 1.0 / len(symbols)
    return {sym: w for sym in symbols}


def insert_dummy_prediction(db, ts: datetime, final_weights: dict):
    """
    Insert a dummy AlphaFusionNet_predictions document for testing.

    Structure matches the real project:

        {
            "timestamp": ts,
            "policy": {"risk_controls": {...}},
            "final_weights": {...},
            "risk_controls": {...},
            "reasoning": "Backtest dummy prediction"
        }
    """
    col = db["AlphaFusionNet_predictions"]
    doc = {
        "timestamp": ts,
        "policy": {"source": "backtest_dummy", "risk_controls": {"dummy": True}},
        "final_weights": final_weights,
        "risk_controls": {"dummy": True},
        "reasoning": "Backtest dummy prediction from metric_backtesting.py",
    }
    col.insert_one(doc)
    logger.info("Inserted dummy prediction at %s", ts.isoformat())


def run_backtest_window(
    db,
    ch_client,
    ch_table: str,
    symbols,
    portfolio_weights,
    window_id: str,
    start_time_utc: datetime,
    benchmark_symbol: str,
    nav0: float = NAV0_DEFAULT,
    window_hours: int = WINDOW_HOURS,
):
    """
    Offline simulation of a full live 4-hour window:

      - Initializes window state at t0.
      - Loops minute by minute from t0 to t1.
      - At each minute:
          * fetches prices via get_minute_close_price (with fallbacks),
          * computes metrics snapshot,
          * persists to Mongo.

    This mimics what a per-minute scheduler (metric_live_service) would
    do in real-time, but runs historically and without sleeping.
    """
    logger.info(
        "Simulating backtest window %s from %s (hours=%d)",
        window_id,
        start_time_utc,
        window_hours,
    )

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

    # last known prices for fallback logic
    last_prices = dict(window_doc.get("entry_prices", {}))
    last_benchmark_price = window_doc.get("benchmark_entry")

    as_of = t0
    while as_of <= t1:
        prices_t = {}
        stale_symbols = []

        # per-symbol prices
        for sym in symbols:
            price, is_stale = get_minute_close_price(
                ch_client=ch_client,
                ch_table=ch_table,
                symbol=sym,
                as_of=as_of,
            )
            if price is None:
                fallback = last_prices.get(sym)
                if fallback is not None:
                    price = fallback
                    is_stale = True
                else:
                    logger.warning("No price available at all for %s at %s", sym, as_of)
                    continue

            prices_t[sym] = price
            last_prices[sym] = price
            if is_stale:
                stale_symbols.append(sym)

        # benchmark price
        bench_price, bench_is_stale = get_minute_close_price(
            ch_client=ch_client,
            ch_table=ch_table,
            symbol=benchmark_symbol,
            as_of=as_of,
        )
        if bench_price is None:
            bench_price = last_benchmark_price
            bench_is_stale = True

        last_benchmark_price = bench_price

        if bench_is_stale and benchmark_symbol not in stale_symbols:
            stale_symbols.append(benchmark_symbol)

        # compute snapshot
        snapshot = compute_metrics_snapshot(
            window_doc=window_doc,
            prices_t=prices_t,
            benchmark_price_t=bench_price,
            as_of=as_of,
        )

        # persist snapshot
        persist_window_snapshot(
            db=db,
            window_id=window_id,
            snapshot=snapshot,
            stale_symbols=stale_symbols,
        )

        as_of += timedelta(minutes=1)

    # mark window ended
    mark_window_ended(db, window_id)
    logger.info("Completed backtest window %s", window_id)


def month_range_dates(back_months: int) -> list[date]:
    """
    Build list of calendar dates over the last `back_months` months,
    ending yesterday (UTC).
    """
    today = datetime.utcnow().date()
    end_date = today - timedelta(days=1)

    # approximate: back_months * 31 days; we'll just filter by month later
    approx_days = back_months * 31
    start_candidate = end_date - timedelta(days=approx_days)

    # real start: move to the first day of that candidate's month
    start_date = start_candidate.replace(day=1)

    dates = []
    d = start_date
    while d <= end_date:
        dates.append(d)
        d += timedelta(days=1)
    return dates


# ------------------------
# Entrypoint
# ------------------------
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

    # Market config
    market_cfg = cfg.get("market", {})
    trading_symbols = market_cfg.get("symbols_usdt", [])
    benchmark_symbols = market_cfg.get("benchmark_symbol", ["SP:SPX"])
    benchmark_symbol = benchmark_symbols[0]

    if not trading_symbols:
        raise RuntimeError("No trading symbols defined in market.symbols_usdt")

    logger.info("Backtesting on %d symbols", len(trading_symbols))

    # Build calendar days for backtest
    all_dates = month_range_dates(BACKTEST_MONTHS)
    if not all_dates:
        logger.warning("No dates produced for backtest; exiting.")
        raise SystemExit(0)

    # Filter to months actually in the last BACKTEST_MONTHS
    # (we'll infer months from actual simulated dates)
    logger.info(
        "Backtest date range: %s to %s", all_dates[0].isoformat(), all_dates[-1].isoformat()
    )

    # Collect months we touch, for monthly metrics later
    touched_months = set()

    for d in all_dates:
        # Optionally skip weekends: uncomment if you want Mon-Fri only
        # if d.weekday() >= 5:
        #     continue

        touched_months.add((d.year, d.month))

        # Define t0 window start at 14:00 UTC
        t0 = datetime(d.year, d.month, d.day, WINDOW_START_HOUR_UTC, 0, 0)

        # Equal-weight dummy portfolio over all trading symbols
        final_weights = generate_equal_weights(trading_symbols)

        # Insert dummy AlphaFusionNet_prediction at t0 - 5 minutes
        pred_ts = t0 - timedelta(minutes=5)
        insert_dummy_prediction(db, pred_ts, final_weights)

        # window_id consistent with live service (YYYYMMDD_HHMM)
        window_id = f"{t0:%Y%m%d_%H%M}"

        logger.info("Simulating window %s for date %s", window_id, d.isoformat())

        run_backtest_window(
            db=db,
            ch_client=ch_client,
            ch_table=ch_cfg["table"],
            symbols=trading_symbols,
            portfolio_weights=final_weights,
            window_id=window_id,
            start_time_utc=t0,
            benchmark_symbol=benchmark_symbol,
            nav0=NAV0_DEFAULT,
            window_hours=WINDOW_HOURS,
        )

    # ------------------------
    # Compute monthly metrics for each touched month
    # ------------------------
    logger.info("Computing monthly metrics for touched months: %s", touched_months)

    for (year, month) in sorted(touched_months):
        logger.info("Computing monthly metrics for %04d-%02d", year, month)
        metrics = compute_monthly_performance_metrics(
            db=db,
            ch_client=ch_client,
            ch_table=ch_cfg["table"],
            year=year,
            month=month,
            benchmark_symbol=benchmark_symbol,
            trading_symbols=trading_symbols,
        )
        logger.info("Monthly metrics for %04d-%02d: %s", year, month, metrics)

    logger.info("Backtesting / bulk simulation completed.")
