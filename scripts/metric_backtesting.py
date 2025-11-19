"""
Backtesting / Bulk Simulation Service
=====================================

This script back-fills MongoDB with **synthetic AlphaFusionNet predictions**
and simulated **live windows** over a short historical period, using **real prices**
from ClickHouse.

Current Behavior (simplified)
-----------------------------
- Only uses the **last N calendar days before today (UTC)** (default: 5 days).
- Within that range:
    • If a day is a **common trading day** across all symbols (from the
      trading calendar), we simulate a window for it.
    • If it is NOT a trading day, we simply skip it.

- Window definition:
    • t0 = 14:30 UTC
    • t1 = t0 + 4 hours (18:30 UTC)

- For each trading day:
    1. Create a dummy AlphaFusionNet_predictions document at t1.
    2. Initialize a window with entry prices at t0.
    3. Simulate minute-by-minute from t0 to t1 using:
         - get_minute_close_price
         - compute_metrics_snapshot
         - persist_window_snapshot
    4. Mark the window as ENDED.
    5. After all windows, compute monthly metrics for months touched.

Important Notes
---------------
- We do NOT do extra "lookback window" pre-validation here. If ClickHouse
  has gaps, the internal metric utils will handle fallbacks (or log warnings).
- We reduce noisy logs from lib.metric_utils / c_utils to avoid console spam.
- We handle Ctrl+C gracefully, without doing weird things in the signal handler.
- Mongo is empty initially? Totally fine. We only depend on ClickHouse for prices.

Configuration
-------------
Reads config from `AFN_config.yaml`:

    • clickhouse: host, port, database, table, user, password
    • novo_mongo: host, port, database, user, password, authSource
    • market.symbols_usdt: trading universe
    • market.benchmark_symbol: e.g. ["SP:SPX"]

Usage
-----
    python -m scripts.metric_backtesting

You can adjust:
    BACKTEST_DAYS              : how many calendar days back from today (UTC)
    WINDOW_START_HOUR_UTC      : start hour (default 14 for 14:00 UTC)
    WINDOW_START_MINUTE_UTC    : start minute (default 30 for 14:30 UTC)
    WINDOW_HOURS               : window length (default 4 hours)
------
Author: Elham Esmaeilnia
Version: 2.1.0 (5-day backtest, 14:30–18:30 UTC window, calendar-only trading day filter)
"""

import logging
import os
from datetime import datetime, timedelta, timezone

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

# calendar-based trading days for backtest
from scripts.compute_trading_days_service import (
    compute_common_trading_days_between,
)

# ------------------------
# Basic logging
# ------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Reduce noisy warnings from inner metric code during backtest
logging.getLogger("lib.metric_utils").setLevel(logging.ERROR)
logging.getLogger("c_utils").setLevel(logging.ERROR)

CONFIG_FILE = os.environ.get("ALPHAFUSIONNET_CONFIG", "config/AFN_config.yaml")

# How many calendar days back from today (UTC) to consider
BACKTEST_DAYS = 5

# Window definition: [14:30, 18:30] UTC
WINDOW_START_HOUR_UTC = 14
WINDOW_START_MINUTE_UTC = 30
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
    Offline simulation of a full live window [t0, t1].

    - Initializes window state at t0.
    - Loops minute by minute from t0 to t1.
    - At each minute:
        * fetches prices via get_minute_close_price (with fallbacks),
        * computes metrics snapshot,
        * persists to Mongo.

    start_time_utc is a naive datetime interpreted as UTC.
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

    total_minutes = int((t1 - t0).total_seconds() // 60) + 1
    logger.info(
        "Window %s runs from %s to %s (%d minutes)",
        window_id,
        t0.isoformat(),
        t1.isoformat(),
        total_minutes,
    )

    # last known prices for fallback logic
    last_prices = dict(window_doc.get("entry_prices", {}))
    last_benchmark_price = window_doc.get("benchmark_entry")

    # Progress checkpoints for logging (25%, 50%, 75%, 100%)
    checkpoints = set(
        max(1, int(total_minutes * frac)) for frac in (0.25, 0.5, 0.75, 1.0)
    )

    as_of = t0
    minute_idx = 0

    while as_of <= t1:
        minute_idx += 1
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
                # Fallback: last known price in this window
                fallback = last_prices.get(sym)
                if fallback is not None:
                    price = fallback
                    is_stale = True
                else:
                    # rare: absolutely no price seen yet for this symbol in this window
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

        if minute_idx in checkpoints:
            logger.info(
                "Window %s progress: %d/%d minutes (%.1f%%)",
                window_id,
                minute_idx,
                total_minutes,
                100.0 * minute_idx / total_minutes,
            )

        as_of += timedelta(minutes=1)

    # mark window ended
    mark_window_ended(db, window_id)
    logger.info("Completed backtest window %s", window_id)


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
    mongo_cfg = cfg["novo_mongo"]
    mongo_client, db = init_mongo_client(mongo_cfg)

    try:
        # Market config
        market_cfg = cfg.get("market", {})
        trading_symbols = market_cfg.get("symbols_usdt", [])
        benchmark_symbols = market_cfg.get("benchmark_symbol", ["SP:SPX"])
        benchmark_symbol = benchmark_symbols[0]

        if not trading_symbols:
            raise RuntimeError("No trading symbols defined in market.symbols_usdt")

        logger.info("Backtesting on %d symbols", len(trading_symbols))

        # ------------------------
        # Determine backtest date range (last BACKTEST_DAYS calendar days)
        # ------------------------
        today_utc = datetime.now(timezone.utc).date()
        end_date = today_utc - timedelta(days=1)        # yesterday
        start_date = today_utc - timedelta(days=BACKTEST_DAYS)

        logger.info(
            "Calendar backtest range (calendar days): %s to %s",
            start_date.isoformat(),
            end_date.isoformat(),
        )

        # Get common trading days in this fixed calendar range
        all_trading_days = compute_common_trading_days_between(
            start_date.isoformat(),
            end_date.isoformat(),
        )

        if not all_trading_days:
            logger.warning(
                "No common trading days between %s and %s; exiting.",
                start_date.isoformat(),
                end_date.isoformat(),
            )
            raise SystemExit(0)

        # These are already trading days only. No extra checks.
        all_dates = sorted(all_trading_days)

        logger.info(
            "Found %d trading days in range: %s ... %s",
            len(all_dates),
            all_dates[0].isoformat(),
            all_dates[-1].isoformat(),
        )

        touched_months = set()
        total_windows = len(all_dates)
        logger.info("Total windows (days) to simulate: %d", total_windows)

        # ------------------------
        # Run windows
        # ------------------------
        try:
            for idx, d in enumerate(all_dates, start=1):
                touched_months.add((d.year, d.month))

                t0 = datetime(
                    d.year,
                    d.month,
                    d.day,
                    WINDOW_START_HOUR_UTC,
                    WINDOW_START_MINUTE_UTC,
                    0,
                )
                t1 = t0 + timedelta(hours=WINDOW_HOURS)

                logger.info(
                    "[%d/%d] Preparing window for trading date %s (t0=%s, t1=%s)",
                    idx,
                    total_windows,
                    d.isoformat(),
                    t0.isoformat(),
                    t1.isoformat(),
                )

                # Equal-weight dummy portfolio over all trading symbols
                final_weights = generate_equal_weights(trading_symbols)

                # Insert dummy AlphaFusionNet_prediction at t1
                pred_ts = t1
                insert_dummy_prediction(db, pred_ts, final_weights)

                # window_id consistent with live service (YYYYMMDD_%H%M)
                window_id = f"{t0:%Y%m%d_%H%M}"

                logger.info(
                    "[%d/%d] Simulating window %s for trading date %s",
                    idx,
                    total_windows,
                    window_id,
                    d.isoformat(),
                )

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

        except KeyboardInterrupt:
            logger.warning(
                "Backtest interrupted by user (Ctrl+C). "
                "Current window finishes; remaining days are skipped."
            )

        # ------------------------
        # Compute monthly metrics for each touched month
        # ------------------------
        if touched_months:
            logger.info(
                "Computing monthly metrics for touched months: %s", touched_months
            )

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
        else:
            logger.info("No months touched (no windows run); skipping monthly metrics.")

        logger.info("Backtesting / bulk simulation completed.")

    finally:
        # Clean shutdown of DB clients
        try:
            mongo_client.close()
        except Exception:
            pass
        try:
            if hasattr(ch_client, "disconnect"):
                ch_client.disconnect()
            elif hasattr(ch_client, "close"):
                ch_client.close()
        except Exception:
            pass
