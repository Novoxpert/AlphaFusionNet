#!/usr/bin/env python3
"""
Backtesting / Bulk Simulation Service
=====================================

This script back-fills MongoDB with **AlphaFusionNet metrics and windows**
over a short historical period, using:

  • **real prices** from ClickHouse
  • **real AlphaFusionNet predictions** already stored in MongoDB
    (collection: AlphaFusionNet_predictions)

Behavior
--------
- Uses the last N **calendar days before today (UTC)** (default: 90 days).
- Within that range:
    • If a day is a **common trading day** across all symbols (from the
      trading calendar), we simulate a window for it.
    • For each trading day, we fetch the **latest prediction** in
      AlphaFusionNet_predictions whose timestamp is **strictly before**
      that UTC calendar day (timestamp < day 00:00:00).
      The document's `final_weights` are used as the portfolio weights,
      and its `risk_controls` are used to compute per-asset SL/TP/CONF.
    • If no earlier prediction is found, we skip that day.

- Window definition (ALL IN UTC):
    • t0 = 14:30 UTC
    • t1 = t0 + 4 hours (18:30 UTC)
    • We compute a snapshot for **every minute**, but:
        - OHLCV is loaded ONCE from ClickHouse for the whole window,
          then used in memory.
        - All snapshots for the window are persisted in MongoDB with
          a single bulk write per collection.

Important Notes
---------------
- No extra "lookback window" pre-validation; we forward-fill within the window
- Prices come from ClickHouse only; predictions must already exist in Mongo.

Configuration
-------------
Reads config from `AFN_config.yaml`:

    • clickhouse: host, port, database, table, user, password
    • novo_mongo: host, port, database, user, password, authSource
    • market.symbols_usdt: trading universe
    • market.benchmark_symbol: e.g. ["SP:SPX"]

Usage
-----
    python -m scripts.metric_backtesting --days 90
    python -m scripts.metric_backtesting -d 30

You can adjust:
    BACKTEST_DAYS_DEFAULT      : how many calendar days back from today (UTC)
    WINDOW_START_HOUR_UTC      : start hour (default 14 for 14:00 UTC)
    WINDOW_START_MINUTE_UTC    : start minute (default 30 for 14:30 UTC)
    WINDOW_HOURS               : window length (default 4 hours)

------
Author: Elham Esmaeilnia(elham.e.shirvani@gmail.com)
Date: 2025 Nov 23
Version: 3.2.0  (risk_controls + SL/TP/CONF integration)
"""

import logging
import os
from datetime import datetime, timedelta, timezone
import argparse
import yaml
from dotenv import load_dotenv

from lib.db_utils import init_clickhouse_client, init_mongo_client
from lib.metric_utils import (
    init_window_state,
    compute_metrics_snapshot,
    persist_window_snapshots_bulk,
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

# Default number of calendar days back from today (UTC) to consider
BACKTEST_DAYS_DEFAULT = 60

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
# Prediction helper
# ------------------------
def get_prediction_for_day(db, day):
    """
    Fetch the latest AlphaFusionNet prediction whose timestamp is
    strictly BEFORE the given UTC calendar day, and return its
    `final_weights` dict AND `risk_controls` dict.

    Example:
        day = 2025-01-14
        -> search for timestamp < 2025-01-14 00:00:00 UTC
        -> take the latest one (sort by timestamp desc)

    Parameters
    ----------
    db  : MongoDB database handle
    day : datetime.date (assumed UTC calendar day)

    Returns
    -------
    (final_weights, risk_controls)
        final_weights : dict or None
        risk_controls : dict (possibly empty) or None if no doc
    """
    col = db["AlphaFusionNet_predictions"]

    # Start of this trading day in UTC
    day_start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)

    # Find the latest prediction strictly before this day_start
    doc = col.find_one(
        {
            "timestamp": {
                "$lt": day_start,
            }
        },
        sort=[("timestamp", -1)],  # latest overall before that day
    )

    if not doc:
        logger.warning(
            "No AlphaFusionNet prediction found BEFORE trading day %s "
            "(searched for timestamp < %s). Skipping this day.",
            day.isoformat(),
            day_start.isoformat(),
        )
        return None, None

    final_weights = doc.get("final_weights") or {}
    if not final_weights:
        logger.warning(
            "Prediction document used for %s (timestamp=%s) has empty final_weights. "
            "Skipping this day.",
            day.isoformat(),
            doc.get("timestamp"),
        )
        return None, None

    risk_controls = doc.get("risk_controls") or {}

    logger.info(
        "Using AlphaFusionNet prediction from %s for trading day %s "
        "(latest prediction before %s, weights for %d symbols, "
        "risk_controls for %d symbols).",
        doc.get("timestamp"),
        day.isoformat(),
        day_start.isoformat(),
        len(final_weights),
        len(risk_controls),
    )

    return final_weights, risk_controls


# ------------------------
# Price preload + window simulation
# ------------------------
def preload_window_prices(
    ch_client,
    ch_table: str,
    symbols,
    benchmark_symbol: str,
    t0: datetime,
    t1: datetime,
):
    """
    Bulk-load OHLCV close prices from ClickHouse for all symbols + benchmark
    between [t0, t1], and build a per-minute, forward-filled price grid.

    IMPORTANT: t0 and t1 are assumed to be UTC datetimes (tz-aware).
    If they are naive, they will be treated as UTC.
    """
    # Normalize to tz-aware UTC
    if t0.tzinfo is None:
        t0 = t0.replace(tzinfo=timezone.utc)
    if t1.tzinfo is None:
        t1 = t1.replace(tzinfo=timezone.utc)

    all_syms = sorted(set(symbols) | {benchmark_symbol})

    logger.info(
        "Preloading prices from ClickHouse for %d symbols from %s to %s (table=%s)",
        len(all_syms),
        t0.isoformat(),
        t1.isoformat(),
        ch_table,
    )

    # Build list of per-minute timestamps in window (keep UTC tzinfo)
    minutes = []
    cur = t0
    while cur <= t1:
        minutes.append(cur)
        cur += timedelta(minutes=1)

    # Single ClickHouse query for all symbols and the full window
    # Assumes schema: (symbol, candle_time, close) with candle_time in UTC
    query = f"""
        SELECT symbol, candle_time, close
        FROM {ch_table}
        WHERE symbol IN %(symbols)s
          AND candle_time >= %(start_ts)s
          AND candle_time <= %(end_ts)s
        ORDER BY symbol, candle_time
    """

    logger.info("DEBUG CH: querying %s for %d symbols", ch_table, len(all_syms))

    rows = ch_client.execute(
        query,
        {
            "symbols": tuple(all_syms),
            "start_ts": t0,
            "end_ts": t1,
        },
    )

    logger.info(
        "DEBUG CH: got %d raw rows from %s between %s and %s",
        len(rows),
        ch_table,
        t0.isoformat(),
        t1.isoformat(),
    )

    # Raw prices: symbol -> {timestamp -> price}
    raw = {sym: {} for sym in all_syms}
    for sym, ts, close in rows:
        raw.setdefault(sym, {})[ts] = close

    prices_by_symbol = {}

    for sym in all_syms:
        sym_raw = raw.get(sym, {})
        sym_grid = {}
        last = None
        for ts in minutes:
            if ts in sym_raw:
                last = sym_raw[ts]
            if last is not None:
                sym_grid[ts] = float(last)
        prices_by_symbol[sym] = sym_grid

        if not sym_grid:
            logger.warning(
                "No prices at all for %s in [%s, %s]. This symbol will be skipped in snapshots.",
                sym,
                t0.isoformat(),
                t1.isoformat(),
            )

    return prices_by_symbol, minutes


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
    risk_controls=None,
):
    """
    Offline simulation of a full live window [t0, t1], with per-minute snapshots,
    but using preloaded OHLCV prices from ClickHouse and bulk Mongo writes.

    - Initializes window state at t0 (including SL/TP/CONF from risk_controls
      if provided).
    - Preloads all OHLCV for [t0, t1] for symbols + benchmark.
    - Loops minute by minute using the in-memory price grid.
    - Collects all snapshots in memory.
    - Persists all snapshots in one bulk update to `windows` and `live_metrics`.

    start_time_utc MUST be a UTC datetime (tz-aware).
    """
    if start_time_utc.tzinfo is None:
        start_time_utc = start_time_utc.replace(tzinfo=timezone.utc)

    logger.info(
        "Simulating backtest window %s from %s (hours=%d)",
        window_id,
        start_time_utc.isoformat(),
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
        risk_controls=risk_controls, 
    )

    t0 = window_doc["t0"]
    t1 = window_doc["t1"]

    # Ensure UTC tzinfo for safety
    if t0.tzinfo is None:
        t0 = t0.replace(tzinfo=timezone.utc)
    if t1.tzinfo is None:
        t1 = t1.replace(tzinfo=timezone.utc)

    # Preload all prices for the window from ClickHouse
    prices_by_symbol, minutes = preload_window_prices(
        ch_client=ch_client,
        ch_table=ch_table,
        symbols=symbols,
        benchmark_symbol=benchmark_symbol,
        t0=t0,
        t1=t1,
    )

    total_minutes = len(minutes)
    logger.info(
        "Window %s runs from %s to %s (%d minutes, preloaded prices)",
        window_id,
        t0.isoformat(),
        t1.isoformat(),
        total_minutes,
    )

    # last known prices within this window (extra safety)
    last_prices = dict(window_doc.get("entry_prices", {}))
    last_benchmark_price = window_doc.get("benchmark_entry")

    # Progress checkpoints for logging (25%, 50%, 75%, 100%)
    checkpoints = set(
        max(1, int(total_minutes * frac)) for frac in (0.25, 0.5, 0.75, 1.0)
    )

    snapshots = []
    stale_batches = []

    for idx, as_of in enumerate(minutes, start=1):
        prices_t = {}
        stale_symbols = []

        # per-symbol prices from preloaded grid
        for sym in symbols:
            sym_grid = prices_by_symbol.get(sym, {})
            price = sym_grid.get(as_of)

            if price is None:
                # extra fallback: last seen in this window
                fallback = last_prices.get(sym)
                if fallback is not None:
                    price = fallback
                    stale_symbols.append(sym)
                else:
                    logger.warning(
                        "No price available for %s at %s (even after preload)",
                        sym,
                        as_of.isoformat(),
                    )
                    continue

            prices_t[sym] = price
            last_prices[sym] = price

        # benchmark price
        bench_grid = prices_by_symbol.get(benchmark_symbol, {})
        bench_price = bench_grid.get(as_of)

        bench_is_stale = False
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

        snapshots.append(snapshot)
        stale_batches.append(stale_symbols)

        if idx in checkpoints:
            logger.info(
                "Window %s progress: %d/%d minutes (%.1f%%)",
                window_id,
                idx,
                total_minutes,
                100.0 * idx / total_minutes,
            )

    # Bulk persist all snapshots for this window
    persist_window_snapshots_bulk(
        db=db,
        window_id=window_id,
        snapshots=snapshots,
        stale_symbols_per_snapshot=stale_batches,
    )

    # mark window ended
    mark_window_ended(db, window_id)
    logger.info("Completed backtest window %s", window_id)


# ------------------------
# Entrypoint
# ------------------------
if __name__ == "__main__":
    # ------------------------
    # CLI arguments
    # ------------------------
    parser = argparse.ArgumentParser(
        description="Backfill AlphaFusionNet windows and metrics over past N days."
    )
    parser.add_argument(
        "--days",
        "-d",
        type=int,
        default=BACKTEST_DAYS_DEFAULT,
        help=f"How many calendar days back from today (UTC) to backtest (default: {BACKTEST_DAYS_DEFAULT})",
    )
    args = parser.parse_args()
    backtest_days = args.days

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

    logger.info(
        "Using ClickHouse database=%s table=%s",
        ch_cfg["database"],
        ch_cfg["table"],
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
        # Determine backtest date range (last BACKTEST_DAYS calendar days, UTC)
        # ------------------------
        today_utc = datetime.now(timezone.utc).date()
        end_date = today_utc - timedelta(days=1)  # yesterday (UTC)
        start_date = today_utc - timedelta(days=backtest_days)

        logger.info(
            "Calendar backtest range (UTC calendar days): %s to %s",
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
        logger.info("Total trading days (potential windows): %d", total_windows)

        # ------------------------
        # Run windows
        # ------------------------
        windows_run = 0

        try:
            for idx, d in enumerate(all_dates, start=1):
                logger.info(
                    "[%d/%d] Preparing window for trading date %s",
                    idx,
                    total_windows,
                    d.isoformat(),
                )

                # 1) Get portfolio weights + risk_controls from latest AlphaFusionNet_predictions BEFORE this day
                final_weights, risk_controls = get_prediction_for_day(db, d)
                if not final_weights:
                    # already logged inside helper
                    continue

                # If you want to restrict to configured trading_symbols, uncomment:
                # portfolio_weights = {
                #     sym: float(w)
                #     for sym, w in final_weights.items()
                #     if sym in trading_symbols
                # }
                # if not portfolio_weights:
                #     logger.warning(
                #         "Prediction used for %s has no overlap with configured trading_symbols; skipping.",
                #         d.isoformat(),
                #     )
                #     continue
                #
                # # Also filter risk_controls to the same symbol set
                # risk_controls = {
                #     sym: rc
                #     for sym, rc in (risk_controls or {}).items()
                #     if sym in portfolio_weights
                # }

                portfolio_weights = final_weights

                # 2) Build t0 as UTC-aware datetime
                t0 = datetime(
                    d.year,
                    d.month,
                    d.day,
                    WINDOW_START_HOUR_UTC,
                    WINDOW_START_MINUTE_UTC,
                    0,
                    tzinfo=timezone.utc,
                )
                t1 = t0 + timedelta(hours=WINDOW_HOURS)

                logger.info(
                    "[%d/%d] Using portfolio weights from AlphaFusionNet_predictions. "
                    "Simulating window for trading date %s (t0=%s, t1=%s, UTC)",
                    idx,
                    total_windows,
                    d.isoformat(),
                    t0.isoformat(),
                    t1.isoformat(),
                )

                # window_id consistent with live service (YYYYMMDD_%H%M)
                window_id = f"{t0:%Y%m%d_%H%M}"

                run_backtest_window(
                    db=db,
                    ch_client=ch_client,
                    ch_table=ch_cfg["table"],
                    symbols=trading_symbols,
                    portfolio_weights=portfolio_weights,
                    window_id=window_id,
                    start_time_utc=t0,
                    benchmark_symbol=benchmark_symbol,
                    nav0=NAV0_DEFAULT,
                    window_hours=WINDOW_HOURS,
                    risk_controls=risk_controls, 
                )

                touched_months.add((d.year, d.month))
                windows_run += 1

        except KeyboardInterrupt:
            logger.warning(
                "Backtest interrupted by user (Ctrl+C). "
                "Current window finishes; remaining days are skipped."
            )

        # ------------------------
        # Compute monthly metrics for each touched month
        # ------------------------
        if touched_months and windows_run > 0:
            logger.info(
                "Computing monthly metrics for months touched by at least one window: %s",
                touched_months,
            )

            for (year, month) in sorted(touched_months):
                logger.info("Computing monthly metrics for %04d-%02d", year, month)
                metrics = compute_monthly_performance_metrics(
                    db=db,
                    year=year,
                    month=month,
                    benchmark_symbol=benchmark_symbol,
                    trading_symbols=trading_symbols,
                )
                logger.info("Monthly metrics for %04d-%02d: %s", year, month, metrics)
        else:
            logger.info(
                "No windows run (no predictions or no trading days); skipping monthly metrics."
            )

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
