"""
Metric Utilities for AlphaFusionNet
===================================

This module implements all shared logic for **portfolio metric computation**
in the AlphaFusionNet project. It covers:

    1. Live 4-hour window valuation and per-minute snapshots.
    2. Month-to-date (MTD) performance aggregation and monthly KPIs.
    3. Helper utilities for reading OHLCV data from ClickHouse and
       persisting state/metrics into MongoDB.

High-Level Responsibilities
---------------------------
1. **ClickHouse helpers**
   - `fetch_ohlcv_range`:
       Generic range query for OHLCV candles, using a `candle_time`
       timestamp and returning a pandas DataFrame with a normalized
       `dateTime` column.
   - `get_minute_close_price`:
       Fetches the close price for a specific minute (`as_of`), with an
       optional short lookback window and a "stale" flag if it falls
       back to an older candle.

2. **Live 4-Hour Window Initialization**
   - `compute_positions_from_weights`:
       Given portfolio weights, entry prices, and initial NAV, computes
       per-symbol capital allocation, position quantities `q_k`, and
       initial cash reserve. This assumes **no intra-window rebalancing**.
   - `init_window_state`:
       Creates (or reuses) a document in MongoDB `windows` that represents
       a single live window:
           • window_id, t0, t1
           • entry_prices at t0 (one candle exactly at t0 per symbol)
           • positions (q_k, allocated_capital, p_entry, weight)
           • cash_initial and nav_initial
       Entry prices are **fixed at t0** and remain constant throughout
       the 4-hour window.

3. **Per-Minute Live Valuation**
   - `compute_portfolio_value`:
       Computes the live NAV:
           V(t) = Σ_k [ q_k * P_k(t) ] + cash
       plus the per-symbol marked-to-market value.
   - `compute_metrics_snapshot`:
       Using a `window_doc` and current minute prices:
           • NAV live vs NAV0
           • portfolio return Rp(t), PnL(t)
           • benchmark return R_b(t) and alpha vs benchmark
           • per-symbol contribution Contr_k(t)
           • per-symbol current prices for the dashboard
       The result is a serializable dict (snapshot) keyed by `as_of`.
   - `persist_window_snapshot`:
       Persists a snapshot into:
           • `windows.live_history` (embedded array on the window doc)
           • `live_metrics` collection (flattened for quick dashboard reads)
       Also tracks the count of stale symbols and timestamps.
   - `mark_window_ended`:
       Marks a given window as ENDED in the `windows` collection when the
       4-hour live interval is over.

4. **Monthly / Day-in-Month Performance Metrics**
   - `_daily_open_close_from_minute_prices`:
       Converts minute-level OHLC data per symbol into **daily returns**:
           daily_return = close / open - 1
       using first and last close of each calendar day.
   - `compute_portfolio_daily_returns`:
       Combines daily symbol returns with daily weights (from window docs)
       to produce a **daily portfolio return time series** for a month:
           portfolio_return[date] = Σ_s W[date, s] * R[date, s]

   - Basic metrics on the daily series:
       • `compute_winning_percentage`:
             % of days with portfolio_return > 0
       • `compute_longest_positive_streak`:
             longest streak of consecutive positive daily returns
       • `compute_rolling_return_consistency`:
             % of rolling windows (default 5-day) whose mean return > 0

   - `compute_monthly_performance_metrics`:
       Main MTD (month-to-date) aggregator:

           1. Select all `windows` whose `t0` lies in the target month.
           2. Extract daily portfolio weights from those windows.
           3. Pull minute-level prices for the month from ClickHouse
              for all trading symbols.
           4. Aggregate to daily symbol returns and compute the
              portfolio daily return series.
           5. Calculate:
                 • winning_percentage_pct
                 • consistency_score_periods (max positive streak)
                 • rolling_return_consistency_pct (5-day MAs > 0)
                 • n_trading_days_in_month
                 • first_trading_day / last_trading_day
           6. Upsert a single document in MongoDB `monthly` collection
              for each `month_id = "YYYY-MM"`, containing:
                 • last_metrics           : latest metrics snapshot
                 • last_snapshot_date     : date of this computation (UTC)
                 • portfolio_daily_returns: {date -> daily_return}
                 • metrics_history        : array of past snapshots
                   (one per day the service ran)

       This structure enables the dashboard to show both:
           • current **month-to-date KPIs** (from `last_metrics`)
           • historical evolution of these KPIs during the month
             (from `metrics_history`).

Data Model Summary
------------------
MongoDB collections touched by this module:

1. `windows`
   One document per live 4-hour window.
   Stores entry state, positions, and an embedded `live_history` array
   of minute-level snapshots.

2. `live_metrics`
   One document per minute per window (flattened).
   Optimized for front-end queries like:
       - "latest snapshot for current window"
       - "time series of NAV over the last window"

3. `monthly`
   One document per calendar month (`month_id = "YYYY-MM"`).
   Stores:
       - month-to-date daily portfolio returns
       - latest monthly metrics
       - full history of daily metric snapshots (`metrics_history`).

Constants
---------
• NAV0_DEFAULT: default initial capital (100_000.0 USD)  
• ROLLING_WINDOW_DAYS: default rolling window length (5 days) for
  consistency calculations.

This module is intentionally pure in terms of business logic: I/O is
limited to Mongo + ClickHouse, and the functions are structured to be
reused by both the live metric service and the monthly metric service.

-------
Author: Elham Esmaeilnia(elham.e.shirvani@gmail.com)
Date: 2025 Nov 17
Version: 1.0.0 

"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

NAV0_DEFAULT = 100_000.0
ROLLING_WINDOW_DAYS = 5


# ------------------------
# ClickHouse helpers
# ------------------------
def fetch_ohlcv_range(
    ch_client,
    ch_table: str,
    symbol: str,
    start_utc: datetime,
    end_utc: datetime,
) -> pd.DataFrame:
    """
    Fetch OHLCV data for a symbol between start_utc and end_utc (inclusive).
    Assumes ClickHouse has a 'candle_time' column and uses UTC timestamps.

    NOTE:
      - If start_utc == end_utc, this effectively fetches the OHLCV row
        at exactly that timestamp (the window entry candle).
    """
    # make sure we don't accidentally have microseconds mismatch
    start_utc = start_utc.replace(microsecond=0)
    end_utc = end_utc.replace(microsecond=0)

    start_str = start_utc.strftime("%Y-%m-%d %H:%M:%S")
    end_str = end_utc.strftime("%Y-%m-%d %H:%M:%S")

    q = f"""
    SELECT *
    FROM {ch_table}
    WHERE symbol = '{symbol}'
      AND candle_time >= '{start_str}'
      AND candle_time <= '{end_str}'
    ORDER BY candle_time ASC
    """
    data = ch_client.execute(q)
    if not data:
        return pd.DataFrame()

    cols = [c[0] for c in ch_client.execute(f"DESCRIBE TABLE {ch_table}")]
    df = pd.DataFrame(data, columns=cols)
    df["dateTime"] = pd.to_datetime(df["candle_time"], utc=True)
    return df


def get_minute_close_price(
    ch_client,
    ch_table: str,
    symbol: str,
    as_of: datetime,
    lookback_minutes: int = 2,
) -> Tuple[Optional[float], bool]:
    """
    Return the close price at 'as_of' minute.
    If missing, fallback to last known within lookback_minutes.
    Returns (price, is_stale).
    """
    # align to minute
    as_of = as_of.replace(second=0, microsecond=0)
    start_utc = as_of - timedelta(minutes=lookback_minutes)

    df = fetch_ohlcv_range(ch_client, ch_table, symbol, start_utc, as_of)
    if df.empty:
        logger.warning("No OHLCV data for %s in lookback window ending at %s", symbol, as_of)
        return None, True

    # Prefer exact as_of if exists, otherwise last row in window
    df_sorted = df.sort_values("dateTime")
    exact = df_sorted[df_sorted["dateTime"] == as_of]
    if not exact.empty:
        price = float(exact.iloc[-1]["close"])
        return price, False

    # Use last known
    price = float(df_sorted.iloc[-1]["close"])
    return price, True


# ------------------------
# Portfolio initialization
# ------------------------
def compute_positions_from_weights(
    portfolio_weights: Dict[str, float],
    entry_prices: Dict[str, float],
    nav0: float = NAV0_DEFAULT,
) -> Tuple[Dict[str, Dict[str, float]], float, float]:
    """
    Given weights and entry prices, compute:
      positions: {symbol: {"q": quantity, "allocated_capital": A_k, "p_entry": P_k(t0)}}
      cash0    : NAV0 * (1 - sum_w)
      sum_w    : total weights
    """
    positions: Dict[str, Dict[str, float]] = {}
    sum_w = float(sum(portfolio_weights.values()))

    for symbol, w_k in portfolio_weights.items():
        p0 = entry_prices.get(symbol)
        if p0 is None or np.isnan(p0) or p0 <= 0:
            logger.warning("Invalid entry price for %s, skipping position", symbol)
            continue

        A_k = nav0 * w_k
        q_k = A_k / p0
        positions[symbol] = {
            "q": float(q_k),
            "allocated_capital": float(A_k),
            "p_entry": float(p0),
            "weight": float(w_k),
        }

    cash0 = nav0 * (1.0 - sum_w)
    return positions, cash0, sum_w

def compute_sl_tp_levels(
    portfolio_weights: Dict[str, float],
    entry_prices: Dict[str, float],
    risk_controls: Dict[str, Dict[str, float]],
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Compute per-symbol SL/TP *price levels* and confidence.

    Inputs:
      - portfolio_weights: {symbol -> weight}
      - entry_prices     : {symbol -> entry_price at t0}
      - risk_controls    : {
                              symbol: {
                                  "SL": sl_fraction,
                                  "TP": tp_fraction,
                                  "CONF": confidence
                              },
                           }

    Logic (per symbol s):
      let p0 = entry price
      let w  = weight
      let sl = abs(SL), tp = abs(TP)  (fractions, e.g. 0.05 -> 5%)

      if w >= 0 (long):
          SL_price = p0 * (1 - sl)
          TP_price = p0 * (1 + tp)
      else (short):
          SL_price = p0 * (1 + sl)
          TP_price = p0 * (1 - tp)

    Returns:
      sl_prices      : {symbol -> SL_price}
      tp_prices      : {symbol -> TP_price}
      confidences    : {symbol -> CONF}
    """
    sl_prices: Dict[str, float] = {}
    tp_prices: Dict[str, float] = {}
    confidences: Dict[str, float] = {}

    for sym, w in portfolio_weights.items():
        p0 = entry_prices.get(sym)
        rc = risk_controls.get(sym, {})

        if p0 is None:
            continue

        sl_raw = rc.get("SL")
        tp_raw = rc.get("TP")
        conf = rc.get("CONF")

        # Require both SL and TP to compute levels
        if sl_raw is None or tp_raw is None:
            continue

        # Use absolute value so negative/positive conventions don't break math
        sl = abs(float(sl_raw))
        tp = abs(float(tp_raw))

        # Long vs short logic
        if w >= 0:
            sl_price = p0 * (1.0 - sl)
            tp_price = p0 * (1.0 + tp)
        else:
            sl_price = p0 * (1.0 + sl)
            tp_price = p0 * (1.0 - tp)

        sl_prices[sym] = float(sl_price)
        tp_prices[sym] = float(tp_price)

        if conf is not None:
            confidences[sym] = float(conf)

    return sl_prices, tp_prices, confidences

def init_window_state(
    db,
    ch_client,
    ch_table: str,
    window_id: str,
    symbols: List[str],
    portfolio_weights: Dict[str, float],
    benchmark_symbol: str,
    start_time_utc: datetime,
    nav0: float = NAV0_DEFAULT,
    window_hours: int = 4,
    risk_controls: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict:
    """
    Initialize and persist a window document in Mongo 'windows' collection.

    - Reads entry close prices **exactly at t0** (start_time_utc) for all symbols & benchmark.
    - Computes positions and cash0.
    """
    windows_col = db["windows"]

    existing = windows_col.find_one({"window_id": window_id})
    if existing:
        logger.info("Window %s already exists in DB, reusing", window_id)
        return existing

    if risk_controls is None:
        risk_controls = {}

    # Align t0 to minute boundary to match candle_time
    start_time_utc = start_time_utc.replace(second=0, microsecond=0)

    entry_prices: Dict[str, float] = {}
    for sym in symbols:
        # Use ONLY the candle at exactly t0
        df = fetch_ohlcv_range(
            ch_client,
            ch_table,
            sym,
            start_time_utc,
            start_time_utc,  # start == end -> only t0 candle
        )
        if df.empty:
            logger.warning("No entry price found for %s at t0=%s", sym, start_time_utc)
            continue

        # we expect a single row at that timestamp; use its close
        p0 = float(df.iloc[0]["close"])
        entry_prices[sym] = p0

    # Benchmark entry price, also exactly at t0
    df_bench = fetch_ohlcv_range(
        ch_client,
        ch_table,
        benchmark_symbol,
        start_time_utc,
        start_time_utc,  # only t0 candle
    )
    benchmark_entry = float(df_bench.iloc[0]["close"]) if not df_bench.empty else None

    # Compute initial positions
    positions, cash0, total_weight = compute_positions_from_weights(
        portfolio_weights, entry_prices, nav0=nav0
    )

    # Compute SL/TP levels and confidence per symbol from risk_controls
    sl_prices, tp_prices, confidences = compute_sl_tp_levels(
        portfolio_weights=portfolio_weights,
        entry_prices=entry_prices,
        risk_controls=risk_controls,
    )

    # Enrich positions with SL/TP/CONF for convenience
    for sym, pos in positions.items():
        if sym in sl_prices:
            pos["sl_price"] = sl_prices[sym]
        if sym in tp_prices:
            pos["tp_price"] = tp_prices[sym]
        if sym in confidences:
            pos["confidence"] = confidences[sym]

    window_doc = {
        "window_id": window_id,
        "timestamp": start_time_utc,
        "t0": start_time_utc,
        "t1": start_time_utc + timedelta(hours=window_hours),
        "nav_initial": float(nav0),
        "portfolio_weights": portfolio_weights,
        "total_weight": float(total_weight),
        "entry_prices": entry_prices,
        "positions": positions,
        "cash_initial": float(cash0),
        "benchmark_symbol": benchmark_symbol,
        "benchmark_entry": benchmark_entry,
        "portfolio_confidence": confidences,  
        "sl_prices": sl_prices,             
        "tp_prices": tp_prices,              
        "rp_t0": 0.0,
        "pnl_t0": 0.0,
        "stale_count": 0,
        "status": "LIVE",
        # each snapshot is per-minute metrics
        "live_history": [],
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }

    windows_col.insert_one(window_doc)
    logger.info("Initialized window %s with %d positions", window_id, len(positions))

    return window_doc


# ------------------------
# Per-minute valuation
# ------------------------
def compute_portfolio_value(
    positions: Dict[str, Dict[str, float]],
    prices_t: Dict[str, float],
    cash0: float,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute portfolio live NAV and per-symbol value contributions.
    """
    value_per_symbol: Dict[str, float] = {}
    total_value = cash0

    for sym, pos in positions.items():
        q = pos["q"]
        p_t = prices_t.get(sym)
        if p_t is None:
            continue
        v_k = q * p_t
        value_per_symbol[sym] = v_k
        total_value += v_k

    return float(total_value), value_per_symbol


def compute_metrics_snapshot(
    window_doc: Dict,
    prices_t: Dict[str, float],
    benchmark_price_t: Optional[float],
    as_of: datetime,
) -> Dict:
    """
    Given window_doc state and current prices, compute full metric snapshot:
      - NAV, return, PnL
      - benchmark return, alpha
      - per-symbol contribution
      - per-symbol current prices (close) for dashboard
    """
    nav0 = float(window_doc["nav_initial"])
    cash0 = float(window_doc["cash_initial"])
    positions = window_doc["positions"]
    entry_prices = window_doc.get("entry_prices", {})
    benchmark_entry = window_doc.get("benchmark_entry")

    nav_t, value_per_symbol = compute_portfolio_value(positions, prices_t, cash0)

    rp_t = nav_t / nav0 - 1.0
    pnl_t = nav_t - nav0

    # benchmark & alpha
    r_b = None
    alpha = None
    if benchmark_entry and benchmark_price_t:
        r_b = benchmark_price_t / benchmark_entry - 1.0
        alpha = rp_t - r_b

    # per-symbol contribution
    contrib: Dict[str, float] = {}
    for sym, pos in positions.items():
        q = pos["q"]
        p0 = entry_prices.get(sym)
        p_t = prices_t.get(sym)
        if p0 is None or p_t is None:
            continue
        contrib[sym] = q * (p_t - p0) / nav0

    snapshot = {
        "as_of": as_of,
        "window_id": window_doc["window_id"],
        "nav_initial": nav0,
        "nav_live": nav_t,
        "cash": cash0,
        "portfolio_return": rp_t,
        "pnl": pnl_t,
        "benchmark_price": benchmark_price_t,
        "benchmark_return": r_b,
        "alpha_vs_benchmark": alpha,
        # explicit for dashboard: per-symbol current close prices
        "symbol_prices": prices_t,
        # keep old key name for convenience / backward compat
        "prices": prices_t,
        "value_per_symbol": value_per_symbol,
        "contribution_per_symbol": contrib,
    }
    return snapshot


def persist_window_snapshot(
    db,
    window_id: str,
    snapshot: Dict,
    stale_symbols: List[str],
):
    """
    Update 'windows' document with new snapshot and stale info.
    Also write into 'live_metrics' collection for dashboard.
    """
    windows_col = db["windows"]
    live_col = db["live_metrics"]

    snapshot_doc = {
        **snapshot,
        "stale_symbols": stale_symbols,
        "stale_count": len(stale_symbols),
        "created_at": datetime.utcnow(),
    }

    # Append to window history
    windows_col.update_one(
        {"window_id": window_id},
        {
            "$push": {"live_history": snapshot_doc},
            "$set": {
                "stale_count": len(stale_symbols),
                "updated_at": datetime.utcnow(),
            },
        },
    )

    # Also store a flattened live snapshot collection
    live_col.insert_one(snapshot_doc)


def mark_window_ended(db, window_id: str):
    windows_col = db["windows"]
    windows_col.update_one(
        {"window_id": window_id},
        {"$set": {"status": "ENDED", "ended_at": datetime.utcnow()}},
    )
    logger.info("Window %s marked as ENDED", window_id)


# ------------------------
# Monthly performance metrics
# ------------------------
def _daily_open_close_from_minute_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given minute-level OHLCV data with columns:
      ['symbol', 'dateTime', 'close']
    compute daily open/close and return per symbol:
      daily_return = close / open - 1
    Returns DataFrame: index=date, columns=['symbol', 'daily_return'] (long format).
    """
    df = df.copy()
    df["date"] = df["dateTime"].dt.date

    # sort by time to get first and last
    df = df.sort_values(["symbol", "date", "dateTime"])

    grouped = df.groupby(["symbol", "date"])
    open_price = grouped["close"].first()
    close_price = grouped["close"].last()

    daily = pd.DataFrame({"open": open_price, "close": close_price}).reset_index()
    daily["daily_return"] = daily["close"] / daily["open"] - 1.0

    return daily[["date", "symbol", "daily_return"]]


def compute_portfolio_daily_returns(
    daily_returns_long: pd.DataFrame,
    daily_weights_long: pd.DataFrame,
) -> pd.Series:
    """
    Compute daily portfolio returns:

      portfolio_return[date] = sum_s( W[date, symbol] * R[date, symbol] )

    Inputs:
      daily_returns_long: columns ['date', 'symbol', 'daily_return']
      daily_weights_long: columns ['date', 'symbol', 'weight']

    Returns:
      pd.Series indexed by date with portfolio returns.
    """
    # pivot to wide matrices
    R = daily_returns_long.pivot(index="date", columns="symbol", values="daily_return")
    W = daily_weights_long.pivot(index="date", columns="symbol", values="weight")

    # align
    W = W.reindex_like(R).fillna(0.0)

    # element-wise multiply and row-wise sum
    port_ret = (W * R).sum(axis=1)
    port_ret.name = "portfolio_return"
    return port_ret.sort_index()


def compute_winning_percentage(portfolio_returns: pd.Series) -> float:
    wins = (portfolio_returns > 0).sum()
    n = len(portfolio_returns)
    if n == 0:
        return 0.0
    return 100.0 * wins / n


def compute_longest_positive_streak(portfolio_returns: pd.Series) -> int:
    """
    Longest run of consecutive positive daily returns.
    """
    positive = portfolio_returns > 0
    max_streak = 0
    current = 0
    for is_win in positive:
        if is_win:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return int(max_streak)


def compute_rolling_return_consistency(
    portfolio_returns: pd.Series,
    window: int = ROLLING_WINDOW_DAYS,
) -> float:
    """
    % of rolling windows with positive average return.
    """
    if len(portfolio_returns) < window:
        return 0.0

    rolling_mean = portfolio_returns.rolling(window=window).mean().dropna()
    if rolling_mean.empty:
        return 0.0
    n_pos = (rolling_mean > 0).sum()
    n_total = len(rolling_mean)
    return 100.0 * n_pos / n_total


def _extract_price_series_from_windows(
    month_windows: List[Dict],
    trading_symbols: List[str],
    first_day: datetime,
    next_month: datetime,
) -> pd.DataFrame:
    """
    Build a minute-level price DataFrame from windows.live_history,
    using the *actual* symbol prices used in live snapshots.

    Returns a DataFrame with columns:
      ['symbol', 'dateTime', 'close']
    where:
      - dateTime is the snapshot 'as_of' timestamp
      - close is the price used for that symbol at that minute
    """
    records = []

    for win in month_windows:
        live_history = win.get("live_history", []) or []
        for snap in live_history:
            as_of = snap.get("as_of")
            if as_of is None:
                continue

            # Ensure we only take snapshots within [first_day, next_month)
            # (Mongo should already give datetime objects for as_of)
            if not (first_day <= as_of < next_month):
                continue

            prices = snap.get("symbol_prices") or snap.get("prices") or {}
            if not prices:
                continue

            for sym, p in prices.items():
                if sym not in trading_symbols:
                    continue
                try:
                    price = float(p)
                except (TypeError, ValueError):
                    continue

                records.append(
                    {
                        "symbol": sym,
                        "dateTime": as_of,
                        "close": price,
                    }
                )

    if not records:
        return pd.DataFrame(columns=["symbol", "dateTime", "close"])

    df = pd.DataFrame(records)
    return df

def compute_monthly_performance_metrics(
    db,
    year: int,
    month: int,
    benchmark_symbol: str,   # currently unused; placeholder for future alpha vs benchmark
    trading_symbols: List[str],
) -> Dict:
    """
    High-level monthly metrics calculator (month-to-date).

    BASED ON LIVE SNAPSHOTS (windows.live_history), NOT DIRECTLY ON CLICKHOUSE.

    - Uses the *actual minute prices* that were used in live computation
      (including fallback logic for missing prices).
    - Aggregates these minute-level prices to daily open/close per symbol.
    - Uses daily portfolio weights from window docs (one per trading day).
    - Computes:
        winning_percentage_pct
        consistency_score_periods
        rolling_return_consistency_pct
        n_trading_days_in_month
        first_trading_day
        last_trading_day

    Side effects:
      - Upserts a single document per month_id in db['monthly'].
      - Document contains:
          * last_metrics            : latest metrics snapshot for the month
          * last_snapshot_date      : yyyy-mm-dd of this run (UTC date)
          * portfolio_daily_returns : mapping date -> return
          * metrics_history         : array of all past snapshots for the month

    """
    windows_col = db["windows"]
    monthly_col = db["monthly"]

    # Determine date range for the month (calendar, not trading days)
    first_day = datetime(year, month, 1)
    if month == 12:
        next_month = datetime(year + 1, 1, 1)
    else:
        next_month = datetime(year, month + 1, 1)
    last_day = next_month - timedelta(days=1)

    # Fetch all windows whose t0 is in this month
    month_windows = list(
        windows_col.find(
            {
                "t0": {
                    "$gte": first_day,
                    "$lt": next_month,
                }
            }
        )
    )
    snapshot_date = datetime.utcnow().date().isoformat()

    if not month_windows:
        logger.warning("No window docs found for %04d-%02d", year, month)
        metrics = {
            "winning_percentage_pct": 0.0,
            "consistency_score_periods": 0,
            "rolling_return_consistency_pct": 0.0,
            "n_trading_days_in_month": 0,
            "first_trading_day": None,
            "last_trading_day": None,
        }

        month_id = f"{year:04d}-{month:02d}"
        monthly_col.update_one(
            {"month_id": month_id},
            {
                "$set": {
                    "month_id": month_id,
                    "year": year,
                    "month": month,
                    "last_snapshot_date": snapshot_date,
                    "last_metrics": metrics,
                    "portfolio_daily_returns": {},
                    "updated_at": datetime.utcnow(),
                },
                "$setOnInsert": {"created_at": datetime.utcnow()},
                "$push": {
                    "metrics_history": {
                        "snapshot_date": snapshot_date,
                        "metrics": metrics,
                        "created_at": datetime.utcnow(),
                    }
                },
            },
            upsert=True,
        )
        return metrics

    # ------------------------
    # Build daily weights from window docs
    # ------------------------
    weights_records = []
    for win in month_windows:
        date_key = win["t0"].date()
        w = win.get("portfolio_weights", {})
        for sym, w_k in w.items():
            weights_records.append(
                {"date": date_key, "symbol": sym, "weight": float(w_k)}
            )

    daily_weights_long = pd.DataFrame(weights_records)
    # restrict to trading_symbols
    daily_weights_long = daily_weights_long[
        daily_weights_long["symbol"].isin(trading_symbols)
    ]

    # ------------------------
    # Build minute-level price series from windows.live_history
    # ------------------------
    price_df = _extract_price_series_from_windows(
        month_windows=month_windows,
        trading_symbols=trading_symbols,
        first_day=first_day,
        next_month=next_month,
    )

    if price_df.empty:
        logger.warning(
            "No price data extracted from windows.live_history for %04d-%02d",
            year,
            month,
        )
        metrics = {
            "winning_percentage_pct": 0.0,
            "consistency_score_periods": 0,
            "rolling_return_consistency_pct": 0.0,
            "n_trading_days_in_month": 0,
            "first_trading_day": None,
            "last_trading_day": None,
        }

        month_id = f"{year:04d}-{month:02d}"
        monthly_col.update_one(
            {"month_id": month_id},
            {
                "$set": {
                    "month_id": month_id,
                    "year": year,
                    "month": month,
                    "last_snapshot_date": snapshot_date,
                    "last_metrics": metrics,
                    "portfolio_daily_returns": {},
                    "updated_at": datetime.utcnow(),
                },
                "$setOnInsert": {"created_at": datetime.utcnow()},
                "$push": {
                    "metrics_history": {
                        "snapshot_date": snapshot_date,
                        "metrics": metrics,
                        "created_at": datetime.utcnow(),
                    }
                },
            },
            upsert=True,
        )
        return metrics

    # Aggregate minute-level prices -> daily open/close per symbol
    daily_returns_long = _daily_open_close_from_minute_prices(price_df)

    # Filter returns to month range (safety)
    daily_returns_long = daily_returns_long[
        (daily_returns_long["date"] >= first_day.date())
        & (daily_returns_long["date"] <= last_day.date())
    ]

    portfolio_daily_returns = compute_portfolio_daily_returns(
        daily_returns_long, daily_weights_long
    )

    if portfolio_daily_returns.empty:
        metrics = {
            "winning_percentage_pct": 0.0,
            "consistency_score_periods": 0,
            "rolling_return_consistency_pct": 0.0,
            "n_trading_days_in_month": 0,
            "first_trading_day": None,
            "last_trading_day": None,
        }

        month_id = f"{year:04d}-{month:02d}"
        monthly_col.update_one(
            {"month_id": month_id},
            {
                "$set": {
                    "month_id": month_id,
                    "year": year,
                    "month": month,
                    "last_snapshot_date": snapshot_date,
                    "last_metrics": metrics,
                    "portfolio_daily_returns": {},
                    "updated_at": datetime.utcnow(),
                },
                "$setOnInsert": {"created_at": datetime.utcnow()},
                "$push": {
                    "metrics_history": {
                        "snapshot_date": snapshot_date,
                        "metrics": metrics,
                        "created_at": datetime.utcnow(),
                    }
                },
            },
            upsert=True,
        )
        return metrics

    # ------------------------
    # Compute metrics from daily returns
    # ------------------------
    winning_percentage_pct = compute_winning_percentage(portfolio_daily_returns)
    consistency_score_periods = compute_longest_positive_streak(portfolio_daily_returns)
    rolling_return_consistency_pct = compute_rolling_return_consistency(
        portfolio_daily_returns, window=ROLLING_WINDOW_DAYS
    )

    n_trading_days_in_month = int(len(portfolio_daily_returns))
    first_trading_day = portfolio_daily_returns.index.min()
    last_trading_day = portfolio_daily_returns.index.max()

    metrics = {
        "winning_percentage_pct": float(winning_percentage_pct),
        "consistency_score_periods": int(consistency_score_periods),
        "rolling_return_consistency_pct": float(rolling_return_consistency_pct),
        "n_trading_days_in_month": n_trading_days_in_month,
        "first_trading_day": first_trading_day.isoformat(),
        "last_trading_day": last_trading_day.isoformat(),
    }

        # ------------------------
    # Upsert monthly doc with history
    # ------------------------
    month_id = f"{year:04d}-{month:02d}"

    # convert index to ISO strings for Mongo (string keys only)
    pdr_dict = {d.isoformat(): float(v) for d, v in portfolio_daily_returns.items()}

    monthly_col.update_one(
        {"month_id": month_id},
        {
            "$set": {
                "month_id": month_id,
                "year": year,
                "month": month,
                "last_snapshot_date": snapshot_date,
                "last_metrics": metrics,
                "portfolio_daily_returns": pdr_dict,
                "updated_at": datetime.utcnow(),
            },
            "$setOnInsert": {"created_at": datetime.utcnow()},
            "$push": {
                "metrics_history": {
                    "snapshot_date": snapshot_date,
                    "metrics": metrics,
                    "created_at": datetime.utcnow(),
                }
            },
        },
        upsert=True,
    )
    

    logger.info(
        "Stored monthly metrics snapshot for %s (as_of=%s)",
        month_id,
        snapshot_date,
    )
    return metrics

def persist_window_snapshots_bulk(
    db,
    window_id: str,
    snapshots: List[Dict],
    stale_symbols_per_snapshot: List[List[str]],
):
    """
    Bulk version of persist_window_snapshot, optimized for backtesting.

    - Takes a list of snapshot dicts (as returned by compute_metrics_snapshot).
    - Takes a parallel list of stale symbol lists (same length).
    - Appends all snapshots to windows.live_history with a single $push + $each.
    - Inserts all snapshots into live_metrics with a single insert_many.

    This is intended for offline / backtesting workloads where you already
    have all snapshots computed in memory and want to minimize Mongo round-trips.
    """
    if not snapshots:
        return

    if len(snapshots) != len(stale_symbols_per_snapshot):
        raise ValueError("snapshots and stale_symbols_per_snapshot must have same length")

    windows_col = db["windows"]
    live_col = db["live_metrics"]

    now = datetime.utcnow()
    snapshot_docs = []
    for snap, stale_syms in zip(snapshots, stale_symbols_per_snapshot):
        doc = {
            **snap,
            "stale_symbols": stale_syms,
            "stale_count": len(stale_syms),
            "created_at": now,
        }
        snapshot_docs.append(doc)

    # Append all snapshots to the window's live_history in one operation
    windows_col.update_one(
        {"window_id": window_id},
        {
            "$push": {"live_history": {"$each": snapshot_docs}},
            "$set": {
                # last snapshot's stale_count as aggregate
                "stale_count": snapshot_docs[-1]["stale_count"],
                "updated_at": now,
            },
        },
    )

    # Insert all snapshots into live_metrics in one operation
    live_col.insert_many(snapshot_docs)