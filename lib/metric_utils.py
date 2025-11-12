from __future__ import annotations
import logging
import math
from typing import Optional, Tuple, Dict

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# -------------------------------
# ClickHouse data fetcher
# -------------------------------

def fetch_ohlcv_range(ch_client,
                       table: str,
                       symbol: str,
                       start_utc: datetime,
                       end_utc: datetime,
                       time_col: str = "candle_time",
                       cols_required: Optional[list] = None) -> pd.DataFrame:
    """
    Fetch rows for a symbol in [start_utc, end_utc] from a ClickHouse table and return a DataFrame.

    Parameters
    - ch_client: a clickhouse_driver.Client instance
    - table: fully qualified table name
    - symbol: symbol to query (string)
    - start_utc, end_utc: datetimes (assumed timezone-aware UTC or naive UTC)
    - time_col: column name that stores the timestamp in ClickHouse
    - cols_required: list of columns to select. If None, select * and then infer.

    Returns: pd.DataFrame with at least columns ['symbol', time_col, ...]
    - datetime column will be named 'dateTime' (pandas.Timestamp, tz-aware UTC)
    """
    if start_utc.tzinfo is None:
        start_utc = start_utc.replace(tzinfo=None)
    if end_utc.tzinfo is None:
        end_utc = end_utc.replace(tzinfo=None)

    start_str = start_utc.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    end_str = end_utc.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    select_cols = ", ".join(cols_required) if cols_required else "*"
    q = f"""
        SELECT {select_cols}
        FROM {table}
        WHERE symbol = '{symbol}'
          AND {time_col} >= '{start_str}'
          AND {time_col} <= '{end_str}'
        ORDER BY {time_col} ASC
    """
    data = ch_client.execute(q)
    if not data:
        return pd.DataFrame()

    # infer column names
    if cols_required:
        cols = cols_required
    else:
        desc = ch_client.execute(f"DESCRIBE TABLE {table}")
        cols = [c[0] for c in desc]

    df = pd.DataFrame(data, columns=cols)

    # normalize timestamp column to pandas datetime UTC
    if time_col in df.columns:
        df['dateTime'] = pd.to_datetime(df[time_col], utc=True)
    else:
        # fallback: try to find a time-like column
        for cand in ['timestamp', 'ts', 'time', 'date']:
            if cand in df.columns:
                df['dateTime'] = pd.to_datetime(df[cand], utc=True)
                break

    return df


# -------------------------------
# Minute -> Daily aggregation
# -------------------------------

def compute_daily_returns(minute_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate minute-level prices to daily open, close and daily return per symbol.

    Input columns expected: ['datetime' or 'dateTime', 'symbol', 'price']

    Returns DataFrame with index 'date' (YYYY-MM-DD string) and columns: [symbol, open, close, return]
    Actually returns a tidy DataFrame with columns: [date, symbol, open, close, return]
    """
    if minute_prices.empty:
        return pd.DataFrame(columns=['date', 'symbol', 'open', 'close', 'return'])

    df = minute_prices.copy()
    # normalize column names
    if 'dateTime' in df.columns:
        df['datetime'] = pd.to_datetime(df['dateTime'], utc=True)
    elif 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    else:
        raise ValueError("minute_prices must have a datetime/dateTime column")

    if 'symbol' not in df.columns or 'price' not in df.columns:
        raise ValueError("minute_prices must contain 'symbol' and 'price' columns")

    df['date'] = df['datetime'].dt.strftime('%Y-%m-%d')

    # compute first and last price per day/symbol
    grouped = df.sort_values(['symbol', 'datetime']).groupby(['symbol', 'date'])

    open_price = grouped.first().reset_index()[['symbol', 'date', 'price']].rename(columns={'price': 'open'})
    close_price = grouped.last().reset_index()[['symbol', 'date', 'price']].rename(columns={'price': 'close'})

    daily = pd.merge(open_price, close_price, on=['symbol', 'date'], how='outer')
    # drop rows where either open or close is missing
    daily = daily.dropna(subset=['open', 'close']).copy()
    daily['return'] = daily['close'] / daily['open'] - 1.0

    # ensure ordering
    daily = daily.sort_values(['date', 'symbol']).reset_index(drop=True)
    return daily


# -------------------------------
# Build portfolio from daily_returns and weights
# -------------------------------

def build_portfolio_returns(daily_returns: pd.DataFrame,
                            weights: pd.DataFrame,
                            year_month: Optional[Tuple[int, int]] = None,
                            spx_symbol: str = '^GSPC') -> Tuple[pd.Series, Optional[pd.Series], Dict]:
    """
    daily_returns: tidy df with columns ['date', 'symbol', 'return'] (as produced by compute_daily_returns)
    weights: tidy df with columns ['date', 'symbol', 'weight']
    year_month: (year, month) to filter. If None, uses latest complete month present in daily_returns

    Returns: (portfolio_returns_series, spx_returns_series_or_none, metadata dict)
    portfolio_returns_series: pd.Series indexed by date (YYYY-MM-DD) of portfolio daily returns
    spx_returns_series: pd.Series aligned with portfolio (if benchmark present)
    metadata: includes n_trading_days_in_month, first_trading_day, last_trading_day
    """
    if daily_returns.empty:
        raise ValueError("daily_returns is empty")

    # pivot returns to matrix R[date, symbol]
    R = daily_returns.pivot(index='date', columns='symbol', values='return')

    # If month not given, pick last full calendar month in the data
    if year_month is None:
        # find last date in R, then take previous full month
        last_date = pd.to_datetime(R.index).max()
        # pick last full month: if last_date is at month end, use that month; otherwise use previous month
        last_day_of_month = last_date + pd.offsets.MonthEnd(0)
        if last_date >= last_day_of_month:
            year = last_date.year
            month = last_date.month
        else:
            prev = last_date - pd.offsets.MonthBegin(1)
            year = prev.year
            month = prev.month
    else:
        year, month = year_month

    # filter dates within target month
    all_dates = pd.to_datetime(R.index)
    mask = (all_dates.year == year) & (all_dates.month == month)
    Rm = R.loc[mask]
    if Rm.empty:
        raise ValueError(f"No price data found for target month {year}-{month:02d}")

    dates = Rm.index.astype(str)

    # Prepare weights matrix W[date, symbol]
    W = weights.copy()
    W['date'] = pd.to_datetime(W['date']).dt.strftime('%Y-%m-%d')
    Wm = W[W['date'].isin(dates)]

    if Wm.empty:
        # If no weights for the month, treat as all cash
        logger.warning("No weights available for month; returning zero returns (cash)")
        portfolio_returns = pd.Series(0.0, index=dates)
        metadata = {
            'n_trading_days_in_month': len(dates),
            'first_trading_day': dates[0],
            'last_trading_day': dates[-1]
        }
        return portfolio_returns, None, metadata

    # build pivot W matrix and align columns with Rm
    Wmat = Wm.pivot(index='date', columns='symbol', values='weight').reindex(index=dates).fillna(0.0)

    # normalize weights per day (tolerance check)
    row_sums = Wmat.sum(axis=1)
    # handle zero rows (all cash): keep zeros
    nonzero_mask = row_sums > 0
    Wmat.loc[nonzero_mask] = Wmat.loc[nonzero_mask].div(row_sums.loc[nonzero_mask], axis=0)

    # align Rm columns with Wmat columns
    # ensure both matrices have same set of symbols; missing returns treated as NaN and ignored in dot product
    symbols = sorted(list(set(Rm.columns).union(set(Wmat.columns))))
    Rm = Rm.reindex(columns=symbols)
    Wmat = Wmat.reindex(columns=symbols).fillna(0.0)

    # Compute portfolio returns per day: sum(W * R) where R NaN treated as 0 (symbol excluded)
    # But we should exclude symbols that lack open/close (those rows are NaN in Rm)
    Rm_filled = Rm.fillna(0.0)
    portfolio_values = (Wmat * Rm_filled).sum(axis=1)
    portfolio_returns = pd.Series(portfolio_values.values, index=dates)

    # Benchmark returns if present
    spx_returns = None
    if spx_symbol in Rm.columns:
        spx_returns = Rm[spx_symbol].reindex(index=dates)

    metadata = {
        'n_trading_days_in_month': len(dates),
        'first_trading_day': dates[0],
        'last_trading_day': dates[-1]
    }

    return portfolio_returns, spx_returns, metadata


# -------------------------------
# Compute metrics for dashboard
# -------------------------------

def compute_dashboard_metrics(portfolio_returns: pd.Series,
                              rolling_window_days: int = 5,
                              clustering_scaling: int = 10) -> Dict:
    """
    Given a series of daily portfolio returns indexed by date (strings YYYY-MM-DD), compute metrics described in the brief.

    Returns a dict with keys:
      - winning_percentage_pct
n     - consistency_score_periods
      - rolling_return_consistency_pct
      - return_vol_clustering_score_0_10
      - plus metadata: n_trading_days_in_month, first_trading_day, last_trading_day
    """
    r = portfolio_returns.dropna()
    N = len(r)
    out = {
        'winning_percentage_pct': float('nan'),
        'consistency_score_periods': None,
        'rolling_return_consistency_pct': float('nan'),
        'return_vol_clustering_score_0_10': float('nan'),
        'n_trading_days_in_month': N
    }

    if N < 5:
        logger.warning("Fewer than 5 trading days in month; metrics will be NaN where applicable")
        out.update({
            'first_trading_day': r.index[0] if N > 0 else None,
            'last_trading_day': r.index[-1] if N > 0 else None
        })
        return out

    wins_bool = r > 0
    wins = wins_bool.sum()
    out['winning_percentage_pct'] = 100.0 * float(wins) / float(N)

    # longest consecutive True run
    streak = 0
    max_streak = 0
    for val in wins_bool:
        if val:
            streak += 1
            if streak > max_streak:
                max_streak = streak
        else:
            streak = 0
    out['consistency_score_periods'] = int(max_streak)

    # rolling mean 5-day
    rolling_mean = r.rolling(window=rolling_window_days, min_periods=rolling_window_days).mean()
    rolling_windows = rolling_mean.dropna()
    if len(rolling_windows) == 0:
        out['rolling_return_consistency_pct'] = float('nan')
    else:
        out['rolling_return_consistency_pct'] = 100.0 * float((rolling_windows > 0).sum()) / float(len(rolling_windows))

    # volatility clustering using autocorr of abs returns lag-1
    vol_proxy = r.abs()
    vol_proxy_mean = vol_proxy.mean()
    if vol_proxy_mean == 0 or len(vol_proxy) < 2:
        rho = 0.0
    else:
        # compute lag-1 autocorrelation
        v = vol_proxy - vol_proxy_mean
        rho = (v[:-1].values * v[1:].values).sum() / ( (v[:-1].values**2).sum() )
        # safeguard numerical issues
        if not np.isfinite(rho):
            rho = 0.0
    if rho <= 0:
        score = 0.0
    else:
        score = min(clustering_scaling, clustering_scaling * rho)
    out['return_vol_clustering_score_0_10'] = float(score)

    out['first_trading_day'] = r.index[0]
    out['last_trading_day'] = r.index[-1]

    return out


# -------------------------------
# Persistence helpers
# -------------------------------

def save_metrics_snapshot(db, collection_name: str, window_id: str, metrics: dict, portfolio_series: Optional[pd.Series] = None):
    """
    Save a snapshot to MongoDB collection. The dashboard team reads these documents.

    Document shape:
    {
        'window_id': str,
        'timestamp': datetime.utcnow(),
        'metrics': {...},
        'portfolio_series': {date: value, ...}   # optional
    }
    """
    coll = db[collection_name]
    doc = {
        'window_id': window_id,
        'timestamp': datetime.utcnow(),
        'metrics': metrics
    }
    if portfolio_series is not None:
        # convert to simple dict
        doc['portfolio_series'] = portfolio_series.to_dict()
    coll.insert_one(doc)
    logger.info(f"Inserted metrics snapshot for window {window_id} into {collection_name}")


# -------------------------------
# High level: month compute pipeline
# -------------------------------

def compute_month_metrics_and_persist(db,
                                      minute_prices: pd.DataFrame,
                                      weights: pd.DataFrame,
                                      mongo_collection: str = 'live_metrics_monthly',
                                      year_month: Optional[Tuple[int, int]] = None,
                                      spx_symbol: str = '^GSPC') -> dict:
    """
    Convenience function that runs the pipeline: compute daily returns, build portfolio returns for the month,
    compute metrics and persist result into MongoDB collection.

    Returns metrics dict (also persisted).
    """
    daily = compute_daily_returns(minute_prices)
    portfolio_series, spx_series, metadata = build_portfolio_returns(daily, weights, year_month, spx_symbol=spx_symbol)
    metrics = compute_dashboard_metrics(portfolio_series)
    # merge metadata into metrics
    metrics.update(metadata)

    # optional supporting series
    # save portfolio daily series as part of document
    save_metrics_snapshot(db, mongo_collection, window_id=f"monthly_{metadata['first_trading_day']}_{metadata['last_trading_day']}", metrics=metrics, portfolio_series=portfolio_series)

    return metrics