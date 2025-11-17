"""
compute_trading_days_service.py

Description:
    Computes and caches common trading days across all symbols in the AFN
    configuration. This is used to determine which days are valid for
    scheduling tasks such as data ingestion, predictions, and live metric
    calculations, ensuring that workflows only run on days when trading occurs.

Features:
    - Reads symbols from `config/AFN_config.yaml`.
    - Maps symbol prefixes (e.g., NASDAQ, NYSE, BINANCE) to their exchanges.
    - Supports multiple calendar sources:
        1. exchange_calendars (preferred)
        2. pandas_market_calendars (fallback)
        3. Last-resort: Mon-Fri assumption
    - Handles crypto and FX symbols as "always open" or Mon-Fri respectively.
    - Computes the intersection of trading days across all exchanges (common days).
    - Caches the results in `data/trading_days_cache.json` for later use.

Functions:
    - load_symbols(): Load all symbols from the AFN configuration.
    - extract_prefix(symbol): Extract the exchange prefix from a symbol string.
    - get_sessions_for_exchange(prefix, start_date, end_date): Returns all open
      dates for a given exchange prefix in the specified date range.
    - compute_common_trading_days(months_ahead=13): Compute and save all common
      trading days up to `months_ahead` months in the future.

Usage:
    python scripts/compute_trading_days_service.py
    # This will compute and cache the trading days for all symbols.

Notes:
    - Crypto symbols are treated as always open.
    - FX symbols are assumed open Mon-Fri unless specific holidays are added.
    - The cache file is read by `lib/trading_calendar_utils.py` to enforce
      trading-day restrictions in Celery tasks.
-------
Author: Elham Esmaeilnia(elham.e.shirvani@gmail.com)
Date: 2025 Nov 17
Version: 1.0.0 
"""
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
import yaml
import json
from pathlib import Path

# Prefer exchange_calendars
try:
    import exchange_calendars as xc
except Exception as e:
    xc = None
    print("exchange_calendars not available:", e)

# Fallback: pandas-market-calendars
try:
    import pandas_market_calendars as mcal
except Exception as e:
    mcal = None
    print("pandas-market-calendars not available:", e)

CONFIG_PATH = Path("config/AFN_config.yaml")
OUT_PATH = Path("data/trading_days_cache.json")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Map prefix in config to exchange_calendars ID or special tag
EXCHANGE_MAP = {
    # Crypto (always open)
    "BINANCE": "CRYPTO",  # BTC, ETH, BNB, SOL, XRP
    # US equities
    "NASDAQ": "XNAS",     # AAPL, MSFT, NVDA, GOOGL, META, AMZN, COST, AVGO, ASML, QCOM, PEP
    "NYSE": "XNYS",       # CRM, NOW, ORCL, TSM, IBM, JPM, BAC, WFC, MS, BRK.A, XOM, CVX, KO, MCD
    # Indices
    "SP": "CBOE",         # SPX
    "TVC": None,          # TradingView tickers (IXIC, NI225, UKX) – Mon-Fri fallback
    "XETR": "XETR",       # DAX
    # FX
    "FX": None,           # EURUSD – Mon-Fri
    "FX_IDC": None,       # USDJPY – Mon-Fri
    "OANDA": "CRYPTO",    # XAUUSD – treat as always open
    "BIST": None,         # XAGUSD1! – Mon-Fri fallback (Istanbul)
    "CBOE": "CBOE",       # VIX
}

# Symbols/exchanges considered "always open" (crypto, OANDA)
ALWAYS_OPEN_PREFIXES = {"BINANCE", "CRYPTO", "OANDA"}

def load_symbols():
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    syms = cfg.get("market", {}).get("symbols_usdt", []) or []
    return syms

def extract_prefix(symbol):
    # symbol looks like "NASDAQ:AAPL" or "BINANCE:BTCUSDT" or "FX:EURUSD"
    if ":" in symbol:
        return symbol.split(":")[0].upper()
    return None

def get_sessions_for_exchange(prefix, start_date, end_date):
    """
    Returns a set of dates (date objects) when the exchange is open between start_date and end_date inclusive.
    For always-open exchanges returns all dates in range (including weekends for crypto).
    For FX, we assume Monday-Friday (no exchange-level holidays). If you have a specific FX holiday list, supply it.
    """
    prefix = prefix.upper()
    if prefix in ALWAYS_OPEN_PREFIXES:
        # crypto: open every day
        days = set()
        cur = start_date
        while cur <= end_date:
            days.add(cur)
            cur += timedelta(days=1)
        return days

    if prefix in ("FX", "FX_IDC", "OANDA"):
        # treat as Mon-Fri open
        days = set()
        cur = start_date
        while cur <= end_date:
            if cur.weekday() < 5:
                days.add(cur)
            cur += timedelta(days=1)
        return days

    cal_id = EXCHANGE_MAP.get(prefix)
    # if we have a mapping and exchange_calendars available:
    if cal_id and xc:
        try:
            cal = xc.get_calendar(cal_id)
            # sessions_in_range returns Timestamp index in UTC-localized session start times
            sessions = cal.sessions_in_range(start_date.isoformat(), end_date.isoformat())
            # convert to dates (session index are Timestamps)
            return set(s.date() for s in sessions)
        except Exception as e:
            print(f"exchange_calendars error for {prefix}/{cal_id}: {e}")

    # fallback to pandas-market-calendars if available and cal_id exists as a known short name
    if mcal and cal_id:
        try:
            pmcal = mcal.get_calendar(cal_id)  # note: mapping name differences may exist
            schedule = pmcal.schedule(start_date=start_date.isoformat(), end_date=end_date.isoformat())
            return set(d.date() for d in schedule.index)
        except Exception as e:
            print(f"pandas-market-calendars error for {prefix}/{cal_id}: {e}")

    # Last-resort fallback: assume Mon-Fri
    print(f"Warning: no calendar available for prefix {prefix}. Falling back to Mon-Fri.")
    days = set()
    cur = start_date
    while cur <= end_date:
        if cur.weekday() < 5:
            days.add(cur)
        cur += timedelta(days=1)
    return days


def compute_common_trading_days(months_ahead=13):
    now = date.today()
    end = (now + relativedelta(months=months_ahead)).replace(day=31)  # ensure full-month coverage
    symbols = load_symbols()

    # collect calendars for each distinct prefix
    prefixes = {extract_prefix(s) for s in symbols if extract_prefix(s)}
    print("Found prefixes:", prefixes)

    per_exchange_days = {}
    for p in prefixes:
        days = get_sessions_for_exchange(p, now, end)
        per_exchange_days[p] = days
        print(f"{p}: {len(days)} sessions from {now} to {end}")

    # intersect across all prefixes
    # for fairness, if some prefix is always-open it won't restrict others
    all_sets = list(per_exchange_days.values())
    if not all_sets:
        common = set()
    else:
        common = set.intersection(*all_sets)

    # sort and output list of iso dates
    common_list = sorted(common)
    out = [d.isoformat() for d in common_list]

    # Save cache
    OUT_PATH.write_text(json.dumps({
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "start": now.isoformat(),
        "end": end.isoformat(),
        "common_trading_days": out,
        "per_exchange_counts": {p: len(v) for p, v in per_exchange_days.items()}
    }, indent=2))

    print(f"Wrote {len(out)} common trading days to {OUT_PATH}")
    return out


if __name__ == "__main__":
    compute_common_trading_days()
