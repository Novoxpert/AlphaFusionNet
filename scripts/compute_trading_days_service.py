"""
compute_trading_days_service.py

Description:
    Computes and caches common trading days across all symbols in the AFN
    configuration. This is used to determine which days are valid for
    scheduling tasks such as data ingestion, predictions, metric computation,
    and backtesting, ensuring that workflows only run on days when trading occurs.

Features:
    - Reads symbols from `config/AFN_config.yaml`.
    - Maps symbol prefixes (e.g., NASDAQ, NYSE, BINANCE, IG, OANDA) to their exchanges.
    - Supports multiple calendar sources:
        1. exchange_calendars (preferred)
        2. pandas_market_calendars (fallback)
        3. Last-resort: Mon–Fri assumption
    - Handles crypto symbols as "always open" (24/7).
    - FX / metals via IG / OANDA treated as 24/5 (Mon–Fri, no holidays).
    - Computes the intersection of trading days across all exchanges (common days).
    - Forward mode:
        * Computes from today (UTC) into the future and caches to
          `data/trading_days_cache.json`.
    - Backward / range mode:
        * Computes over arbitrary [start_date, end_date] ranges for backtesting.

Notes:
    - This script works purely with dates; all logic is UTC-based.
    - This script ensures only that all exchanges are open on those days.
      Having OHLCV at 14:00 UTC for every symbol must be validated with
      data/metric services.

Author: Elham Esmaeilnia (elham.e.shirvani@gmail.com)
Date: 2025 Nov 17
Version: 1.1.0 (added backward / range helpers)
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

# ---------------------------------------------------------------------------
# Exchange mapping
# ---------------------------------------------------------------------------
# Map symbol prefix in config to exchange_calendars ID or special handling tag.
#
# Prefixes present in AFN_config.yaml:
#   BINANCE, NASDAQ, NYSE, SP, TVC, XETR, IG, OANDA, CBOE
#
# - BINANCE: crypto (24/7) — always open.
# - NASDAQ: US equities → XNAS calendar.
# - NYSE  : US equities → XNYS calendar.
# - SP    : SPX index (S&P 500). True CBOE calendar not available in
#           exchange_calendars, so we approximate with NYSE (XNYS) which
#           matches US equity holidays.
# - CBOE  : VIX. Same case as SPX → fallback to NYSE (XNYS).
# - TVC   : TradingView indices (IXIC, UKX) → no exchange calendar available,
#           fallback to Mon–Fri.
# - XETR  : German XETRA (DAX) → XETR calendar.
# - IG    : FX pairs → treat as 24/5 (Mon–Fri).
# - OANDA : FX metals → treat as 24/5 (Mon–Fri).
#
EXCHANGE_MAP = {
    # Crypto (always open)
    "BINANCE": "CRYPTO",

    # US equities
    "NASDAQ": "XNAS",
    "NYSE": "XNYS",

    # Indices
    # SPX & VIX use NYSE calendar (closest holiday schedule)
    "SP": "XNYS",        # SP:SPX
    "CBOE": "XNYS",      # CBOE:VIX

    "TVC": None,         # Mon–Fri fallback
    "XETR": "XETR",      # German DAX

    # FX / metals — treat as 24/5
    "IG": None,          # USDCAD, EURUSD, USDJPY
    "OANDA": None,       # XAUUSD, XAGUSD
}

# Symbols/exchanges considered "always open" (crypto, 24/7).
# IMPORTANT: OANDA is *not* always open; it's 24/5, so do NOT include it here.
ALWAYS_OPEN_PREFIXES = {"BINANCE", "CRYPTO"}


def load_symbols():
    """Load the symbol universe from AFN_config.yaml."""
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    syms = cfg.get("market", {}).get("symbols_usdt", []) or []
    return syms


def extract_prefix(symbol: str):
    """
    symbol looks like "NASDAQ:AAPL" or "BINANCE:BTCUSDT" or "IG:EURUSD".
    Returns the uppercase prefix or None if no prefix is found.
    """
    if ":" in symbol:
        return symbol.split(":", 1)[0].upper()
    return None


def get_sessions_for_exchange(prefix: str, start_date: date, end_date: date):
    """
    Returns a set of dates (date objects) when the exchange is open between
    start_date and end_date inclusive.

    - For ALWAYS_OPEN_PREFIXES (e.g., BINANCE crypto): returns all dates
      in the range, including weekends (24/7).
    - For IG / OANDA / generic FX prefixes: we assume 24/5 (Mon–Fri) with
      no exchange-level holidays (simplified model).
    - For mapped exchanges with exchange_calendars: use that calendar.
    - Otherwise: Mon–Fri fallback.
    """
    prefix = prefix.upper()

    # 24/7 always-open markets (crypto)
    if prefix in ALWAYS_OPEN_PREFIXES:
        days = set()
        cur = start_date
        while cur <= end_date:
            days.add(cur)
            cur += timedelta(days=1)
        return days

    # FX / metals (IG, OANDA, plus generic FX prefixes if they ever appear)
    if prefix in ("IG", "OANDA", "FX", "FX_IDC"):
        days = set()
        cur = start_date
        while cur <= end_date:
            # 24/5: open Monday–Friday, closed on weekends
            if cur.weekday() < 5:
                days.add(cur)
            cur += timedelta(days=1)
        return days

    cal_id = EXCHANGE_MAP.get(prefix)

    # If we have a mapping and exchange_calendars available:
    if cal_id and xc is not None:
        try:
            cal = xc.get_calendar(cal_id)
            # sessions_in_range takes strings "YYYY-MM-DD"
            sessions = cal.sessions_in_range(start_date.isoformat(), end_date.isoformat())
            # convert to date objects
            return set(s.date() for s in sessions)
        except Exception as e:
            print(f"exchange_calendars error for {prefix}/{cal_id}: {e}")

    # Fallback to pandas-market-calendars if available and cal_id exists
    if mcal is not None and cal_id:
        try:
            pmcal = mcal.get_calendar(cal_id)
            schedule = pmcal.schedule(start_date=start_date.isoformat(), end_date=end_date.isoformat())
            return set(idx.date() for idx in schedule.index)
        except Exception as e:
            print(f"pandas-market-calendars error for {prefix}/{cal_id}: {e}")

    # Last-resort fallback: assume Mon–Fri.
    print(f"Warning: no specific calendar available for prefix {prefix}. Falling back to Mon–Fri.")
    days = set()
    cur = start_date
    while cur <= end_date:
        if cur.weekday() < 5:
            days.add(cur)
        cur += timedelta(days=1)
    return days


# ---------------------------------------------------------------------------
# CORE RANGE FUNCTION
# ---------------------------------------------------------------------------
def compute_common_trading_days_range(
    start_date: date,
    end_date: date,
    return_counts: bool = False,
):
    """
    Core function: compute common trading days between start_date and end_date
    (inclusive), using the AFN_config.yaml symbol universe.

    Args:
        start_date (date): UTC start date (inclusive).
        end_date (date): UTC end date (inclusive).
        return_counts (bool): If True, also return per-exchange session counts.

    Returns:
        If return_counts is False:
            List[date]: sorted list of common trading days.
        If return_counts is True:
            (List[date], Dict[str, Set[date]]): common days + per-exchange day sets.
    """
    symbols = load_symbols()

    # Collect unique prefixes from your universe
    prefixes = {extract_prefix(s) for s in symbols if extract_prefix(s)}
    print(f"Found prefixes: {prefixes}")

    per_exchange_days = {}
    for p in prefixes:
        days = get_sessions_for_exchange(p, start_date, end_date)
        per_exchange_days[p] = days
        print(f"{p}: {len(days)} sessions from {start_date} to {end_date}")

    all_sets = list(per_exchange_days.values())
    if not all_sets:
        common = set()
    else:
        common = set.intersection(*all_sets)

    common_list = sorted(common)

    if return_counts:
        return common_list, per_exchange_days
    return common_list


# ---------------------------------------------------------------------------
# FORWARD-LOOKING (PRODUCTION CACHE) FUNCTION
# ---------------------------------------------------------------------------
def compute_common_trading_days(months_ahead: int = 13):
    """
    Compute the intersection of trading days for all prefixes found in the
    AFN_config.yaml symbol list, from today (UTC) up to months_ahead months
    in the future (end-of-month).

    This is the forward-looking function used for Celery / scheduling. It
    writes to `data/trading_days_cache.json`.

    Returns:
        List[str]: sorted ISO date strings (YYYY-MM-DD) of common trading days.
    """
    # Use UTC date, NOT local date.
    now = datetime.utcnow().date()

    # Correct way to get "months_ahead" months ahead, then snap to end-of-month.
    # relativedelta(day=31) means "last day of the month".
    end = now + relativedelta(months=months_ahead, day=31)

    common_list, per_exchange_days = compute_common_trading_days_range(
        start_date=now,
        end_date=end,
        return_counts=True,
    )

    # Sort and output list of ISO dates
    out = [d.isoformat() for d in common_list]

    # Save cache (timestamps also in UTC)
    OUT_PATH.write_text(
        json.dumps(
            {
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "start": now.isoformat(),
                "end": end.isoformat(),
                "common_trading_days": out,
                "per_exchange_counts": {p: len(v) for p, v in per_exchange_days.items()},
            },
            indent=2,
        )
    )

    print(f"Wrote {len(out)} common trading days to {OUT_PATH}")
    return out


# ---------------------------------------------------------------------------
# BACKWARD / ARBITRARY-RANGE HELPERS FOR BACKTESTING
# ---------------------------------------------------------------------------
def compute_common_trading_days_past(months_back: int = 13):
    """
    Backward-looking helper for backtesting.

    Computes common trading days from `months_back` months ago (start-of-month)
    until today (UTC).

    Args:
        months_back (int): Number of months to look backwards.

    Returns:
        List[date]: sorted list of common trading-day dates.
    """
    today_utc = datetime.utcnow().date()
    # First day of the month `months_back` months ago
    start = (today_utc - relativedelta(months=months_back)).replace(day=1)
    end = today_utc

    return compute_common_trading_days_range(start, end, return_counts=False)


def compute_common_trading_days_between(start_iso: str, end_iso: str):
    """
    Compute common trading days between two explicit ISO dates (UTC).

    Args:
        start_iso (str): "YYYY-MM-DD" start date (inclusive).
        end_iso   (str): "YYYY-MM-DD" end date (inclusive).

    Returns:
        List[date]: sorted list of common trading-day dates.
    """
    start = date.fromisoformat(start_iso)
    end = date.fromisoformat(end_iso)
    return compute_common_trading_days_range(start, end, return_counts=False)


if __name__ == "__main__":
    # Default behavior: forward-looking cache for production scheduling
    compute_common_trading_days()
