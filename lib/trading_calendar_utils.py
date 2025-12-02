"""
trading_calendar_utils.py

Description:
    Utility functions to manage and query common trading days for multiple assets
    in your trading universe. This module reads a cached JSON file containing
    precomputed trading days and provides helpers to determine whether today is
    a trading day and to get the next available trading day.

All dates are UTC-based:
    - The cache is generated using datetime.utcnow().date().
    - "Today" in this module means the current UTC date, not the local system
      timezone date.

Usage Examples:
    from lib.trading_calendar_utils import is_today_common_trading_day, next_common_trading_day

    if is_today_common_trading_day():
        print("We can run scheduled trading tasks today (UTC-based).")

    next_day = next_common_trading_day()
    print(f"The next trading day (UTC) is: {next_day}")

Notes:
    - The cache file should be generated and updated by a separate script
      (e.g., `compute_trading_days_service.py`) to reflect holidays and weekends.
    - All dates are handled as `datetime.date` objects.
    - If the cache is empty or missing, functions return an empty set or None.

Author: Elham Esmaeilnia (elham.e.shirvani@gmail.com)
Date: 2025 Nov 17
Version: 1.0.1 (UTC date handling)
"""
from pathlib import Path
import json
from datetime import date, datetime, time, timedelta, timezone

CACHE = Path("data/trading_days_cache.json")


def utc_today() -> date:
    """
    Returns today's date in UTC.

    Important:
        Do NOT use date.today() (which depends on local system timezone)
        in a trading system where everything is defined in UTC.
    """
    return datetime.now(timezone.utc).date()


def load_common_days():
    """
    Load the set of all common trading days from the cache.

    Returns:
        Set[date]: set of UTC dates that are common trading days
                   (one ISO string per date in the cache).
    """
    if not CACHE.exists():
        return set()
    j = json.loads(CACHE.read_text())
    days = j.get("common_trading_days", [])
    # These are saved as "YYYY-MM-DD", so parse as pure date objects.
    return set(date.fromisoformat(d) for d in days)


def is_today_common_trading_day() -> bool:
    """
    Returns True if the current UTC date is a common trading day.
    """
    today_utc = utc_today()
    return today_utc in load_common_days()


def next_common_trading_day(after_date: date = None):
    """
    Returns the next common trading day strictly after `after_date`.

    Args:
        after_date (date, optional): Base date (UTC). If None, uses today's
            UTC date.

    Returns:
        date or None: Next common trading day after `after_date`, or None
                      if there is no future day in the cache.
    """
    s = sorted(load_common_days())
    if not s:
        return None

    if after_date is None:
        after_date = utc_today()

    for d in s:
        if d > after_date:
            return d
    return None

def previous_common_trading_day(before_date: date = None):
    """
    Returns the previous common trading day strictly before `before_date`.

    Args:
        before_date (date, optional):
            The reference date (UTC). If None, uses today's UTC date.

    Returns:
        date or None:
            The most recent trading day before `before_date`.
            Returns None if no earlier day is available.
    """
    s = sorted(load_common_days())
    if not s:
        return None

    if before_date is None:
        before_date = utc_today()

    prev = None
    for d in s:
        if d < before_date:
            prev = d
        else:
            break
    return prev
