"""
trading_calendar_utils.py

Description:
    Utility functions to manage and query common trading days for multiple assets
    in your trading universe. This module reads a cached JSON file containing
    precomputed trading days and provides helpers to determine whether today is
    a trading day and to get the next available trading day.

File Structure:
    - CACHE: Path to the cached JSON file storing trading days.
    - load_common_days(): Load the set of all common trading days from the cache.
    - is_today_common_trading_day(): Returns True if today is a trading day.
    - next_common_trading_day(after_date=None): Returns the next trading day
      after the given date (defaults to today).

Usage Examples:
    from lib.trading_calendar_utils import is_today_common_trading_day, next_common_trading_day

    if is_today_common_trading_day():
        print("We can run scheduled trading tasks today.")
    
    next_day = next_common_trading_day()
    print(f"The next trading day is: {next_day}")

Notes:
    - The cache file should be generated and updated by a separate script
      (e.g., `compute_trading_days_service.py`) to reflect holidays and weekends.
    - All dates are handled as `datetime.date` objects.
    - If the cache is empty or missing, functions return an empty set or None.

-------
Author: Elham Esmaeilnia(elham.e.shirvani@gmail.com)
Date: 2025 Nov 17
Version: 1.0.0 
"""
from pathlib import Path
import json
from datetime import date, datetime, time, timedelta

CACHE = Path("data/trading_days_cache.json")

def load_common_days():
    if not CACHE.exists():
        return set()
    j = json.loads(CACHE.read_text())
    days = j.get("common_trading_days", [])
    return set(datetime.fromisoformat(d).date() for d in days)

def is_today_common_trading_day():
    return date.today() in load_common_days()

def next_common_trading_day(after_date=None):
    s = sorted(load_common_days())
    if not s:
        return None
    if after_date is None:
        after_date = date.today()
    for d in s:
        if d > after_date:
            return d
    return None
