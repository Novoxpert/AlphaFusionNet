#!/usr/bin/env python3
"""
model_backtesting.py
====================
Run a full AlphaFusionNet backtest pipeline for past months.

For each TRADING DAY in the chosen past month, this script:

  1. Finds the PREVIOUS trading day (walking back day by day).
  2. Builds a UTC window on that previous trading day:
         start = 14:30:00 UTC
         end   = 18:30:00 UTC
  3. Runs, sequentially:
         - ChronoBridge chronobridge_service (synchronize mode, custom window)
         - NeuralFusionCore prediction_service (synchronize mode)
         - NetWeaver netweaver_prediction_service
         - alphafusionnet_service

Trading-day logic:
  • Tries to import `is_common_trading_day(date)` from lib.trading_calendar_utils.
  • If not found, falls back to a simple weekday-based rule (Mon–Fri).

Usage examples
--------------
# Last month (relative to today)
python model_backtesting.py --month 1

# Two months ago
python model_backtesting.py --month 2

Arguments
---------
--month N  : how many months back to backtest
             1 = last month, 2 = two months ago, ...

Author: Elham Esmaeilnia (elham.e.shirvani@gmail.com)
Date: 2025 Nov 29
Version: 1.0.0
"""

import argparse
import logging
import subprocess
import sys
from datetime import datetime, date, time, timedelta, timezone
from pathlib import Path

# ---------------------------------------------------------------------
# Optional trading calendar integration
# ---------------------------------------------------------------------
try:
    # If your utils expose this, we use it
    from lib.trading_calendar_utils import is_common_trading_day  # type: ignore

    def is_trading_day(d: date) -> bool:
        """Wrapper around project trading-calendar util."""
        return bool(is_common_trading_day(d))

except Exception:
    # Fallback: simple weekday-based trading days (Mon–Fri)
    def is_trading_day(d: date) -> bool:
        """Fallback trading-day rule: Monday–Friday only."""
        return d.weekday() < 5


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent                    # .../AlphaFusionNet

def run_script(module: str, *args: str) -> None:
    """
    Run another Python module as a script in the same virtualenv.

    Parameters
    ----------
    module : str
        Dotted module path, e.g. "apps.ChronoBridge.scripts.chronobridge_service"
    *args : str
        Extra CLI arguments (already tokenized).
    """
    cmd = [sys.executable, "-m", module, *map(str, args)]
    logging.info("[RUN] %s", " ".join(cmd))

    # IMPORTANT: run from project root so 'apps' and 'scripts' are importable
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)

def run_script_safe(module: str, *args: str) -> None:
    """
    Run a Python module, BUT never interrupt the pipeline if it fails.

    - Logs the error
    - Continues execution
    """
    cmd = [sys.executable, "-m", module, *map(str, args)]
    logging.info("[RUN-SAFE] %s", " ".join(cmd))

    try:
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
    except subprocess.CalledProcessError as e:
        logging.error(
            "⚠️ NetWeaver step failed: %s (exit code %s). Continuing...",
            e, e.returncode
        )
    except Exception as e:
        logging.error("⚠️ Unexpected error in NetWeaver step: %s. Continuing...", e)

def month_from_offset(months_back: int, ref_date: date | None = None) -> tuple[int, int]:
    """
    Given a reference date (default: today UTC), return (year, month)
    for `months_back` months in the past.

    Example:
        ref_date = 2025-11-29, months_back=0 -> (2025, 11)  # current month
        ref_date = 2025-11-29, months_back=1 -> (2025, 10)
        ref_date = 2025-01-05, months_back=2 -> (2024, 11)
    """
    if ref_date is None:
        ref_date = datetime.now(timezone.utc).date()

    year = ref_date.year
    month = ref_date.month - months_back
    while month <= 0:
        month += 12
        year -= 1
    return year, month


def iter_trading_days_in_month(year: int, month: int):
    """Yield all trading days (per is_trading_day) in the given month."""
    first = date(year, month, 1)
    if month == 12:
        first_next = date(year + 1, 1, 1)
    else:
        first_next = date(year, month + 1, 1)

    d = first
    while d < first_next:
        if is_trading_day(d):
            yield d
        d += timedelta(days=1)


def previous_trading_day(d: date) -> date:
    """
    Find the previous trading day STRICTLY before date d.
    Walks back day by day until a trading day is found.
    """
    prev = d - timedelta(days=1)
    # Safety bound: don't walk back forever
    for _ in range(365):
        if is_trading_day(prev):
            return prev
        prev -= timedelta(days=1)
    raise RuntimeError(f"Could not find previous trading day within 1 year before {d}")


# ---------------------------------------------------------------------
# Main backtesting loop
# ---------------------------------------------------------------------
def backtest_last_days(n_days: int, mode: str = "synchronize") -> None:
    today = datetime.now(timezone.utc).date()

    # collect last n trading days
    days = []
    d = today
    while len(days) < n_days:
        d -= timedelta(days=1)
        if is_trading_day(d):
            days.append(d)

    days.reverse()  # oldest first

    logging.info("Backtesting last %d trading days: %s", n_days, days)

    for trade_day in days:
        prev_day = previous_trading_day(trade_day)

        start_dt = datetime(prev_day.year, prev_day.month, prev_day.day, 14, 30, tzinfo=timezone.utc)
        end_dt   = datetime(prev_day.year, prev_day.month, prev_day.day, 18, 30, tzinfo=timezone.utc)

        start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        end_str   = end_dt.strftime("%Y-%m-%d %H:%M:%S")

        logging.info("=== Backtest day %s ===", trade_day)

        run_script("apps.ChronoBridge.scripts.chronobridge_service",
                   "--mode", mode, "--start_date", start_str, "--end_date", end_str)

        run_script("apps.NeuralFusionCore.scripts.prediction_service",
                   "--mode", mode)

        run_script_safe("apps.NetWeaver.src.services.netweaver_prediction_service",
                        "--start_time", start_str, "--end_time", end_str,
                        "--future_steps", "60", "--no_timestamp")

        run_script("scripts.alphafusionnet_service")
        
def backtest_month(months_back: int, mode: str = "synchronize") -> None:
    """
    Run the backtest pipeline for all trading days in the month that is
    `months_back` months before today.

    - months_back = 0 → current month, but only up to yesterday (no future days)
    - months_back = 1 → last month (full month)
    - months_back = 2 → two months ago, etc.
    """
    today_utc = datetime.now(timezone.utc).date()
    year, month = month_from_offset(months_back, today_utc)

    logging.info(
        "Starting backtest for month offset=%d → %04d-%02d",
        months_back,
        year,
        month,
    )

    for trade_day in iter_trading_days_in_month(year, month):
        # If this is the *current* month, skip days that are today or in the future
        if (year, month) == (today_utc.year, today_utc.month) and trade_day >= today_utc:
            logging.info(
                "Skipping future/ongoing trading day %s (today=%s).",
                trade_day.isoformat(),
                today_utc.isoformat(),
            )
            continue

        prev_day = previous_trading_day(trade_day)

        start_dt = datetime(
            prev_day.year, prev_day.month, prev_day.day, 14, 30, 0, tzinfo=timezone.utc
        )
        end_dt = datetime(
            prev_day.year, prev_day.month, prev_day.day, 18, 30, 0, tzinfo=timezone.utc
        )

        start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        end_str = end_dt.strftime("%Y-%m-%d %H:%M:%S")

        logging.info(
            "=== Backtest for trading day %s using window %s → %s "
            "(prev trading day=%s) ===",
            trade_day.isoformat(),
            start_str,
            end_str,
            prev_day.isoformat(),
        )

        # 1) ChronoBridge
        run_script(
            "apps.ChronoBridge.scripts.chronobridge_service",
            "--mode",
            mode,
            "--start_date",
            start_str,
            "--end_date",
            end_str,
        )

        # 2) NeuralFusionCore
        run_script(
            "apps.NeuralFusionCore.scripts.prediction_service",
            "--mode",
            mode,
        )

        # 3) NetWeaver (safe)
        run_script_safe(
            "apps.NetWeaver.src.services.netweaver_prediction_service",
            "--start_time",
            start_str,
            "--end_time",
            end_str,
            "--future_steps",
            "60",
            "--no_timestamp",
        )

        # 4) AlphaFusionNet aggregation
        run_script(
            "scripts.alphafusionnet_service",
        )

        logging.info("=== Finished backtest day for %s ===", trade_day.isoformat())


# ---------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Run AlphaFusionNet backtesting for past time."
    )

    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Backtest only the last N trading days (ignores --month if provided).",
    )
    parser.add_argument(
        "--month",
        type=int,
        default=1,
        help=(
            "How many months back to backtest "
            "(1 = last month, 2 = two months ago, ...)."
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="synchronize",
        help="Feature / prediction mode (usually 'synchronize' or 'bridge').",
    )

    args = parser.parse_args()

    if args.days:
        backtest_last_days(args.days, mode=args.mode)
    else:
        if args.month < 0:
            raise ValueError("--month must be >= 0 (0 = this month, 1 = last month, ...).")
        backtest_month(months_back=args.month, mode=args.mode)
    

    


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )
    main()
