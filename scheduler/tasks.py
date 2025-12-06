"""
tasks.py
Description: Celery tasks for daily updates, predictions, metric computation,
             and startup workflows. Now includes global trading-day filtering.
Author: Elham Esmaeilnia (elham.e.shirvani@gmail.com)
Date: 2025 Oct 4
updated: 2025 Nov 19 (UTC alignment, calendar utils integration)
Version: 1.0.2 
"""
import sys
import logging
import subprocess
from pathlib import Path
from datetime import datetime, timedelta, timezone

from celery import Celery
from celery.signals import before_task_publish
import time
import torch

# Import your trading-day utilities (UTC-based)
from lib.trading_calendar_utils import (
    is_today_common_trading_day,
    previous_common_trading_day,
)

# --------------------------------------------------------------------
# CELERY SETUP
# --------------------------------------------------------------------
app = Celery(
    "tasks",
    broker="redis://localhost:6379/1",
    backend="redis://localhost:6379/1",
)

# Make Celery fully UTC-based
app.conf.timezone = "UTC"
app.conf.enable_utc = True

# Prevent celery from retrying skipped publishes
app.conf.task_publish_retry = False

BASE_DIR = Path(__file__).resolve().parent

# --------------------------------------------------------------------
# TRADING-DAY FILTER: SKIP SCHEDULED TASKS ON NON-TRADING DAYS
# --------------------------------------------------------------------
SCHEDULED_TASKS = {
    "tasks.daily_update",
    "tasks.prediction_14_30PM",
    "tasks.calculate_metric_live",
}


@before_task_publish.connect
def skip_if_not_trading_day(headers=None, **kwargs):
    """
    Prevent Celery beat from publishing a scheduled task if today is NOT
    a common trading day across all symbols (UTC-based).
    """
    if not headers:
        return

    task_name = headers.get("task", "")

    if task_name in SCHEDULED_TASKS:
        # We silently skip (do NOT raise!)
        if not is_today_common_trading_day():
            print(f"[SKIP] {task_name} blocked — today is NOT a trading day (UTC).")
            return False   # silently cancel publish!


# --------------------------------------------------------------------
# UTILITY FUNCTIONS
# --------------------------------------------------------------------
def run_script(script, *args):
    """Runs a Python script or module."""
    if script == "-m":
        cmd = [sys.executable, script, *args]
    else:
        cmd = [sys.executable, str(BASE_DIR / script), *args]

    print(f"[TASK] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=BASE_DIR)
    print(f"[TASK] Finished {script}")


def run_script_safe(script, *args):
    """Runs a Python script or module safely, logging errors."""
    if script == "-m":
        cmd = [sys.executable, script, *args]
    else:
        cmd = [sys.executable, str(BASE_DIR / script), *args]

    print(f"[TASK] Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, cwd=BASE_DIR)
        print(f"[TASK] Finished {script}")
    except subprocess.CalledProcessError as e:
        logging.error(
            "Service step failed: %s (exit code %s). Continuing...",
            e, e.returncode,
        )
    except Exception as e:
        logging.error("Unexpected error in this service step: %s. Continuing...", e)


def run_background(script, *args):
    """Runs a script/module in background (detached)."""
    if script == "-m":
        cmd = [sys.executable, script, args[0], *args[1:]]
    else:
        script_path = BASE_DIR / script
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")
        cmd = [sys.executable, str(script_path), *args]

    print(f"[TASK] Starting background process: {' '.join(cmd)}")
    subprocess.Popen(
        cmd,
        cwd=BASE_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        close_fds=True,
    )


# --------------------------------------------------------------------
# TASK DEFINITIONS
# --------------------------------------------------------------------

# --------- INITIAL STARTUP ----------
@app.task
def initial_run():
    run_script("-m", "scripts.compute_trading_days_service")
    run_script("-m", "apps.ChronoBridge.scripts.data_ingest_service", "--mode", "historical", "--days", "150")
    run_script("-m", "apps.ChronoBridge.scripts.features_service", "--mode", "train", "--history_days", "150")
    run_script("-m", "apps.NeuralFusionCore.scripts.train_service", "--epochs", "50")
    run_script("-m", "apps.ChronoBridge.scripts.chronobridge_service", "--mode", "bridge", "--history_days", "150")
    run_background("-m", "apps.ChronoBridge.scripts.chronobridge_api_service")
    run_script_safe("-m", "apps.NetWeaver.src.services.netweaver_train_service", "--latest_month", "4", "--no_analysis")
    print("[TASK] API services started in background")


# --------- DAILY WORKFLOW ----------
@app.task
def daily_update():
    run_script("-m", "apps.NeuralFusionCore.scripts.data_ingest_service", "--mode", "latest", "--hours", "20")
    run_script("-m", "apps.NeuralFusionCore.scripts.features_service", "--mode", "finetune", "--latest_hours", "20")
    run_script("-m", "apps.NeuralFusionCore.scripts.finetune_service", "--epochs", "30")
    run_script("-m", "apps.ChronoBridge.scripts.chronobridge_service", "--mode", "bridge", "--hours", "20")
    run_script_safe("-m", "apps.NetWeaver.src.services.netweaver_finetune_service", "--latest_hours", "20", "--no_analysis")


# --------- METRIC LIVE ----------
@app.task
def calculate_metric_live():
    run_script("-m", "scripts.metric_live_service")


# --------- PREDICTION AT 14:30 UTC ----------
@app.task
def prediction_14_30PM():
    start_prediction_time = time.time()
    torch.cuda.empty_cache()

    today = datetime.now(timezone.utc).date()
    last_trading_day = previous_common_trading_day(today)
    if last_trading_day is None:
        raise RuntimeError("No previous trading day found in cache!")

    # Prediction pipeline
    run_script("-m", "apps.ChronoBridge.scripts.chronobridge_service",
               "--mode", "synchronize",
               "--start_date", str(int(datetime(last_trading_day.year, last_trading_day.month, last_trading_day.day, 14, 30, tzinfo=timezone.utc).timestamp())),
               "--end_date", str(int(datetime(last_trading_day.year, last_trading_day.month, last_trading_day.day, 18, 30, tzinfo=timezone.utc).timestamp())))

    run_script("-m", "apps.NeuralFusionCore.scripts.prediction_service",
               "--mode", "synchronize")

    run_script_safe("-m", "apps.NetWeaver.src.services.netweaver_prediction_service",
                    "--start_str", str(int(datetime(last_trading_day.year, last_trading_day.month, last_trading_day.day, 14, 30, tzinfo=timezone.utc).timestamp())),
                    "--end_str", str(int(datetime(last_trading_day.year, last_trading_day.month, last_trading_day.day, 18, 30, tzinfo=timezone.utc).timestamp())),
                    "--future_steps", "60",
                    "--no_timestamp")

    run_script("-m", "scripts.alphafusionnet_service")

    print(f"Time elapsed: {time.time() - start_prediction_time:.2f} seconds")

    # === schedule metric live between 14:30 and 18:30 UTC ===

    now_utc = datetime.now(timezone.utc)
    start_dt = datetime(last_trading_day.year, last_trading_day.month, last_trading_day.day, 14, 30, tzinfo=timezone.utc)
    end_dt   = datetime(last_trading_day.year, last_trading_day.month, last_trading_day.day, 18, 30, tzinfo=timezone.utc)

    # If we're BEFORE the 14:30 window -> wait until exactly 14:30
    if now_utc < start_dt:
        initial_delay = (start_dt - now_utc).total_seconds()
        print(f"[TASK] Waiting {initial_delay:.2f}s until 14:30 UTC to start metric_live")
        calculate_metric_live.apply_async(countdown=initial_delay)
        first_metric_time = start_dt

    # If we're already IN the window
    elif start_dt <= now_utc <= end_dt:
        print("[TASK] Starting metric_live immediately (inside window).")
        calculate_metric_live.delay()
        first_metric_time = now_utc

    # If it's past 18:30 -> nothing
    else:
        print("[TASK] prediction_14_30PM: current time is past 18:30 UTC; no metrics scheduled.")
        return "Prediction finished. No metrics scheduled (past 18:30 UTC)."

    # schedule every full minute after first execution
    t = (first_metric_time + timedelta(minutes=1)).replace(second=0, microsecond=0)

    while t <= end_dt:
        delta = (t - now_utc).total_seconds()
        calculate_metric_live.apply_async(countdown=delta)
        t += timedelta(minutes=1)

    run_script("-m", "scripts.metric_monthly_service")
    return "Prediction finished. Metrics scheduled every minute for full trading window."
