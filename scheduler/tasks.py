"""
tasks.py
Description: Celery tasks for daily updates, predictions, metric computation,
             and startup workflows. Now includes global trading-day filtering.
Author: Elham Esmaeilnia(elham.e.shirvani@gmail.com)
Date: 2025 Oct 4
updated: 2025 Nov 19 (UTC alignment, calendar utils integration)
Version: 1.0.1
"""
import sys
import subprocess
from pathlib import Path
from datetime import datetime, timedelta, timezone

from celery import Celery
from celery.signals import before_task_publish

# Import your trading-day utilities (UTC-based)
from lib.trading_calendar_utils import (
    is_today_common_trading_day,
    next_common_trading_day,
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

BASE_DIR = Path(__file__).resolve().parent

# --------------------------------------------------------------------
# TRADING-DAY FILTER: SKIP SCHEDULED TASKS ON NON-TRADING DAYS
# --------------------------------------------------------------------
SCHEDULED_TASKS = {
    "tasks.daily_update",
    "tasks.prediction_14PM",
    "tasks.live_test_18PM_pluse_10min",
}


@before_task_publish.connect
def skip_if_not_trading_day(headers=None, **kwargs):
    """
    Prevent Celery beat from publishing a scheduled task if today is NOT
    a common trading day across all symbols.

    Note:
        is_today_common_trading_day() uses the current UTC date, so this
        decision is fully UTC-based and independent of local system time.
    """
    if not headers:
        return

    task_name = headers.get("task", "")

    if task_name in SCHEDULED_TASKS:
        if not is_today_common_trading_day():
            print(f"[SKIP] {task_name} blocked — today is NOT a trading day (UTC).")
            # Raising any exception here prevents publishing the task.
            raise Exception("Skip on non-trading day")


# --------------------------------------------------------------------
# UTILITY FUNCTIONS
# --------------------------------------------------------------------
def run_script(script, *args):
    """Runs a Python script or module."""
    if script == "-m":
        # Example: run_script('-m', 'scripts.compute_trading_days_service')
        cmd = [sys.executable, script, *args]
    else:
        cmd = [sys.executable, str(BASE_DIR / script), *args]

    print(f"[TASK] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=BASE_DIR)
    print(f"[TASK] Finished {script}")


def run_background(script, *args):
    """Runs a script/module in background (detached)."""
    if script == "-m":
        # Example: run_background('-m', 'scripts.alphafusionnet_api_service')
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
    # 1) Compute trading days cache (UTC-based)
    run_script("-m", "scripts.compute_trading_days_service")

    # 2) Historical ingest / feature / train pipeline
    run_script(
        "-m",
        "apps.ChronoBridge.scripts.data_ingest_service",
        "--mode",
        "historical",
        "--days",
        "150",
    )

    run_script(
        "-m",
        "apps.ChronoBridge.scripts.features_service",
        "--mode",
        "train",
        "--history_days",
        "150",
    )

    run_script(
        "-m",
        "apps.NeuralFusionCore.scripts.train_service",
        "--epochs",
        "50",
    )

    # FIX: module name without .py when using -m
    run_script(
        "-m",
        "apps.ChronoBridge.scripts.chronobridge_service",
        "--mode",
        "bridge",
        "--history_days",
        "150",
    )

    run_script(
        "-m",
        "apps.NetWeaver.src.services.netweaver_train_service",
        "--latest_month",
        "4",
        "--no_analysis",
    )

    # 3) Start APIs in background
    run_background("-m", "apps.ChronoBridge.scripts.chronobridge_api_service")
    run_background("-m", "scripts.alphafusionnet_api_service")

    print("[TASK] API services started in background")


# --------- DAILY WORKFLOW ----------
@app.task
def daily_update():
    # All these scripts should themselves treat time in UTC internally.
    run_script(
        "-m",
        "apps.NeuralFusionCore.scripts.data_ingest_service",
        "--mode",
        "latest",
        "--hours",
        "20",
    )

    run_script(
        "-m",
        "apps.NeuralFusionCore.scripts.features_service",
        "--mode",
        "finetune",
        "--latest_hours",
        "20",
    )

    run_script(
        "-m",
        "apps.NeuralFusionCore.scripts.finetune_service",
        "--epochs",
        "30",
    )

    run_script(
        "-m",
        "apps.ChronoBridge.scripts.chronobridge_service",
        "--mode",
        "bridge",
        "--hours",
        "20",
    )

    run_script(
        "-m",
        "apps.NetWeaver.src.services.netweaver_finetune_service",
        "--latest_hours",
        "20",
        "--no_analysis",
    )


# --------- METRIC LIVE ----------
@app.task
def calculate_metric_live():
    run_script("-m", "scripts.metric_live_service")


# --------- PREDICTION AT 14:00 UTC ----------
@app.task
def prediction_14PM():
    """
    Prediction task intended to run at 14:00 UTC.

    After finishing the prediction workflow, it schedules metric calculations
    every minute until 18:00 UTC (same day).
    """

    # 1) Prediction workflow (all internal scripts should use UTC)
    run_script(
        "-m",
        "apps.ChronoBridge.scripts.chronobridge_service",
        "--mode",
        "synchronize",
        "--hours",
        "7",
    )

    run_script(
        "-m",
        "apps.NeuralFusionCore.scripts.prediction_service",
        "--mode",
        "synchronize",
        "--hours",
        "7",
    )

    run_script(
        "-m",
        "apps.NetWeaver.src.services.netweaver_prediction_service",
        "--latest_hours",
        "7",
        "--future_steps",
        "80",
        "--no_timestamp",
    )

    run_script("-m", "scripts.alphafusionnet_service")
    run_script("-m", "scripts.metric_monthly_service")

    # 2) Schedule live-metric runs every minute until 18:00 UTC
    now_utc = datetime.now(timezone.utc)
    end_time_utc = now_utc.replace(hour=18, minute=0, second=0, microsecond=0)

    # If for some reason this runs after 18:00 UTC, don't schedule anything
    if end_time_utc <= now_utc:
        print("[TASK] prediction_14PM: current time is past 18:00 UTC; no metrics scheduled.")
        return "Prediction finished. No metrics scheduled (past 18:00 UTC)."

    t = now_utc
    while t <= end_time_utc:
        delta = (t - now_utc).total_seconds()
        # countdown is relative, so UTC/offset is preserved by definition
        calculate_metric_live.apply_async(countdown=delta)
        t += timedelta(minutes=1)

    return "Prediction finished. Metrics scheduled every minute until 18:00 UTC."


# --------- LIVE TEST AT 18:05 UTC ----------
@app.task
def live_test_18PM_pluse_10min():
    # You can implement whatever validation / smoke test you want here,
    # but the scheduler (Celery beat) should trigger this at 18:10 UTC.
    return "Live test at 18:10 UTC placeholder."
