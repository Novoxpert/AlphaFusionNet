
"""
tasks.py
Description: Celery tasks for daily updates, predictions, metric computation,
             and startup workflows. Now includes global trading-day filtering.
Author: Elham Esmaeilnia(elham.e.shirvani@gmail.com)
Date: 2025 Oct 4
updated: 2025 Oct 19
Version: 1.0.0 
"""
import sys
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

from celery import Celery
from celery.signals import before_task_publish

# Import your trading-day utilities
from lib.trading_calender_utils import (
    is_today_common_trading_day,
    next_common_trading_day
)

# --------------------------------------------------------------------
# CELERY SETUP
# --------------------------------------------------------------------
app = Celery(
    'tasks',
    broker='redis://localhost:6379/1',
    backend='redis://localhost:6379/1'
)

BASE_DIR = Path(__file__).resolve().parent


# --------------------------------------------------------------------
# TRADING-DAY FILTER: SKIP SCHEDULED TASKS ON NON-TRADING DAYS
# --------------------------------------------------------------------
SCHEDULED_TASKS = {
    "tasks.daily_update",
    "tasks.prediction_14PM",
    "tasks.live_test_18PM_pluse_10min"
}

@before_task_publish.connect
def skip_if_not_trading_day(headers=None, **kwargs):
    """
    Prevent Celery beat from publishing a scheduled task if today is NOT
    a common trading day across all symbols.
    """
    if not headers:
        return

    task_name = headers.get("task", "")

    if task_name in SCHEDULED_TASKS:
        if not is_today_common_trading_day():
            print(f"[SKIP] {task_name} blocked — today is NOT a trading day.")
            raise Exception("Skip on non-trading day")


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


def run_background(script, *args):
    """Runs a script/module in background."""
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
        close_fds=True
    )


# --------------------------------------------------------------------
# TASK DEFINITIONS
# --------------------------------------------------------------------

# --------- INITIAL STARTUP ----------
@app.task
def initial_run():

    run_script('-m', 'scripts.compute_trading_days_service')

    run_script('-m', 'apps.ChronoBridge.scripts.data_ingest_service',
               '--mode', 'historical', '--days', '150')

    run_script('-m', 'apps.ChronoBridge.scripts.features_service',
               '--mode', 'train', '--history_days', '150')

    run_script('-m', 'apps.NeuralFusionCore.scripts.train_service', '--epochs', '50')

    run_script('-m', 'apps.ChronoBridge.scripts.chronobridge_service.py',
               '--mode', 'bridge', '--history_days', '150')

    run_script('-m', 'apps.NetWeaver.src.services.netweaver_train_service',
               '--latest_month', '4', '--no_analysis')

    run_background('-m', 'apps.ChronoBridge.scripts.chronobridge_api_service')
    run_background('-m', 'scripts.alphafusionnet_api_service')

    print("[TASK] API service started in background")


# --------- DAILY WORKFLOW ----------
@app.task
def daily_update():

    run_script('-m', 'apps.NeuralFusionCore.scripts.data_ingest_service',
               '--mode', 'latest', '--hours', '20')

    run_script('-m', 'apps.NeuralFusionCore.scripts.features_service',
               '--mode', 'finetune', '--latest_hours', '20')

    run_script('-m', 'apps.NeuralFusionCore.scripts.finetune_service',
               '--epochs', '30')

    run_script('-m', 'apps.ChronoBridge.scripts.chronobridge_service.py',
               '--mode', 'bridge', '--hours', '20')

    run_script('-m', 'apps.NetWeaver.src.services.netweaver_finetune_service',
               '--latest_hours', '20', '--no_analysis')


# --------- METRIC LIVE ----------
@app.task
def calculate_metric_live():
    run_script('-m', 'scripts.metric_live_service')


# --------- PREDICTION AT 14:00 ----------
@app.task
def prediction_14PM():

    # 1) Prediction workflow
    run_script('-m', 'apps.ChronoBridge.scripts.chronobridge_service',
               '--mode', 'synchronize', '--hours', '7')

    run_script('-m', 'apps.NeuralFusionCore.scripts.prediction_service',
               '--mode', 'synchronize', '--hours', '7')

    run_script('-m', 'apps.NetWeaver.src.services.netweaver_prediction_service',
               '--latest_hours', '7', '--future_steps', '80', '--no_timestamp')

    run_script('-m', 'scripts.alphafusionnet_service')
    run_script('-m', 'scripts.metric_monthly_service')

    # 2) Schedule live-metric runs every min until 18:00
    now = datetime.now()
    end_time = now.replace(hour=18, minute=0, second=0, microsecond=0)

    t = now
    while t <= end_time:
        delta = (t - now).total_seconds()
        calculate_metric_live.apply_async(countdown=delta)
        t += timedelta(minutes=1)

    return "Prediction finished. Metrics scheduled."


# --------- LIVE TEST AT 18:05 ----------
@app.task
def live_test_18PM_pluse_10min():
    return