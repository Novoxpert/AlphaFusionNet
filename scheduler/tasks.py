
"""
tasks.py
Description: Configures the Celery beat scheduler to automatically trigger periodic workflows
             (daily updates and 4-hourly predictions) at defined times.
Author: Elham Esmaeilnia(elham.e.shirvani@gmail.com)
Date: 2025 Oct 4
updated: 2025 Oct 19
Version: 1.0.0 
"""
import subprocess, sys
from celery import Celery
from pathlib import Path

# Celery setup
app = Celery('tasks', broker='redis://localhost:6379/1',backend='redis://localhost:6379/1')

# Automatically detect project directory (where tasks.py is located)
BASE_DIR = Path(__file__).resolve().parent

from pathlib import Path
import subprocess
import sys

BASE_DIR = Path(__file__).resolve().parent

def run_script(script, *args):
    """
    Helper to run Python scripts or modules from the project folder.
    
    Supports both:
        - Running modules (e.g. run_script('-m', 'NeuralFusionCore.data_ingest_service', '--mode', 'latest'))
        - Running scripts (e.g. run_script('NeuralFusionCore/data_ingest_service.py', '--mode', 'latest'))

    Args:
        script (str): Either '-m' (to run a module) or the path to a Python script.
        *args: Additional arguments passed to the script or module.
    """
    if script == "-m":
        # Running as module
        cmd = [sys.executable, script, *args]
    else:
        # Running as local script file
        cmd = [sys.executable, str(BASE_DIR / script), *args]

    print(f"[TASK] Running {' '.join(cmd)}")

    subprocess.run(cmd, check=True, cwd=BASE_DIR)

    print(f"[TASK] Finished {script}")


def run_background(script, *args):
    """
    Launch a Python module or script in the background using subprocess.Popen.

    Supports both:
        - Modules: run_background('-m', 'NeuralFusionCore.scripts.api_service')
        - Scripts: run_background('NeuralFusionCore/scripts/api_service.py')

    Args:
        script (str): Either '-m' (for module execution) or a Python script path.
        *args: Additional CLI args.
    """
    if script == "-m":
        # Run a module: e.g. python -m NeuralFusion.api_service
        cmd = [sys.executable, script, *args]
    else:
        # Run a file: e.g. python BASE_DIR/NeuralFusion/api_service.py
        cmd = [sys.executable, str(BASE_DIR / script), *args]

    print(f"[TASK] Starting background process: {' '.join(cmd)}")

    subprocess.Popen(
        cmd,
        cwd=BASE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
# --------- One-time startup workflow ----------
@app.task
def initial_run():
   
    run_script('-m','apps.NeuralFusionCore.scripts.data_ingest_service', '--mode', 'latest', '--hours', '6')
    run_script('-m','apps.NeuralFusionCore.scripts.features_service', '--mode', 'train', '--latest_hours', '6')
    run_script('-m','apps.NeuralFusionCore.scripts.train_service', '--epochs', '5')
    run_script('-m','apps.NeuralFusionCore.scripts.prediction_service', '--hours', '1')
    #run_background('-m', 'apps.NeuralFusionCore.scripts.api_service')
    run_background('-m', 'scripts.alphafusionnet_api_service')
    run_script('-m','apps.ChronoBridge.scripts.chronobridge_service.py', '--hours', '1')
    run_background('-m', 'apps.ChronoBridge.scripts.chronobridge_api_service')
   
    print("[TASK] API service started in background")

# --------- Daily workflow ----------
@app.task
def daily_update():

    run_script('-m','apps.NeuralFusionCore.scripts.data_ingest_service', '--mode', 'latest', '--hours', '6')
    run_script('-m','apps.NeuralFusionCore.scripts.features_service', '--mode', 'finetune', '--latest_hours', '6')
    run_script('-m','apps.NeuralFusionCore.scripts.finetune_service', '--epochs', '2')

# --------- 4-hourly prediction workflow ----------
@app.task
def prediction_4h():
 
    run_script('-m','apps.NeuralFusionCore.scripts.prediction_service', '--hours', '1')
    run_script('-m','apps.ChronoBridge.scripts.chronobridge_service', '--hours', '1')
    run_script('-m','scripts.alphafusionnet_service')
    
