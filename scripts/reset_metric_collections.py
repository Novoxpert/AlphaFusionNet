"""
Reset (clean) metric collections and run a 3-month backtest.

This script:
1. Loads AFN_config.yaml 
2. Connects to Mongo via init_mongo_client
3. DROPS the following collections:
       - AlphaFusionNet_predictions
       - windows
       - live_metric
       - monthly
4. Runs metric_backtesting with BACKTEST_DAYS=90
------
Author: Elham Esmaeilnia(elham.e.shirvani@gmail.com)
Version: 1.0.0
"""

import os
import logging
from datetime import timedelta
from importlib import import_module

from lib.db_utils import init_mongo_client
from scripts.metric_backtesting import load_config

# ------------------------
# Logging
# ------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("reset_script")


# ------------------------
# Collections to wipe
# ------------------------
TARGET_COLLECTIONS = [
    #"AlphaFusionNet_predictions",
    "windows",
    "live_metrics",
    "monthly",
    #"NeuralFusionCore_predictions",
    #"NetWeaver_predictions",
    #"chrono_bridge"
    ""
    ""
]


def drop_metric_collections(db):
    logger.info("Dropping metric collections:")
    for col in TARGET_COLLECTIONS:
        try:
            db[col].drop()
            logger.info("  ✔ Dropped: %s", col)
        except Exception as e:
            logger.error("  ✖ Error dropping %s: %s", col, e)


def run_backtest_3_months():
    """
    Dynamically import your metric_backtesting entrypoint,
    override BACKTEST_DAYS to 90, and run it.
    """
    logger.info("Running 3-month metric backtest...")

    mod = import_module("scripts.metric_backtesting")

    # override the constant in the imported module
    mod.BACKTEST_DAYS = 90

    # run as if __main__
    mod.__name__ = "__main__"
    mod.__file__ = "scripts/metric_backtesting.py"

    # direct call to the bottom of script
    # (same pattern as running: python -m scripts.metric_backtesting)
    exec(open(mod.__file__).read(), mod.__dict__)


def main():
    logger.info("Loading config...")
    cfg = load_config()

    logger.info("Connecting to Mongo...")
    mongo_cfg = cfg["novo_mongo"]
    mongo_client, db = init_mongo_client(mongo_cfg)

    try:
        drop_metric_collections(db)
    finally:
        try:
            mongo_client.close()
        except:
            pass

    # After cleaning the DB, run the full backtest for 3 months
    #run_backtest_3_months()


if __name__ == "__main__":
    main()
