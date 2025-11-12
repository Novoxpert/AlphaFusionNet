"""
metric_service.py

Provides utilities to:
 - fetch OHLCV (minute or arbitrary range) from ClickHouse
 - compute daily returns from minute-level prices
 - build portfolio daily returns from daily weights
 - compute dashboard metrics for a target month
 - persist metrics snapshots to MongoDB

Assumptions / Requirements
 - clickhouse-driver (Client) is available as `clickhouse_driver.Client`
 - pymongo is available and MongoDB URI/DB name provided via config
 - pandas, numpy

Usage
 - Put this file in your service project, import functions you need.
 - Configure CH and Mongo clients (see `init_clients_from_config`).



Author: Elham Esmaeilnia(elham.e.shirvani@gmail.com)
Date: 2025 Nov 12
Version: 1.1.0 
"""


import time
import logging, os
import yaml
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from lib.db_utils import init_clickhouse_client, init_mongo_client
from lib.metric_utils import fetch_ohlcv_range
from dotenv import load_dotenv
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "config", "AFN_config.yaml")
#-------------------------
# streaming process
#-------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def run_live_metric_stream(db,
                           ch_client,
                           symbol_list,
                           portfolio_weights,
                           window_id,
                           ch_table,
                           start_time_utc,
                           duration_hours=4,
                           update_interval_sec=60,
                           initial_nav=100_000.0):
    """
    Live 4-hour streaming loop: compute and publish real-time portfolio metrics every minute.

    Args:
        db: MongoDB database object
        ch_client: ClickHouse client
        symbol_list: list of symbols in the portfolio
        portfolio_weights: dict {symbol: weight}
        window_id: unique identifier for current trading window (e.g. '2025-11-12T14Z')
        ch_table: ClickHouse table name for market data
        start_time_utc: datetime UTC (t₀)
        duration_hours: 4 by default
        update_interval_sec: 60 (every minute)
        initial_nav: starting NAV in USD
    """

    end_time_utc = start_time_utc + timedelta(hours=duration_hours)
    logger.info(f"Starting live window {window_id} from {start_time_utc} to {end_time_utc}")

    # Fetch entry prices at t₀
    entry_prices = {}
    for sym in symbol_list:
        df = fetch_ohlcv_range(ch_client, ch_table, sym,
                               start_time_utc - timedelta(minutes=5),
                               start_time_utc)
        if df.empty:
            logger.warning(f"No entry price for {sym}")
            return
        entry_prices[sym] = df['close'].iloc[-1]

    # Compute fixed positions
    weights = pd.Series(portfolio_weights)
    weights = weights.clip(lower=0)
    weights = weights / weights.sum()
    qty = {sym: (initial_nav * w) / entry_prices[sym] for sym, w in weights.items()}
    cash0 = initial_nav * (1 - weights.sum())

    # Persist initial window state
    state_doc = {
        "window_id": window_id,
        "t0": start_time_utc,
        "t1": end_time_utc,
        "NAV0": initial_nav,
        "entry_prices": entry_prices,
        "quantities": qty,
        "cash0": cash0,
        "symbols": symbol_list,
        "weights": portfolio_weights,
        "status": "LIVE"
    }
    db["AlphaFusionNet_windows"].insert_one(state_doc)

    # Minute-by-minute loop
    current_time = start_time_utc
    last_prices = entry_prices.copy()

    while datetime.utcnow() < end_time_utc:
        loop_start = datetime.utcnow()

        # Fetch last 5 minutes of data for all symbols
        now = datetime.utcnow()
        for sym in symbol_list:
            df = fetch_ohlcv_range(ch_client, ch_table, sym,
                                   now - timedelta(minutes=5), now)
            if not df.empty:
                last_prices[sym] = df['price'].iloc[-1]

        # Compute portfolio value
        Vt = sum(qty[sym] * last_prices[sym] for sym in symbol_list) + cash0
        Rp = Vt / initial_nav - 1
        PnL = Vt - initial_nav

        snapshot = {
            "asOf": now,
            "window_id": window_id,
            "NAV0": initial_nav,
            "NAVt": Vt,
            "PnL": PnL,
            "cumulative_return": Rp,
            "status": "LIVE" if now < end_time_utc else "ENDED"
        }

        # Save to Mongo for dashboard
        db["AlphaFusionNet_live_metrics"].insert_one(snapshot)
        logger.info(f"[{now}] NAV={Vt:.2f} PnL={PnL:.2f} Return={Rp*100:.2f}%")

        # Sleep until next minute mark
        time_elapsed = (datetime.utcnow() - loop_start).total_seconds()
        sleep_time = max(0, update_interval_sec - time_elapsed)
        time.sleep(sleep_time)

    # Final snapshot marking end of window
    db["AlphaFusionNet_live_metrics"].insert_one({
        "window_id": window_id,
        "asOf": datetime.utcnow(),
        "status": "ENDED"
    })
    logger.info(f"Window {window_id} ended")

def load_config(config_path: str = CONFIG_FILE):
    # Load environment variables from .env
    load_dotenv()

    # Read YAML as text
    with open(config_path, "r") as f:
        raw_yaml = f.read()

    # Substitute ${VAR} with environment values
    expanded_yaml = os.path.expandvars(raw_yaml)

    # Parse YAML
    cfg = yaml.safe_load(expanded_yaml)
    if not cfg:
        raise RuntimeError(f"Config file empty or invalid: {config_path}")

    return cfg

if __name__ == "__main__":

    cfg = load_config()
    ch_cfg = cfg["clickhouse"]
    ch_client = init_clickhouse_client(host=ch_cfg["host"],
        port=int(ch_cfg["port"]),
        database=ch_cfg["database"],
        user=ch_cfg["user"],
        password=ch_cfg["password"]
    )

    mongo_cfg = cfg["mongo"]
    mongo_client, db = init_mongo_client(mongo_cfg)

    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    start_time = now  # assume 14:00 UTC
    weights_doc = db["AlphaFusionNet_predictions"].find_one(sort=[("timestamp", -1)])
    portfolio_weights = weights_doc["final_weights"]

    symbols = list(portfolio_weights.keys())
    run_live_metric_stream(db, ch_client, symbols, portfolio_weights,
                           window_id=f"{start_time:%Y%m%d_%H%M}",
                           ch_table=ch_cfg["table"],
                           start_time_utc=start_time)
