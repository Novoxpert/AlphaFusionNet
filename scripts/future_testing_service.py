#!/usr/bin/env python3
"""
future_testing_service.py

A service script for forward-looking (future) feature extraction and metric 
logging in live or near-real-time testing environments for the AlphaFusionNet 
pipeline. This script bridges the online prediction system and the features/
market data pipeline by:

1. Retrieving the most recent prediction timestamp and weights from MongoDB.
2. Determining the minimum historical data window required for inference.
3. Triggering the data ingest and feature generation services.
4. Loading the computed online features, aligning them to the last prediction timestamp.
5. Storing features and model weights for future audit/analysis in MongoDB.

This script is useful for validating predicted model allocations against 
subsequent market data, enabling real-time model evaluation and risk monitoring.

Author:
    Elham Esmaeilnia <elham.e.shirvani@gmail.com>

Date:
    2025-11-03

Version:
    1.1.0
"""

import os
import sys
import subprocess
import logging
import torch
import pandas as pd
from pymongo import MongoClient
from apps.NeuralFusionCore.config import Paths, FeatureCfg
import time
from pandas.tseries.frequencies import to_offset
from dotenv import load_dotenv

P = Paths(); F = FeatureCfg(); 
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

# --------------------------- MongoDB setup ---------------------------
load_dotenv()
mongo_user = os.getenv("NOVO_MONGO_USER")
mongo_pass = os.getenv("NOVO_MONGO_PASS")
mongo_host = os.getenv("NOVO_MONGO_HOST")
mongo_port = os.getenv("NOVO_MONGO_PORT")
mongo_auth_db = os.getenv("NOVO_MONGO_AUTH_DB")
mongo_db_name = os.getenv("NOVO_MONGO_DB")

mongo_uri = (
    f"mongodb://{mongo_user}:{mongo_pass}@"
    f"{mongo_host}:{mongo_port}/"
    f"{mongo_db_name}?authSource={mongo_auth_db}"
)

client = MongoClient(mongo_uri)
mongo_db = client[mongo_db_name]
alpha_col = mongo_db["AlphaFusionNet_predictions"]

# --------------------------- Data ingest & feature service ---------------------------
def run_data_ingest(start_time, end_time):
    logging.info(f"Running data_ingest_service to fetch data from {start_time} to {end_time}")
    subprocess.run([sys.executable, '-m', 'apps.NeuralFusionCore.scripts.data_ingest_service',
                    '--mode', 'custom',
                    '--start_time', str(start_time),
                    '--end_time', str(end_time)], check=True)

def run_feature_service(start_time, end_time):
    logging.info(f"Running features_service in future_testing mode from {start_time} to {end_time}")
    subprocess.run([sys.executable, '-m', 'apps.NeuralFusionCore.scripts.features_service',
                    '--mode', 'future_testing',
                    '--start_time', str(start_time),
                    '--end_time', str(end_time)], check=True)

# --------------------------- Helpers ---------------------------
def get_last_prediction_timestamp():
    """
    Fetch the last prediction document from AlphaFusionNet_predictions collection
    and return its timestamp and final_weights.
    """
    last_doc = alpha_col.find_one(sort=[("timestamp", -1)])
    if last_doc is None:
        logging.warning("No predictions found in AlphaFusionNet_predictions.")
        return None, None
    return pd.Timestamp(last_doc["timestamp"]), last_doc["final_weights"]

def calculate_ingest_window(last_ts, seq_len, freq="3min"):
    """
    Given the last prediction timestamp, calculate start_time based on seq_len and frequency.
    """
    freq_offset = to_offset(freq)
    start_time = last_ts - pd.Timedelta(minutes=seq_len * freq_offset.n)
    end_time = last_ts
    return start_time, end_time

def compute_metric(features_df, weights):
    """
    Placeholder metric computation function.
    Can be replaced with actual computation logic using weights and features.
    """
    # Example: weighted sum of feature columns (adjust as needed)
    feature_cols = [c for c in features_df.columns if c not in ["dateTime"]]
    metric = (features_df[feature_cols] * pd.Series(weights)).sum(axis=1).iloc[0]
    return metric

# --------------------------- Main ---------------------------
def main():
    start_service_time = time.time()
    torch.cuda.empty_cache()

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--mode", type=str,  default="latest", choices=["latest", "historical"], help="Execution mode")
    args = parser.parse_args()

    if args.mode == "latest":
        # 1. Fetch last prediction timestamp and weights
        last_ts, last_weights = get_last_prediction_timestamp()
        if last_ts is None:
            logging.info("No prediction available. Exiting.")
            return

        # 2. Check if prediction timestamp is in the future
        # Ensure now is UTC-aware
        now = pd.Timestamp.utcnow().tz_localize("UTC") if pd.Timestamp.utcnow().tzinfo is None else pd.Timestamp.utcnow()

        # Ensure last_ts is UTC-aware
        if last_ts.tzinfo is None:
            last_ts = last_ts.tz_localize("UTC")
        else:
            last_ts = last_ts.tz_convert("UTC")

        if now < last_ts:
            logging.info("Prediction timestamp is in the future. Waiting for data.")
            return

        # 3. Calculate minimal data fetch window
        seq_len = F.seq_len 
        start_time, end_time = calculate_ingest_window(last_ts, seq_len, freq="3min")
        logging.info(f"Fetching data from {start_time} to {end_time} for features calculation.")

        # 4. Run data ingest and feature service
        run_data_ingest(start_time, end_time)
        run_feature_service(start_time, end_time)

        # 5. Load features parquet file
        online_path = os.path.join(P.processed_dir, "online_metric.parquet")
        if not os.path.exists(online_path):
            logging.error(f"{online_path} not found. Exiting.")
            return
        df_te = pd.read_parquet(online_path)

        # 6. Select row corresponding to last prediction timestamp
        df_last = df_te[df_te['dateTime'] == last_ts]
        if df_last.empty:
            logging.warning("No feature row found for last prediction timestamp.")
            return

        # 7. Compute metric using last prediction weights
        #metric = compute_metric(df_last, last_weights)
        #logging.info(f"Metric for last prediction at {last_ts}: {metric}")
        # Convert df_last 'dateTime' to datetime
        df_last_copy = df_last.copy()
        df_last_copy['dateTime'] = df_last_copy['dateTime'].apply(lambda x: x.to_pydatetime() if hasattr(x, 'to_pydatetime') else x)

        # Convert last_ts to datetime
        last_ts_dt = last_ts.to_pydatetime() if hasattr(last_ts, 'to_pydatetime') else last_ts

        future_testing_col = mongo_db["AlphaFusionNet_future_testing"]
        doc = {
            "timestamp": last_ts_dt,
            "features": df_last.to_dict(orient="records"),
            "weights": last_weights,
            "created_at": pd.Timestamp.utcnow()
        }
        future_testing_col.insert_one(doc)

    elif args.mode == "historical":
        logging.info("Get historical predictions and data for calculating metrics.")
        # Implement historical mode logic if needed

    logging.info("Future testing cycle complete.")
    end_service_time = time.time()
    print(f"Time elapsed for Future testing service: {end_service_time - start_service_time:.2f} seconds")

if __name__ == "__main__":
    main()