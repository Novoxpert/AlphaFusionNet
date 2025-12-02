"""
View contents of metric-related Mongo collections.

This script uses Mongo client 

Collections inspected:
    - AlphaFusionNet_predictions
    - windows
    - live_metric
    - monthly
    - NeuralFusionCore_predictions
    - NetWeaver_predictions

Usage:
    python -m scripts.view_metric_collections --head 5
    python -m scripts.view_metric_collections --all
------
Author: Elham Esmaeilnia(elham.e.shirvani@gmail.com)
Date: 2025 Nov 30
Version: 1.0.0
"""

import argparse
import json
from pprint import pprint

from lib.db_utils import init_mongo_client
from scripts.metric_backtesting import load_config


TARGET_COLLECTIONS = [
    #"AlphaFusionNet_predictions",
    "windows",
    #"live_metrics",
    #"monthly",
    #"NeuralFusionCore_predictions",
    #"NetWeaver_predictions",
    #"chrono_bridge"
]


def view_collection(db, name, limit=None, show_all=False):
    print(f"\n======================= {name} =======================")

    count = db[name].count_documents({})
    print(f"Total documents: {count}")

    if count == 0:
        return

    if show_all:
        cursor = db[name].find({})
    else:
        cursor = db[name].find({}).limit(limit)

    print(f"\nShowing {'ALL' if show_all else limit} documents:\n")
    for doc in cursor:
        pprint(doc)
        print("-" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head", type=int, default=1,
                        help="Number of documents to preview from each collection")
    parser.add_argument("--all", action="store_true",
                        help="Show ALL documents (careful if collections are large!)")
    args = parser.parse_args()

    cfg = load_config()
    mongo_cfg = cfg["novo_mongo"]
    client, db = init_mongo_client(mongo_cfg)

    print("\nConnected to Mongo ✓\n")

    for col in TARGET_COLLECTIONS:
        view_collection(
            db,
            col,
            limit=args.head,
            show_all=args.all,
        )

    client.close()


if __name__ == "__main__":
    main()
