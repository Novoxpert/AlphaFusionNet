"""
Show Backtest Metrics (Rich Version)
====================================

This script connects to MongoDB using the SAME config + client
as `metric_backtesting.py`, and can:

    • list             : show latest windows
    • window           : show snapshots for one window
    • export           : export a window's NAV/benchmark time series to CSV
    • plot             : plot NAV (and benchmark if present) for a window
    • export_monthly   : export monthly metrics collection to CSV
    • export_live      : export live_metrics collection to CSV
    • export_windows   : export windows collection to CSV
    • export_predictions: export AlphaFusionNet_predictions collection to CSV

Usage examples
--------------
List latest 10 windows:
    python -m scripts.show_backtest_metrics --action list --limit 10

Show first 5 snapshots for a specific window:
    python -m scripts.show_backtest_metrics --action window --window-id 20251118_1430 --limit 5

Export all NAV data for a window to CSV (time series):
    python -m scripts.show_backtest_metrics --action export --window-id 20251118_1430 --out nav_20251118_1430.csv

Plot NAV (and benchmark if available) for a window:
    python -m scripts.show_backtest_metrics --action plot --window-id 20251118_1430

Export monthly metrics (optionally filtered by year/month) to CSV:
    python -m scripts.show_backtest_metrics --action export_monthly --out monthly_metrics.csv
    python -m scripts.show_backtest_metrics --action export_monthly --year 2025 --month 11 --out monthly_2025_11.csv

Export live_metrics (optionally filtered by window_id) to CSV:
    python -m scripts.show_backtest_metrics --action export_live --out live_metrics_all.csv
    python -m scripts.show_backtest_metrics --action export_live --window-id 20251118_1430 --out live_20251118_1430.csv

Export windows collection to CSV:
    python -m scripts.show_backtest_metrics --action export_windows --out windows_all.csv

Export AlphaFusionNet_predictions collection to CSV:
    python -m scripts.show_backtest_metrics --action export_predictions --out preds_all.csv

Notes
-----
• live_metrics time series export for a single window is under --action export
  (NAV/benchmark series only).
• Collection-wide exports (live/windows/predictions/monthly) dump *all* top-level
  fields (except _id). Nested structures are stringified.
------
Author: Elham Esmaeilnia(elham.e.shirvani@gmail.com)
Version: 2.0.0
"""

import os
import argparse
import csv
from datetime import datetime
from dotenv import load_dotenv
import yaml

from lib.db_utils import init_mongo_client

# Optional plotting: if matplotlib isn't installed, plot action will fail softly.
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


CONFIG_FILE = os.environ.get("ALPHAFUSIONNET_CONFIG", "config/AFN_config.yaml")
MONTHLY_COLLECTION = "monthly" 


# ------------------------
# Config loader
# ------------------------
def load_config(config_path: str = CONFIG_FILE):
    load_dotenv()

    with open(config_path, "r") as f:
        raw_yaml = f.read()

    expanded_yaml = os.path.expandvars(raw_yaml)
    cfg = yaml.safe_load(expanded_yaml)
    if not cfg:
        raise RuntimeError(f"Config file empty or invalid: {config_path}")
    return cfg


# ------------------------
# Generic collection -> CSV helper
# ------------------------
def export_collection_to_csv(coll, out_path: str, query: dict | None = None, limit: int | None = None):
    """
    Generic helper: export a collection to CSV.

    - query: Mongo filter dict (or None for all docs)
    - limit: optional integer limit
    - Field names: union of all top-level keys across docs, excluding _id
    - Nested dict/list values: stringified
    """
    if query is None:
        query = {}

    cursor = coll.find(query)
    if limit is not None and limit > 0:
        cursor = cursor.limit(limit)

    docs = list(cursor)
    if not docs:
        print(f"No documents found in collection='{coll.name}' for query={query}. Nothing to export.")
        return 0

    # Collect all top-level field names (excluding _id)
    fieldnames = set()
    for d in docs:
        for k in d.keys():
            if k == "_id":
                continue
            fieldnames.add(k)
    fieldnames = sorted(fieldnames)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for d in docs:
            row = {}
            for k in fieldnames:
                v = d.get(k)
                if isinstance(v, (dict, list)):
                    v = str(v)
                row[k] = v
            writer.writerow(row)

    print(f"Exported {len(docs)} documents from collection='{coll.name}' to {out_path}")
    return len(docs)


# ------------------------
# Actions on windows / live metrics / predictions
# ------------------------
def list_windows(db, limit=10):
    print("\n================ WINDOWS (latest) ================")
    cursor = db["windows"].find().sort("t0", -1).limit(limit)
    for doc in cursor:
        window_id = doc.get("window_id")
        t0 = doc.get("t0")
        t1 = doc.get("t1")
        nav0 = doc.get("nav0", None)
        print(f"- window_id={window_id}, t0={t0}, t1={t1}, nav0={nav0}")


def show_window_snapshots(db, window_id: str, limit: int = 10):
    print(f"\n================ LIVE METRICS for window_id={window_id} ================")
    cursor = (
        db["live_metrics"]
        .find({"window_id": window_id})
        .sort("as_of", 1)
    )

    if limit > 0:
        cursor = cursor.limit(limit)

    count = 0
    for doc in cursor:
        as_of = doc.get("as_of")
        nav = doc.get("nav")
        # Possible benchmark fields
        benchmark_nav = doc.get("benchmark_nav", None)
        benchmark_price = doc.get("benchmark_price", None)
        ret_ = doc.get("ret", None)

        print(
            f"as_of={as_of}, nav={nav}, ret={ret_}, "
            f"benchmark_nav={benchmark_nav}, benchmark_price={benchmark_price}"
        )
        count += 1

    if count == 0:
        print("No live_metrics documents found for this window_id.")


def export_window_to_csv(db, window_id: str, out_path: str):
    """
    Export all snapshots for a window to CSV (time series).
    Columns: as_of, nav, ret, benchmark_nav, benchmark_price
    """
    cursor = (
        db["live_metrics"]
        .find({"window_id": window_id})
        .sort("as_of", 1)
    )

    rows = list(cursor)
    if not rows:
        print(f"No live_metrics documents found for window_id={window_id}. Nothing to export.")
        return

    fieldnames = ["as_of", "nav", "ret", "benchmark_nav", "benchmark_price"]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for doc in rows:
            writer.writerow(
                {
                    "as_of": doc.get("as_of"),
                    "nav": doc.get("nav"),
                    "ret": doc.get("ret"),
                    "benchmark_nav": doc.get("benchmark_nav"),
                    "benchmark_price": doc.get("benchmark_price"),
                }
            )

    print(f"Exported {len(rows)} time-series rows for window_id={window_id} to {out_path}")


def plot_window_nav(db, window_id: str):
    """
    Plot NAV (and benchmark if available) for a given window.
    Requires matplotlib to be installed.
    """
    if not HAS_MPL:
        print("matplotlib is not installed. Install it with `pip install matplotlib` to enable plotting.")
        return

    cursor = (
        db["live_metrics"]
        .find({"window_id": window_id})
        .sort("as_of", 1)
    )
    docs = list(cursor)

    if not docs:
        print(f"No live_metrics documents found for window_id={window_id}. Cannot plot.")
        return

    times = [doc.get("as_of") for doc in docs]
    navs = [doc.get("nav") for doc in docs]

    # Try both possible benchmark fields; prefer benchmark_nav if present
    benchmark_navs = None
    if any("benchmark_nav" in d for d in docs):
        benchmark_navs = [doc.get("benchmark_nav") for doc in docs]
    elif any("benchmark_price" in d for d in docs):
        benchmark_navs = [doc.get("benchmark_price") for doc in docs]

    plt.figure()
    plt.plot(times, navs, label="NAV")

    if benchmark_navs is not None:
        plt.plot(times, benchmark_navs, label="Benchmark", linestyle="--")

    plt.xlabel("Time (as_of)")
    plt.ylabel("Value")
    plt.title(f"NAV curve for window {window_id}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def show_latest_predictions(db, limit=5):
    print("\n================ PREDICTIONS (latest) ================")
    cursor = (
        db["AlphaFusionNet_predictions"]
        .find()
        .sort("timestamp", -1)
        .limit(limit)
    )
    for doc in cursor:
        ts = doc.get("timestamp")
        src = doc.get("policy", {}).get("source")
        print(
            f"timestamp={ts}, policy.source={src}, "
            f"final_weights_keys={list(doc.get('final_weights', {}).keys())[:5]}..."
        )


# ------------------------
# Monthly metrics export
# ------------------------
def export_monthly_to_csv(db, out_path: str, year: int | None = None, month: int | None = None):
    """
    Export documents from the monthly metrics collection to CSV.

    If `year` and/or `month` are provided, they are used as filters.
    Uses the generic collection export helper.
    """
    coll = db[MONTHLY_COLLECTION]

    query = {}
    if year is not None:
        query["year"] = year
    if month is not None:
        query["month"] = month

    print(f"\nExporting monthly metrics from collection='{MONTHLY_COLLECTION}' with query={query or '{}'}")
    export_collection_to_csv(coll, out_path=out_path, query=query or {}, limit=None)


# ------------------------
# Main CLI
# ------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Inspect AlphaFusionNet backtest metrics from MongoDB.")
    parser.add_argument(
        "--action",
        choices=[
            "list",
            "window",
            "export",
            "plot",
            "export_monthly",
            "export_live",
            "export_windows",
            "export_predictions",
        ],
        default="list",
        help=(
            "list: list windows; "
            "window: inspect snapshots for a window; "
            "export: export one window's NAV series; "
            "plot: plot NAV; "
            "export_monthly: export monthly_metrics; "
            "export_live: export live_metrics; "
            "export_windows: export windows; "
            "export_predictions: export AlphaFusionNet_predictions."
        ),
    )
    parser.add_argument(
        "--window-id",
        type=str,
        help="Window ID (e.g. 20251118_1430) for 'window', 'export', or 'export_live' actions.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Limit number of records shown for list/window actions, or for collection exports if you want.",
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        type=str,
        default=None,
        help="Output CSV path for 'export', 'export_monthly', 'export_live', 'export_windows', or 'export_predictions' actions.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Year filter for 'export_monthly' (optional).",
    )
    parser.add_argument(
        "--month",
        type=int,
        default=None,
        help="Month filter for 'export_monthly' (1-12, optional).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config()

    # Mongo
    mongo_cfg = cfg["novo_mongo"]
    mongo_client, db = init_mongo_client(mongo_cfg)

    print("\nConnected to MongoDB:", mongo_cfg.get("host"), mongo_cfg.get("port"))
    print("Database:", mongo_cfg.get("database"))

    try:
        if args.action == "list":
            list_windows(db, limit=args.limit)
            show_latest_predictions(db, limit=5)

        elif args.action == "window":
            if not args.window_id:
                raise SystemExit("You must provide --window-id for action=window")
            show_window_snapshots(db, window_id=args.window_id, limit=args.limit)

        elif args.action == "export":
            if not args.window_id:
                raise SystemExit("You must provide --window-id for action=export")
            out_path = args.out_path or f"nav_{args.window_id}.csv"
            export_window_to_csv(db, window_id=args.window_id, out_path=out_path)

        elif args.action == "plot":
            if not args.window_id:
                raise SystemExit("You must provide --window-id for action=plot")
            plot_window_nav(db, window_id=args.window_id)

        elif args.action == "export_monthly":
            # Decide default filename if not provided
            if args.out_path:
                out_path = args.out_path
            elif args.year is not None and args.month is not None:
                out_path = f"monthly_{args.year:04d}_{args.month:02d}.csv"
            else:
                out_path = "monthly_metrics.csv"

            export_monthly_to_csv(
                db,
                out_path=out_path,
                year=args.year,
                month=args.month,
            )

        elif args.action == "export_live":
            # For live_metrics: optional filter by window_id
            coll = db["live_metrics"]
            query = {}
            if args.window_id:
                query["window_id"] = args.window_id

            if args.out_path:
                out_path = args.out_path
            elif args.window_id:
                out_path = f"live_metrics_{args.window_id}.csv"
            else:
                out_path = "live_metrics_all.csv"

            print(f"\nExporting live_metrics with query={query or '{}'}")
            export_collection_to_csv(coll, out_path=out_path, query=query or {}, limit=None)

        elif args.action == "export_windows":
            coll = db["windows"]
            out_path = args.out_path or "windows_all.csv"
            print("\nExporting windows collection")
            # If you want to limit rows, you can pass limit=args.limit; for now export all
            export_collection_to_csv(coll, out_path=out_path, query={}, limit=None)

        elif args.action == "export_predictions":
            coll = db["AlphaFusionNet_predictions"]
            out_path = args.out_path or "predictions_all.csv"
            print("\nExporting AlphaFusionNet_predictions collection")
            export_collection_to_csv(coll, out_path=out_path, query={}, limit=None)

    finally:
        mongo_client.close()


if __name__ == "__main__":
    main()
