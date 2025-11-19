"""
db_utils.py

Database client initialization utilities for ClickHouse and MongoDB.

This module provides helper functions for creating database client
instances based on configuration parameters:

- `init_clickhouse_client(...)`  
  Safely initializes and returns a ClickHouse client using
  `clickhouse_driver`. Raises a `RuntimeError` if the driver is not
  installed.

- `init_mongo_client(cfg)`  
  Builds a MongoDB connection URI from a configuration mapping and
  returns both the `MongoClient` instance and the selected database.

These helpers are intended to centralize client construction logic,
reduce duplication, and ensure consistent initialization across the
application.

Author: Elham Esmaeilnia (elham.e.shirvani@gmail.com)
Date: 2025-11-17
Version: 1.0.0
"""

from typing import Tuple
from pymongo import MongoClient
try:
    from clickhouse_driver import Client as CHClient
except Exception:  # pragma: no cover - import error handled at runtime
    CHClient = None

# -------------------------------
# Configuration helpers
# -------------------------------

def init_clickhouse_client(host: str = "localhost",
                           port: int = 9000,
                           user: str = None,
                           password: str = None,
                           database: str = None,
                           ) -> "CHClient":
    if CHClient is None:
        raise RuntimeError("clickhouse_driver is not installed in the environment")
    kwargs = {k: v for k, v in dict(host=host, port=port, user=user, password=password, database=database).items() if v is not None}
    return CHClient(**kwargs)


def init_mongo_client(cfg):
    uri = (
        f"mongodb://{cfg['user']}:{cfg['password']}@"
        f"{cfg['host']}:{cfg['port']}/{cfg['database']}?authSource={cfg['authSource']}"
    )
    client = MongoClient(uri)
    db = client[cfg["database"]]
   
    return client, db
