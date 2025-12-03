"""
test_redis_utils.py
====================

Unit tests for redis_utils.py in the lib/ directory.

Purpose:
--------
To verify that Redis-based data retrieval, filtering, and database
helpers function as intended without needing a real Redis server.  
All Redis interactions are mocked with a lightweight in-memory
`FakeRedis` client for isolation and reliability.

Test Coverage:
--------------
1. Redis Data Fetching:
   - Ensures `load_ohlcv_from_redis` correctly loads pickled DataFrames.
   - Ensures `load_ohlcv_from_redis` filters OHLCV data by time.
   - Ensures `load_news_range_from_redis` loads compressed, chunked
     news and filters it by time.

2. Atomic File Swapping:
   - Tests that `atomic_model_swap` safely replaces target files
     while backing up and cleaning up as expected.

Mocking Strategy:
-----------------
- `redis_client` is replaced with an in-memory `FakeRedis` class.
- OHLCV DataFrames are serialized with `pickle` (plain).
- News DataFrames are serialized with `pickle` + `zlib.compress`
  to match the production chunked storage format.
- Temporary directories are used for file swap tests.

Author: Elham Esmaeilnia (elham.e.shirvani@gmail.com) 
Date: 2025-10-13  
Version: v1.0.1
"""

import os
import pickle
import zlib
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from apps.ChronoBridge.lib import redis_utils as RU
from apps.NeuralFusionCore.lib import utils as U


class FakePipeline:
    """Mimic a Redis pipeline for GET operations."""
    def __init__(self, client):
        self.client = client
        self.keys = []

    def get(self, key):
        self.keys.append(key)
        return self  # allow chaining if needed

    def execute(self):
        return [self.client.get(k) for k in self.keys]


class FakeRedis:
    """In-memory fake Redis client for unit testing."""
    def __init__(self):
        self.storage = {}

    def get(self, key):
        return self.storage.get(key)

    def set(self, key, value):
        self.storage[key] = value
        return True

    def delete(self, key):
        return self.storage.pop(key, None)

    def pipeline(self, *args, **kwargs):
        return FakePipeline(self)

    def flushdb(self):
        self.storage.clear()
        return True


# ---------------------- Redis Data Fetch Tests ---------------------- #

def test_load_ohlcv_from_redis_basic(monkeypatch):
    """Ensure load_ohlcv_from_redis loads pickled OHLCV DataFrames."""
    fake_redis = FakeRedis()
    monkeypatch.setattr(RU, "redis_client", fake_redis)

    now = datetime.now(timezone.utc)
    df = pd.DataFrame({"dateTime": [now], "close": [50000.0]})

    # Store plain pickled DataFrame under the expected key
    fake_redis.set("ohlcv:BINANCE:BTCUSDT", pickle.dumps(df))

    ohlcv = RU.load_ohlcv_from_redis(["BINANCE:BTCUSDT"])

    assert "BINANCE:BTCUSDT" in ohlcv
    out_df = ohlcv["BINANCE:BTCUSDT"]
    assert isinstance(out_df, pd.DataFrame)
    assert not out_df.empty
    assert "dateTime" in out_df.columns
    assert "close" in out_df.columns


def test_load_ohlcv_from_redis_with_time_filter(monkeypatch):
    """Ensure load_ohlcv_from_redis correctly filters OHLCV data by time range."""
    fake_redis = FakeRedis()
    monkeypatch.setattr(RU, "redis_client", fake_redis)

    now = datetime.now(timezone.utc)
    df = pd.DataFrame({
        "dateTime": [
            now - timedelta(minutes=10),
            now,
            now + timedelta(minutes=10),
        ],
        "close": [1.0, 2.0, 3.0],
    })

    fake_redis.set("ohlcv:BINANCE:BTCUSDT", pickle.dumps(df))

    start_time = now - timedelta(minutes=5)
    end_time = now + timedelta(minutes=5)
    ohlcv = RU.load_ohlcv_from_redis(["BINANCE:BTCUSDT"], start_time, end_time)

    assert "BINANCE:BTCUSDT" in ohlcv
    df_filtered = ohlcv["BINANCE:BTCUSDT"]
    assert not df_filtered.empty
    # All rows should be within [start_time, end_time]
    assert df_filtered["dateTime"].between(
        pd.to_datetime(start_time, utc=True),
        pd.to_datetime(end_time, utc=True),
    ).all()


def test_load_news_range_from_redis(monkeypatch):
    """Ensure load_news_range_from_redis loads chunked, compressed news and filters by time."""
    fake_redis = FakeRedis()
    monkeypatch.setattr(RU, "redis_client", fake_redis)

    # Define a single-day range
    day = datetime(2025, 10, 13, 0, 0, tzinfo=timezone.utc)
    start_time = day
    end_time = day + timedelta(hours=23, minutes=59)

    # Build a news DataFrame with a few timestamps (one inside, one outside)
    news_df = pd.DataFrame({
        "releasedAt": [
            day - timedelta(hours=1),  # before window
            day + timedelta(hours=1),  # inside window
            day + timedelta(days=1),   # after window
        ],
        "content": ["old", "inside", "future"],
        "news_count": [1, 1, 1],
    })

    # Serialize with pickle + zlib to match production chunk format
    payload = zlib.compress(
        pickle.dumps(news_df, protocol=pickle.HIGHEST_PROTOCOL),
        level=6,
    )

    # Store under the correct chunk key
    key = f"news:{day.strftime('%Y-%m-%d')}"
    fake_redis.set(key, payload)

    df_loaded = RU.load_news_range_from_redis(start_time, end_time)

    assert isinstance(df_loaded, pd.DataFrame)
    assert not df_loaded.empty
    # All loaded rows should be within [start_time, end_time]
    assert df_loaded["releasedAt"].between(
        pd.to_datetime(start_time, utc=True),
        pd.to_datetime(end_time, utc=True),
    ).all()


# ---------------------- Atomic Swap Test ---------------------- #

def test_atomic_model_swap(tmp_path):
    """Test atomic_model_swap safely replaces destination file."""
    src = tmp_path / "model_temp.pkl"
    dest = tmp_path / "model_final.pkl"

    src.write_text("new model content")
    dest.write_text("old model content")

    U.atomic_model_swap(str(src), str(dest))

    assert dest.exists()
    assert dest.read_text() == "new model content"
    # backup should be cleaned up
    assert not os.path.exists(str(dest) + ".bak")
