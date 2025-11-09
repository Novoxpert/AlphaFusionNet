"""
test_redis_utils.py
====================

Unit tests for redis_utils.py in the lib/ directory.

Purpose:
--------
To verify that Redis-based data retrieval, filtering, and file-swapping
helpers function as intended without needing a real Redis server.  
All Redis interactions are mocked with a lightweight in-memory
`FakeRedis` client for isolation and reliability.

Test Coverage:
--------------
1. Redis Data Fetching:
   - Ensures `get_all_redis_data_version1` correctly loads pickled DataFrames.
   - Ensures `get_all_redis_data` filters OHLCV and news data by time.

2. Atomic File Swapping:
   - Tests that `atomic_model_swap` safely replaces target files
     while backing up and cleaning up as expected.

Mocking Strategy:
-----------------
- `redis_client` is replaced with an in-memory `FakeRedis` class.
- DataFrames are serialized with `pickle` before storage.
- Temporary directories are used for file swap tests.

Author: Elham Esmaeilnia (elham.e.shirvani@gmail.com) 
Date: 2025-10-13  
"""

import os
import pickle
import shutil
import tempfile
from datetime import datetime, timedelta
import pandas as pd
import pytest
from apps.ChronoBridge.lib import redis_utils as RU
from apps.NeuralFusionCore.lib import utils as U


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


# ---------------------- Redis Data Fetch Tests ---------------------- #

def test_get_all_redis_data_version1(monkeypatch):
    """Ensure get_all_redis_data_version1 loads pickled OHLCV and news data."""
    fake_redis = FakeRedis()
    monkeypatch.setattr(RU, "redis_client", fake_redis)

    # Fake OHLCV DataFrame
    df = pd.DataFrame({"dateTime": [datetime.utcnow()], "close": [50000]})
    fake_redis.set("ohlcv:BINANCE:BTCUSDT", pickle.dumps(df))

    # Fake News DataFrame
    news_df = pd.DataFrame({"releasedAt": [datetime.utcnow()], "headline": ["BTC hits 50k"]})
    fake_redis.set("news", pickle.dumps(news_df))

    ohlcv, news = RU.get_all_redis_data_version1(["BINANCE:BTCUSDT"])

    assert "BINANCE:BTCUSDT" in ohlcv
    assert isinstance(ohlcv["BINANCE:BTCUSDT"], pd.DataFrame)
    assert not ohlcv["BINANCE:BTCUSDT"].empty
    assert isinstance(news, pd.DataFrame)
    assert "headline" in news.columns


def test_get_all_redis_data_with_time_filter(monkeypatch):
    """Ensure get_all_redis_data correctly filters data by time range."""
    fake_redis = FakeRedis()
    monkeypatch.setattr(RU, "redis_client", fake_redis)

    now = datetime.utcnow()
    df = pd.DataFrame({
        "dateTime": [now - timedelta(minutes=10), now, now + timedelta(minutes=10)],
        "close": [1, 2, 3]
    })
    fake_redis.set("ohlcv:BINANCE:BTCUSDT", pickle.dumps(df))

    news_df = pd.DataFrame({
        "releasedAt": [now - timedelta(hours=1), now, now + timedelta(hours=1)],
        "headline": ["old", "current", "future"]
    })
    fake_redis.set("news", pickle.dumps(news_df))

    start_time = now - timedelta(minutes=5)
    end_time = now + timedelta(minutes=5)
    ohlcv, news = RU.get_all_redis_data(["BINANCE:BTCUSDT"], start_time, end_time)

    df_filtered = ohlcv["BINANCE:BTCUSDT"]
    assert df_filtered["dateTime"].between(start_time, end_time).all()
    assert news["releasedAt"].between(start_time, end_time).any()


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
    assert not os.path.exists(str(dest) + ".bak")
