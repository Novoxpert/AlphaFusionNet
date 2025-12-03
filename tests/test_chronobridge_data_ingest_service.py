#!/usr/bin/env python3
"""
test_chronobridge_data_ingest_service.py
---------------------------
Unit tests for data_ingest_service.py.
Mocks ClickHouse, MongoDB, and Redis clients to test fetch & push functions.

Author: Elham Esmaeilnia(elham.e.shirvani@gmail.com)
Date: 2025 Oct 13
Version: 1.0.2
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta, timezone
import pickle

from apps.ChronoBridge.scripts import data_ingest_service as DIS


# ---------------------------------------------------------------------
# Test fetch_ohlcv_range
# ---------------------------------------------------------------------
def test_fetch_ohlcv_range(monkeypatch):
    """Unit test for fetch_ohlcv_range — should return DataFrame with fake data."""

    def mock_execute(query: str):
        query_lower = query.lower()

        # DESCRIBE TABLE query returns column definitions
        if "describe table" in query_lower:
            return [
                ("symbol", "String"),
                ("candle_time", "DateTime"),
                ("open", "Float64"),
                ("high", "Float64"),
                ("low", "Float64"),
                ("close", "Float64"),
                ("volume", "Float64"),
            ]

        # SELECT query returns fake OHLCV row
        if "select" in query_lower and "from" in query_lower:
            return [
                (
                    "BINANCE:BTCUSDT",
                    "2025-10-13 00:00:00",
                    50000,
                    51000,
                    49500,
                    50500,
                    100,
                )
            ]

        return []

    # Patch ClickHouse client
    monkeypatch.setattr(DIS.ch_client, "execute", mock_execute)

    # Patch the market resampler (to just return the same df)
    monkeypatch.setattr(DIS.M, "resample_to_3m", lambda df, cols: df)

    # Define a sample range
    start = datetime(2025, 10, 13, 0, 0, tzinfo=timezone.utc)
    end = datetime(2025, 10, 13, 1, 0, tzinfo=timezone.utc)

    df = DIS.fetch_ohlcv_range("BINANCE:BTCUSDT", start, end)

    # Assertions
    assert isinstance(df, pd.DataFrame)
    assert not df.empty, "DataFrame should not be empty when mock returns one row"
    assert "symbol" in df.columns
    assert "close" in df.columns
    assert "dateTime" in df.columns


# ---------------------------------------------------------------------
# Test fetch_news_range (updated for cursor + sort + batch_size)
# ---------------------------------------------------------------------
def test_fetch_news_range(monkeypatch):
    now = datetime(2025, 10, 13, 0, 0, tzinfo=timezone.utc)
    docs = [
        {
            "releasedAt": now,
            "title": "Fake News",
            "content": "Some content",
            "assets": [],
        }
    ]

    class MockCursor:
        def __init__(self, docs):
            self._docs = docs

        def sort(self, *args, **kwargs):
            # Return self to allow chaining
            return self

        def batch_size(self, *args, **kwargs):
            # Return self to allow chaining
            return self

        def __iter__(self):
            return iter(self._docs)

    class MockCol:
        def find(self, query, projection=None):
            # Ignore query/projection for this simple test
            return MockCursor(docs)

    class MockDB:
        def __getitem__(self, name):
            
            return MockCol()

    class MockMongoClient:
        def __getitem__(self, name):
            
            return MockDB()

    monkeypatch.setattr(DIS, "mongo_client", MockMongoClient())

    start = now - timedelta(hours=1)
    end = now + timedelta(hours=1)

    df = DIS.fetch_news_range(start, end)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "releasedAt" in df.columns
    assert df.iloc[0]["title"] == "Fake News"


# ---------------------------------------------------------------------
# Test push_ohlcv_to_redis
# ---------------------------------------------------------------------
def test_push_ohlcv_to_redis(monkeypatch):
    pushed_data = {}

    def mock_set(key, value):
        pushed_data[key] = pickle.loads(value)
        return True

    monkeypatch.setattr(DIS.redis_client, "set", mock_set)

    df = pd.DataFrame({"a": [1, 2, 3]})
    DIS.push_ohlcv_to_redis("TEST", df)
    assert "ohlcv:TEST" in pushed_data
    pd.testing.assert_frame_equal(pushed_data["ohlcv:TEST"], df)


# ---------------------------------------------------------------------
# Test push_news_chunked (replacing old push_news_to_redis)
# ---------------------------------------------------------------------
def test_push_news_chunked(monkeypatch):
    """
    Tests that push_news_chunked groups by day and writes compressed payloads
    using Redis pipeline with keys like 'news:YYYY-MM-DD'.
    """

    # Fake pipeline object to capture set() calls
    class FakePipeline:
        def __init__(self):
            self.data = {}

        def set(self, key, value):
            self.data[key] = value
            return True

        def execute(self):
            # No-op for test
            return True

    class FakeRedis:
        def __init__(self):
            self.pipe = FakePipeline()

        def pipeline(self, transaction=False):
            return self.pipe

    fake_redis = FakeRedis()
    monkeypatch.setattr(DIS, "redis_client", fake_redis)

    # Two news rows on the same day
    now = datetime.now(timezone.utc)
    df = pd.DataFrame(
        {
            "title": ["news1", "news2"],
            "content": ["c1", "c2"],
            "releasedAt": [now - timedelta(hours=1), now],
        }
    )
    df["releasedAt"] = pd.to_datetime(df["releasedAt"], utc=True)

    DIS.push_news_chunked(df, freq="D")

    # We expect exactly one key (same day)
    assert len(fake_redis.pipe.data) == 1
    key = list(fake_redis.pipe.data.keys())[0]
    assert key.startswith("news:")

    # Decode compressed payload using the module's helper
    stored_bytes = fake_redis.pipe.data[key]
    stored_df = DIS._loads_df_compressed(stored_bytes)

    # Compare content ignoring index
    pd.testing.assert_frame_equal(
        stored_df.reset_index(drop=True),
        df.reset_index(drop=True),
        check_like=True,
    )
