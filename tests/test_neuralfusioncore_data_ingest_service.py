#!/usr/bin/env python3
"""
test_neuralfusioncore_data_ingest_service.py
---------------------------
Unit tests for data_ingest_service.py.
Mocks ClickHouse, MongoDB, and Redis clients to test fetch & push functions.
Author: Elham Esmaeilnia(elham.e.shirvani@gmail.com)
Date: 2025 Oct 13
Version: 1.0.0
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta, timezone
import pickle
from apps.NeuralFusionCore.scripts import data_ingest_service as DIS

# ---------------------------------------------------------------------
# ✅ Test fetch_ohlcv_range
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
# ✅ Test fetch_news_range
# ---------------------------------------------------------------------
def test_fetch_news_range(monkeypatch):
    # Mock Mongo collection
    class MockCol:
        def find(self, query):
            return [{"releasedAt": datetime(2025, 10, 13, 0, 0, tzinfo=timezone.utc),
                     "title": "Fake News"}]

    class MockMongoClient:
        def __getitem__(self, name):
            return {DIS.MO.MONGO_COLLECTION: MockCol()}

    monkeypatch.setattr(DIS, "mongo_client", MockMongoClient())

    start = datetime(2025, 10, 13, 0, 0, tzinfo=timezone.utc)
    end = datetime(2025, 10, 13, 1, 0, tzinfo=timezone.utc)
    df = DIS.fetch_news_range(start, end)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "releasedAt" in df.columns

# ---------------------------------------------------------------------
# ✅ Test push_ohlcv_to_redis
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
# ✅ Test push_news_to_redis
# ---------------------------------------------------------------------
def test_push_news_to_redis(monkeypatch):
    pushed_data = {}
    monkeypatch.setattr(DIS.redis_client, "set", lambda k, v: pushed_data.setdefault(k, pickle.loads(v)))
    df = pd.DataFrame({"title": ["news1", "news2"]})
    DIS.push_news_to_redis(df)
    assert "news" in pushed_data
    pd.testing.assert_frame_equal(pushed_data["news"], df)
