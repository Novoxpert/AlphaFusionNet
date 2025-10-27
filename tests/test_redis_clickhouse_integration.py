# test_redis_clickhouse_integration.py
"""
Integration tests for ClickHouse → Redis pipeline in data_ingest_service.py

These tests verify that:
1. OHLCV data can be fetched correctly from ClickHouse (mocked).
2. Fetched OHLCV data is pushed to Redis correctly.
3. The data stored in Redis matches the original OHLCV DataFrame.

All external dependencies are mocked:
- ClickHouse, Redis.

Author: Elham Esmaeilnia(elham.e.shirvani@gmail.com)
Date: 2025-10-15
"""

import pytest
import pandas as pd
import pickle
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock
import fakeredis

from apps.NeuralFusionCore.scripts.data_ingest_service import fetch_ohlcv_range, push_ohlcv_to_redis

# -----------------------------
# Fixtures
# -----------------------------
@pytest.fixture
def redis_client():
    return fakeredis.FakeStrictRedis()

@pytest.fixture
def sample_ohlcv_data():
    now = datetime.now(timezone.utc)
    df = pd.DataFrame({
        "candle_time": [now - timedelta(minutes=3), now],
        "symbol": ["BINANCE:BTCUSDT", "BINANCE:BTCUSDT"],
        "open": [30000, 30100],
        "high": [30100, 30200],
        "low": [29950, 30050],
        "close": [30050, 30150],
        "volume": [1.2, 0.8],
    })
    df['dateTime'] = pd.to_datetime(df['candle_time'], utc=True)
    return df

# -----------------------------
# Tests
# -----------------------------
@patch("apps.NeuralFusionCore.scripts.data_ingest_service.ch_client")
def test_fetch_ohlcv_range(mock_ch_client, sample_ohlcv_data):
    # Mock ClickHouse execute
    mock_ch_client.execute.side_effect = [
        # SELECT query → row tuples of actual data
        [tuple(x) for x in sample_ohlcv_data.drop(columns=["dateTime"]).to_numpy()],
        # DESCRIBE TABLE → list of single-element tuples
        [(c,) for c in sample_ohlcv_data.drop(columns=["dateTime"]).columns]
    ]

    start = datetime.now(timezone.utc) - timedelta(hours=1)
    end = datetime.now(timezone.utc)
    df = fetch_ohlcv_range("BINANCE:BTCUSDT", start, end)
    
    assert not df.empty
    assert "dateTime" in df.columns
    assert df.shape[0] == sample_ohlcv_data.shape[0]