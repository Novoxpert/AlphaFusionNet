"""
Integration tests for Mongo → Redis pipeline in data_ingest_service.py

These tests verify that:
1. News data can be fetched correctly from MongoDB (mocked).
2. Fetched news data is pushed to Redis correctly.
3. The data stored in Redis matches the original news DataFrame.

All external dependencies are mocked:
- MongoDB, Redis.

Author: Elham Esmaeilnia(elham.e.shirvani@gmail.com)
Date: 2025-10-15
"""

import pytest
import pandas as pd
import pickle
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock
import fakeredis

from apps.NeuralFusionCore.scripts.data_ingest_service import fetch_news_range, push_news_to_redis

# -----------------------------
# Fixtures
# -----------------------------
@pytest.fixture
def redis_client():
    return fakeredis.FakeStrictRedis()

@pytest.fixture
def sample_news_data():
    now = datetime.now(timezone.utc)
    df = pd.DataFrame({
        "title": ["News1", "News2"],
        "content": ["Content1", "Content2"],
        "releasedAt": [now - timedelta(hours=1), now],
    })
    df['releasedAt'] = pd.to_datetime(df['releasedAt'], utc=True)
    return df

# -----------------------------
# Tests
# -----------------------------
@patch("apps.NeuralFusionCore.scripts.data_ingest_service.mongo_client")
def test_fetch_news_range(mock_mongo_client, sample_news_data):
    # Mock MongoDB collection find
    mock_col = MagicMock()
    mock_col.find.return_value = sample_news_data.to_dict('records')
    mock_mongo_client.__getitem__.return_value = {"news": mock_col}

    start = datetime.now(timezone.utc) - timedelta(hours=2)
    end = datetime.now(timezone.utc)
    df = fetch_news_range(start, end)
    
    assert not df.empty
    assert "releasedAt" in df.columns
    assert df.shape[0] == sample_news_data.shape[0]

def test_push_news_to_redis(redis_client, sample_news_data):
    # Patch redis_client in the service
    with patch("apps.NeuralFusionCore.scripts.data_ingest_service.redis_client", redis_client):
        push_news_to_redis(sample_news_data)
        key = "news"
        stored = pickle.loads(redis_client.get(key))
        pd.testing.assert_frame_equal(stored, sample_news_data)