"""
Integration tests for Mongo → Redis pipeline in data_ingest_service.py

These tests verify that:
1. News data can be fetched correctly from MongoDB (mocked).
2. Fetched news data is pushed to Redis correctly in chunked form.
3. The data stored in Redis matches the original news DataFrame (per bucket).

All external dependencies are mocked:
- MongoDB, Redis.

Author: Elham Esmaeilnia(elham.e.shirvani@gmail.com)
Date: 2025-10-15 (updated for v1.0.2)
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock
import fakeredis
import apps.ChronoBridge.scripts.data_ingest_service as dis


# -----------------------------
# Fixtures
# -----------------------------
@pytest.fixture
def redis_client():
    return fakeredis.FakeStrictRedis()


@pytest.fixture
def sample_news_data():
    now = datetime.now(timezone.utc)
    df = pd.DataFrame(
        {
            "title": ["News1", "News2"],
            "content": ["Content1", "Content2"],
            "releasedAt": [now - timedelta(hours=1), now],
        }
    )
    df["releasedAt"] = pd.to_datetime(df["releasedAt"], utc=True)
    return df


# -----------------------------
# Tests
# -----------------------------
@patch("apps.ChronoBridge.scripts.data_ingest_service.mongo_client")
def test_fetch_news_range(mock_mongo_client, sample_news_data):
    """
    Ensure fetch_news_range builds a DataFrame from Mongo docs returned by a cursor.
    """

    # Prepare fake docs and a fake cursor
    docs = sample_news_data.to_dict("records")

    fake_cursor = MagicMock()
    # Chainable cursor: .sort(...).batch_size(...) → returns same cursor
    fake_cursor.sort.return_value = fake_cursor
    fake_cursor.batch_size.return_value = fake_cursor
    # Iteration over the cursor yields our docs
    fake_cursor.__iter__.return_value = iter(docs)

    # Mock collection
    mock_col = MagicMock()
    mock_col.find.return_value = fake_cursor

    # mongo_client[MO.MONGO_DB][MO.MONGO_COLLECTION] → mock_db[...]
    mock_db = MagicMock()
    mock_db.__getitem__.return_value = mock_col
    mock_mongo_client.__getitem__.return_value = mock_db

    start = datetime.now(timezone.utc) - timedelta(hours=2)
    end = datetime.now(timezone.utc)

    df = dis.fetch_news_range(start, end)

    assert not df.empty
    assert "releasedAt" in df.columns
    assert df.shape[0] == sample_news_data.shape[0]


def test_push_news_chunked_to_redis(redis_client, sample_news_data):
    """
    Ensure push_news_chunked writes compressed DataFrames to Redis with the expected key pattern.
    """

    # Patch redis_client global in the service module
    with patch(
        "apps.ChronoBridge.scripts.data_ingest_service.redis_client", redis_client
    ):
        # Group by day, like the real code (freq="D")
        dis.push_news_chunked(sample_news_data, freq="D")

        # Expect keys like "news:YYYY-MM-DD"
        keys = list(redis_client.scan_iter("news:*"))
        assert len(keys) == 1  # both rows are on the same day in this fixture

        key = keys[0]
        stored_bytes = redis_client.get(key)
        assert stored_bytes is not None

        # Use the module's own loader to decompress + unpickle
        stored_df = dis._loads_df_compressed(stored_bytes)

        # Because all rows are same-day, the chunk should equal the original DataFrame
        # (ordering of columns might differ, so use check_like=True)
        pd.testing.assert_frame_equal(
            stored_df.reset_index(drop=True),
            sample_news_data.reset_index(drop=True),
            check_like=True,
        )
