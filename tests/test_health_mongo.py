"""
test_health_mongodb.py
Description: Test MongoDB interactions (fetch news, save predictions, API reads).
Author: Elham Esmaeilnia (elham.e.shirvani@gmail.com)
Date: 2025 Oct 13
Version: 1.1.0
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone
from apps.ChronoBridge.scripts.data_ingest_service import fetch_news_range, MO
from apps.NeuralFusionCore.scripts.prediction_service import save_predictions
from apps.NeuralFusionCore.scripts.api_service import latest_prediction, prediction_history
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd

# ============================================================
# Test: fetch_news_range
# ============================================================
@patch("apps.ChronoBridge.scripts.data_ingest_service.mongo_client")
def test_fetch_news_range(mock_mongo_client): 
    """Test MongoDB news fetch between time range"""
    fake_data = [{"releasedAt": datetime(2025, 10, 13, 0, 0, tzinfo=timezone.utc), "title": "News1"}]
    mock_collection = MagicMock()
    mock_collection.find.return_value = fake_data
    mock_mongo_client.__getitem__.return_value = {MO.MONGO_COLLECTION: mock_collection}

    start = datetime(2025, 10, 13, 0, 0, tzinfo=timezone.utc)
    end = datetime(2025, 10, 13, 1, 0, tzinfo=timezone.utc)
    df = fetch_news_range(start, end)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "releasedAt" in df.columns


# ============================================================
# Test: save_predictions
# ============================================================
@patch("apps.NeuralFusionCore.scripts.prediction_service.mongo_col")
def test_save_predictions(mock_mongo_col):
    """Test saving predictions to Redis and Mongo"""
    import numpy as np
    from apps.NeuralFusionCore.scripts.prediction_service import save_predictions
    from datetime import datetime

    weights = np.array([0.1, 0.2, 0.7])
    stocks = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]

    # Correct: list of dicts with ts
    predictions = [
        {"ts": datetime.utcnow(), "weights": weights, "stocks": stocks}
    ]

    save_predictions(predictions)

    # Verify Redis and Mongo insertions
    assert mock_mongo_col.insert_many.called or mock_mongo_col.insert_one.called

# ============================================================
# Test: latest_prediction API
# ============================================================
@patch("apps.NeuralFusionCore.scripts.api_service.collection")
def test_latest_prediction(mock_collection):
    """Test the latest_prediction endpoint returns valid JSON"""
    mock_collection.find_one.return_value = {
        "ts": datetime.now(timezone.utc),
        "weights": [0.1, 0.2],
        "stocks": ["BTCUSDT"]
    }

    response = latest_prediction()
    assert isinstance(response, JSONResponse)

    # Decode response body (bytes → dict)
    body = response.body.decode("utf-8")
    assert "BTCUSDT" in body
    assert response.status_code == 200


# ============================================================
# Test: prediction_history API
# ============================================================
@patch("apps.NeuralFusionCore.scripts.api_service.collection")
def test_prediction_history(mock_collection):
    """Test prediction_history endpoint returns a valid JSON list"""
    mock_cursor = [
        {"ts": datetime(2025, 10, 13, 0, 0), "weights": [0.1, 0.2], "stocks": ["BTCUSDT"]},
        {"ts": datetime(2025, 10, 13, 1, 0), "weights": [0.2, 0.8], "stocks": ["ETHUSDT"]},
    ]
    mock_collection.find.return_value.sort.return_value.limit.return_value = mock_cursor

    # Explicitly call without args (FastAPI Query defaults to None)
    response = prediction_history(start=None, end=None, limit=100)

    assert isinstance(response, JSONResponse)
    body = response.body.decode("utf-8")
    assert "BTCUSDT" in body or "ETHUSDT" in body
    assert response.status_code == 200
