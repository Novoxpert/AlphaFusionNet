#!/usr/bin/env python3
"""
test_api_service.py
-------------------
Unit and integration tests for the FastAPI service (api_service.py).
Validates API endpoints for latest prediction, historical data, and health checks.
Author: Elham Esmaeilnia(elham.e.shirvani@gmail.com)
Date: 2025 Oct 13
Version: 1.2.0
"""

import pytest
from datetime import datetime, timedelta, timezone
from fastapi.testclient import TestClient
from apps.NeuralFusionCore.scripts import api_service 
from urllib.parse import quote 

client = TestClient(api_service.app)

# ---------------------------------------------------------------------
# ✅ Health Check Endpoint
# ---------------------------------------------------------------------
def test_health_check():
    """Verify /health returns status ok"""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "time" in body

# ---------------------------------------------------------------------
# ✅ Latest Prediction Endpoint
# ---------------------------------------------------------------------
def test_latest_prediction_empty(monkeypatch):
    """Verify /latest_prediction returns 404 when no data exists"""
    monkeypatch.setattr(api_service.collection, "find_one", lambda sort=None: None)
    response = client.get("/latest_prediction")
    assert response.status_code == 404
    assert response.json() == {"error": "No prediction found"}

def test_latest_prediction_valid(monkeypatch):
    """Verify /latest_prediction returns a valid document when available"""
    mock_doc = {
        "_id": "mockid123",
        "ts": datetime.now(timezone.utc),
        "weights": [0.1, 0.2, -0.3],
        "stocks": ["AAPL", "GOOG", "TSLA"]
    }
    monkeypatch.setattr(api_service.collection, "find_one", lambda sort=None: mock_doc)

    response = client.get("/latest_prediction")
    assert response.status_code == 200
    data = response.json()
    assert "weights" in data
    assert "stocks" in data
    assert isinstance(data["stocks"], list)

# ---------------------------------------------------------------------
# ✅ Prediction History Endpoint
# ---------------------------------------------------------------------
def test_prediction_history(monkeypatch):
    now = datetime.now(timezone.utc)
    mock_docs = [
        {"_id": "mock1", "ts": now - timedelta(hours=2), "weights": [[0.1, 0.2]], "stocks": ["AAPL", "MSFT"]},
        {"_id": "mock2", "ts": now - timedelta(hours=1), "weights": [[0.3, -0.1]], "stocks": ["TSLA", "AMZN"]},
    ]

    class MockCursor:
        def sort(self, key, direction):
            return self
        def limit(self, n):
            return self
        def __iter__(self):
            return iter(mock_docs)

    # Patch collection.find to return mock docs
    monkeypatch.setattr(api_service.collection, "find", lambda query: MockCursor())

    # Use proper ISO strings with +00:00
    start = (now - timedelta(hours=3)).isoformat(timespec="microseconds")
    end = now.isoformat(timespec="microseconds")

    # Ensure tzinfo is explicit
    if start[-6:] != "+00:00":
        start += "+00:00"
    if end[-6:] != "+00:00":
        end += "+00:00"
    
    # Encode timestamps
    start_encoded = quote(start)
    end_encoded = quote(end)
    response = client.get(f"/prediction_history?start={start_encoded}&end={end_encoded}&limit=10")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2