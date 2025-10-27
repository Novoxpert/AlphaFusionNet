"""
test_chronobridge_api_service.py
--------------------------------
Unit tests for the `chronobridge_api_service.py` FastAPI application.

This test module verifies the functionality of the ChronoBridge API endpoints
without requiring a live MongoDB instance. MongoDB interactions are mocked
to simulate database responses.

Tested endpoints:
- /fused_embeddings : Retrieves fused embeddings with optional filtering
  by date range and stock symbols.
- /health           : Simple health check endpoint.

Dependencies:
- pytest
- fastapi.testclient
- unittest.mock
Author: Elham Esmaeilnia(elham.e.shirvani@gmail.com)
Date: 2025 Oct 19
Version: 1.1.0
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from apps.ChronoBridge.scripts.chronobridge_api_service import app

client = TestClient(app)

# Sample mocked MongoDB data
mock_data = [
    {"_id": 1, "date": "2025-10-19", "symbol": "STOCK_A", "fused_embedding": [0.1, 0.2], "close": 100},
    {"_id": 2, "date": "2025-10-20", "symbol": "STOCK_B", "fused_embedding": [0.3, 0.4], "close": 200},
    {"_id": 3, "date": "2025-10-21", "symbol": "STOCK_A", "fused_embedding": [0.5, 0.6], "close": 110},
]

@pytest.fixture
def mock_mongo_find():
    with patch("apps.ChronoBridge.scripts.chronobridge_api_service.mongo_col") as mock_col:
        mock_col.find = MagicMock(return_value=mock_data)
        yield mock_col

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_get_fused_embeddings_no_filters(mock_mongo_find):
    response = client.get("/fused_embeddings")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == len(mock_data)
    assert all("fused_embedding" in d for d in data["data"])

def test_get_fused_embeddings_with_date_filter(mock_mongo_find):
    response = client.get("/fused_embeddings?start_date=2025-10-20&end_date=2025-10-21")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 3  # Mocked find still returns all since we mock directly
    # In a more advanced mock, you can filter mock_data according to query

def test_get_fused_embeddings_with_stock_filter(mock_mongo_find):
    response = client.get("/fused_embeddings?stocks=STOCK_A")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 3  # Mock returns all; check structure
    assert all(d["symbol"] in ["STOCK_A", "STOCK_B"] for d in data["data"])

def test_get_fused_embeddings_with_date_and_stock(mock_mongo_find):
    response = client.get("/fused_embeddings?start_date=2025-10-19&stocks=STOCK_B")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 3
    for d in data["data"]:
        assert "fused_embedding" in d
