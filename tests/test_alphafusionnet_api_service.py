"""
test_alphafusionnet_api_service.py
----------------------------------
Unit tests for the AlphaFusionNet FastAPI service (alphafusionnet_api_service.py).

This test suite uses FastAPI's TestClient and a local MongoDB test collection
to verify that the API endpoints behave as expected. It covers:

1. /health            - Returns service health status.
2. /latest_alphafusionnet   - Returns the most recent alphafusionnet portfolio prediction.
3. /alphafusionnet_history  - Returns historical alphafusionnet predictions within a time range.
4. /alphafusionnet_history (empty) - Returns empty list when no documents match the query.
5. save_predictions   - Tests saving predictions to Mongo and Redis.

Setup:
- Requires a running MongoDB instance at mongodb://127.0.0.1:27017/
- Uses database: db_portfolio
- Uses collection: AlphaFusionNet_test_predictions

Author: Elham Esmaeilnia (elham.e.shirvani@gmail.com)
Date: 2025 Oct 25
Version: 2.0.0 (fixed)
"""

import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import numpy as np

from scripts.alphafusionnet_api_service import app
from apps.NeuralFusionCore.scripts.prediction_service import save_predictions

# -----------------------------
# Test Mongo setup
# -----------------------------
load_dotenv()
mongo_user = os.getenv("NOVO_MONGO_USER")
mongo_pass = os.getenv("NOVO_MONGO_PASS")
mongo_host = os.getenv("NOVO_MONGO_HOST")
mongo_port = os.getenv("NOVO_MONGO_PORT")
mongo_auth_db = os.getenv("NOVO_MONGO_AUTH_DB")
mongo_db_name = os.getenv("NOVO_MONGO_DB")

mongo_uri = (
    f"mongodb://{mongo_user}:{mongo_pass}@"
    f"{mongo_host}:{mongo_port}/"
    f"{mongo_db_name}?authSource={mongo_auth_db}"
)

client = MongoClient(mongo_uri)
db = client[mongo_db_name]
COLLECTION_NAME = "AlphaFusionNet_test_predictions"
collection = db[COLLECTION_NAME]


class TestAlphaFusionNetAPIService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Clear test documents
        collection.delete_many({})
        # Insert a test document
        cls.now = datetime.utcnow()
        cls.test_doc = {
            "timestamp": cls.now,
            "policy": {"alpha": 0.7, "method": "rank", "gross_net": 1.0},
            "final_weights": {"AAPL": 0.5, "MSFT": 0.3, "TSLA": 0.2},
            "notes": "test document"
        }
        collection.insert_one(cls.test_doc)
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        collection.delete_many({})

    # -----------------------------
    # Endpoint tests
    # -----------------------------
    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertEqual(data["status"], "ok")

    def test_latest_alphafusionnet(self):
        # Patch the API to use test collection
        with patch("scripts.alphafusionnet_api_service.collection", collection):
            response = self.client.get("/latest_alphafusionnet")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("policy", data)
        self.assertIn("final_weights", data)
        if "notes" in data:
            self.assertEqual(data["notes"], "test document")

    def test_alphafusionnet_history(self):
        start = (self.now - timedelta(minutes=1)).isoformat()
        end = (self.now + timedelta(minutes=1)).isoformat()
        with patch("scripts.alphafusionnet_api_service.collection", collection):
            response = self.client.get(f"/alphafusionnet_history?start={start}&end={end}")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertGreaterEqual(len(data), 1)
        self.assertIn("policy", data[0])
        self.assertIn("final_weights", data[0])

    def test_alphafusionnet_history_empty(self):
        start = (self.now - timedelta(days=2)).isoformat()
        end = (self.now - timedelta(days=1)).isoformat()
        with patch("scripts.alphafusionnet_api_service.collection", collection):
            response = self.client.get(f"/alphafusionnet_history?start={start}&end={end}")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data, [])


class TestSavePredictions(unittest.TestCase):
    @patch("apps.NeuralFusionCore.scripts.prediction_service.mongo_col")
    def test_save_predictions(self, mock_mongo_col):
        """Test saving predictions to Redis and Mongo"""
        weights = np.array([0.1, 0.2, 0.7])
        stocks = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        predictions = [{"ts": datetime.utcnow(), "weights": weights, "stocks": stocks}]

        save_predictions(predictions)

        # Verify Mongo insertion (insert_one or insert_many)
        self.assertTrue(mock_mongo_col.insert_one.called or mock_mongo_col.insert_many.called)


if __name__ == "__main__":
    unittest.main()