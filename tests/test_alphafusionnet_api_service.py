"""
test_alphafusionnet_api_service.py
-----------------------------
Unit tests for the AlphaFusionNet FastAPI service (alphafusionnet_api_service.py).

This test suite uses FastAPI's TestClient and a local MongoDB test collection
to verify that the API endpoints behave as expected. It covers:

1. /health            - Returns service health status.
2. /latest_alphafusionnet   - Returns the most recent alphafusionnet portfolio prediction.
3. /alphafusionnet_history  - Returns historical alphafusionnet predictions within a time range.
4. /alphafusionnet_history (empty) - Returns empty list when no documents match the query.

Setup:
- Requires a running MongoDB instance at mongodb://127.0.0.1:27017/
- Uses database: db_portfolio
- Uses collection: AlphaFusionNet_predictions

Author: Elham Esmaeilnia (elham.e.shirvani@gmail.com)
Date: 2025 Oct 25
Version: 1.0.0
"""

import unittest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
import pymongo
import json

# Import your FastAPI app
from scripts.alphafusionnet_api_service import app

# -----------------------------
# Test Mongo setup
# -----------------------------
MONGO_URI = "mongodb://127.0.0.1:27017/"
DB_NAME = "db_portfolio"
COLLECTION_NAME = "AlphaFusionNet_predictions"

client = pymongo.MongoClient(MONGO_URI)
db = client[DB_NAME]
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
        # Clean up after tests
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
        response = self.client.get("/latest_alphafusionnet")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("policy", data)
        self.assertIn("final_weights", data)
        self.assertEqual(data["notes"], "test document")

    def test_alphafusionnet_history(self):
        start = (self.now - timedelta(minutes=1)).isoformat()
        end = (self.now + timedelta(minutes=1)).isoformat()
        response = self.client.get(f"/alphafusionnet_history?start={start}&end={end}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertGreaterEqual(len(data), 1)
        self.assertIn("policy", data[0])
        self.assertIn("final_weights", data[0])

    def test_alphafusionnet_history_empty(self):
        # Query outside the timestamp range
        start = (self.now - timedelta(days=2)).isoformat()
        end = (self.now - timedelta(days=1)).isoformat()
        response = self.client.get(f"/alphafusionnet_history?start={start}&end={end}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data, [])

if __name__ == "__main__":
    unittest.main()
