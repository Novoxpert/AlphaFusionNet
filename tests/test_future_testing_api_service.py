#!/usr/bin/env python3
"""
Unit tests for future_testing_api_service.py
Author: Elham Esmaeilnia(elham.e.shirvani@gmail.com)
Date: 2025-11-04
"""

import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import pandas as pd
from scripts.future_testing_api_service import app

class TestFutureTestingAPIService(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch("scripts.future_testing_api_service.future_testing_col")
    def test_get_latest_future_testing_success(self, mock_col):
        # Arrange: mock MongoDB return document
        dummy_doc = {
            "timestamp": pd.Timestamp("2025-01-01 12:00:00"),
            "features": [{"f1": 0.1, "f2": 0.2}],
            "weights": {"AAPL": 1.0, "TSLA": 0.5},
            "created_at": pd.Timestamp("2025-01-01 12:01:00")
        }
        mock_col.find_one.return_value_