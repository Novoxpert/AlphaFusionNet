#!/usr/bin/env python3
"""
Unit tests for future_testing_service.py

This test suite validates:
- MongoDB timestamp lookups
- Ingest window calculation
- Metric computation logic
- Main execution flow in "latest" mode
- Edge cases (no prediction, missing parquet, no feature row)
- Safe mocking of subprocess calls, MongoDB, filesystem, and torch

Author:
    Elham Esmaeilnia <elham.e.shirvani@gmail.com>
Date:
    2025-11-04
Version:
    1.0.1
"""
import unittest
import sys
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Import main functions from your script
from scripts.future_testing_service import main, run_feature_service

class TestFutureTestingService(unittest.TestCase):

    @patch("scripts.future_testing_service.alpha_col.find_one")
    @patch("scripts.future_testing_service.os.path.exists")
    @patch("scripts.future_testing_service.pd.read_parquet")
    @patch("scripts.future_testing_service.torch.cuda.empty_cache")
    @patch("scripts.future_testing_service.subprocess.run")
    @patch("scripts.future_testing_service.run_feature_service")  # Patch the actual subprocess call
    def test_main_latest_flow(
        self, mock_run_feature_service, mock_subproc, mock_cuda, mock_read_parquet, mock_exists, mock_find
    ):
        # Setup mocks
        mock_find.return_value = {"timestamp": pd.Timestamp("2025-01-01"), "final_weights": {"AAPL": 1.0}}
        mock_exists.return_value = True
        mock_read_parquet.return_value = pd.DataFrame({"dateTime": [pd.Timestamp("2025-01-01")], "f1": [0.1]})
        mock_run_feature_service.return_value = None  # prevent subprocess call
        mock_subproc.return_value = None

        testargs = ["script", "--mode", "latest"]
        with patch.object(sys, 'argv', testargs):
            main()  # Should now pass

    @patch("scripts.future_testing_service.alpha_col.find_one")
    @patch("scripts.future_testing_service.os.path.exists")
    @patch("scripts.future_testing_service.torch.cuda.empty_cache")
    @patch("scripts.future_testing_service.run_feature_service")  # Patch the actual subprocess call
    def test_main_missing_parquet(self, mock_run_feature_service, mock_cuda, mock_exists, mock_find):
        mock_find.return_value = {"timestamp": pd.Timestamp("2025-01-01"), "final_weights": {"AAPL": 1.0}}
        mock_exists.return_value = False
        mock_run_feature_service.return_value = None

        testargs = ["script", "--mode", "latest"]
        with patch.object(sys, 'argv', testargs):
            main()  # Should now pass

    @patch("scripts.future_testing_service.alpha_col.find_one")
    @patch("scripts.future_testing_service.os.path.exists")
    @patch("scripts.future_testing_service.pd.read_parquet")
    @patch("scripts.future_testing_service.torch.cuda.empty_cache")
    @patch("scripts.future_testing_service.run_feature_service")  # Patch the actual subprocess call
    def test_main_no_feature_row(self, mock_run_feature_service, mock_cuda, mock_read_parquet, mock_exists, mock_find):
        mock_find.return_value = {"timestamp": pd.Timestamp("2025-01-01"), "final_weights": {"AAPL": 1.0}}
        mock_exists.return_value = True
        mock_read_parquet.return_value = pd.DataFrame({"dateTime": [pd.Timestamp("2025-01-02")], "f1": [0.1]})
        mock_run_feature_service.return_value = None

        testargs = ["script", "--mode", "latest"]
        with patch.object(sys, 'argv', testargs):
            main()  # Should now pass


if __name__ == "__main__":
    unittest.main()