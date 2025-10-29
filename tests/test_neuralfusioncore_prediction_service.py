
"""
# tests/test_neuralfusioncore_prediction_service.py
Unit tests for prediction_service.py

These tests verify that:
1. The prediction service handles missing files gracefully.
2. The end-to-end inference flow executes with mocked dependencies.
3. Model loading and prediction saving logic works safely without real Redis/Mongo/torch.

All external dependencies are mocked:
- MongoDB, Redis, torch, subprocess, file I/O, model, dataset loader.

Author: Elham Esmaeilnia(elham.e.shirvani@gmail.com)
Date: 2025-10-15
"""

import pytest
import json
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
import torch  # Keep real torch for isinstance/type checks

# -------------------- DUMMY DATA --------------------
dummy_df = pd.DataFrame({
    "a": [1, 2],
    "b": [3, 4],
    "cnt": [5, 6],
    "timestamp": [7, 8]
})

dummy_meta = {
    "seq_len": 12,
    "feature_cols": ["a", "b"],
    "feature_list": ["a", "b"],
    "stock_list": ["AAPL"],
    "count_cols": ["cnt"],
    "data_stamp_cols": ["timestamp"]
}

# -------------------- TEST 1: Missing Files --------------------
@patch("transformers.utils.import_utils._is_package_available", return_value=(True, "1.0"))
def test_prediction_service_missing_file(mock_torch_available):
    import apps.NeuralFusionCore.scripts.prediction_service as ps

    ps.redis_client = MagicMock()
    ps.mongo_col = MagicMock()

    with patch("sys.argv", ["prediction_service.py"]), \
         patch("os.path.exists", return_value=False), \
         patch("subprocess.run", return_value=None):
        ps.main()  # Should exit gracefully

# -------------------- TEST 2: Full Flow --------------------
@patch("transformers.utils.import_utils._is_package_available", return_value=(True, "1.0"))
@patch("torch.load")
@patch("torch.no_grad")
def test_prediction_service_flow(mock_no_grad, mock_torch_load, mock_torch_available):
    import apps.NeuralFusionCore.scripts.prediction_service as ps

    # Mock Redis/Mongo
    ps.redis_client = MagicMock()
    ps.mongo_col = MagicMock()
    mock_model = MagicMock()

    # Mock test loader
    mock_te_loader = [
        {"timeseries": MagicMock(), "news": MagicMock(),
         "news_count": MagicMock(), "time_mask": MagicMock()}
    ]

    # torch.load returns a dict of tensors
    mock_torch_load.return_value = {"layer1": torch.tensor([1.0, 2.0])}
    mock_no_grad.return_value.__enter__.return_value = None  # context manager

    # Mock feature weights function
    mock_weights_func = MagicMock(
        return_value=MagicMock(
            squeeze=lambda dim: MagicMock(
                cpu=lambda: MagicMock(
                    numpy=lambda: [[0.5, -0.5]]
                )
            )
        )
    )

    def mock_exists(path):
        return "online_test.parquet" in path or "meta.json" in path

    with patch("sys.argv", ["prediction_service.py"]), \
         patch("os.path.exists", side_effect=mock_exists), \
         patch("apps.NeuralFusionCore.scripts.prediction_service.pd.read_parquet", return_value=dummy_df), \
         patch("builtins.open", mock_open(read_data=json.dumps(dummy_meta))), \
         patch("apps.NeuralFusionCore.scripts.prediction_service.json.load", return_value=dummy_meta), \
         patch("apps.NeuralFusionCore.scripts.prediction_service.make_loaders", return_value=(None, None, mock_te_loader)), \
         patch("apps.NeuralFusionCore.scripts.prediction_service.MarketNewsFusionWeightModel", return_value=mock_model), \
         patch("apps.NeuralFusionCore.scripts.prediction_service.weights_long_short_topk_abs", mock_weights_func), \
         patch("apps.NeuralFusionCore.scripts.prediction_service.subprocess.run", return_value=None):

        ps.main()  # ✅ Should run fully

# -------------------- TEST 3: load_model --------------------
def test_load_model_behavior():
    import sys

    # Patch pymongo to avoid deadlocks
    sys.modules["pymongo"] = MagicMock()
    sys.modules["pymongo.synchronous"] = MagicMock()
    sys.modules["pymongo.synchronous.pool"] = MagicMock()
    sys.modules["pymongo.synchronous.mongo_client"] = MagicMock()

    import apps.NeuralFusionCore.scripts.prediction_service as ps

    dummy_cfg = {"task_name": "classification"}
    fake_weights_path = "dummy.pt"

    # Create mock model
    mock_model_instance = MagicMock()
    mock_model_instance.to.return_value = mock_model_instance

    # torch.load returns real tensor dict
    def debug_load(path, map_location=None):
        return {"model_state_dict": {"layer1": torch.tensor([1, 2, 3])}}

    with patch(
        "apps.NeuralFusionCore.scripts.prediction_service.MarketNewsFusionWeightModel",
        return_value=mock_model_instance
    ), patch("os.path.exists", return_value=True), \
         patch("torch.load", side_effect=debug_load), \
         patch.object(ps, "P", MagicMock(weights_pt=fake_weights_path)):

        ps.load_model(dummy_cfg, feat_cols_len=2, stock_list_len=2, count_dim=1)
        assert mock_model_instance.load_state_dict.called, "load_state_dict() was not called"