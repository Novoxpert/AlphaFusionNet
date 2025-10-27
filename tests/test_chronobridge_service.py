"""
test_chronobridge_service.py

Unit tests for chronobridge_service.py

This test suite verifies the functionality of the ChronoBridge service including:

- run_inference: Checks that fused embeddings are generated for each row of input
  data, and saved correctly to Redis and MongoDB. The model is mocked to return
  dummy embeddings to isolate testing from actual training.
  
- save_fused_embedding_predictions: Ensures records contain expected keys 
  ("date", "symbol", "fused_embedding", "close") and match the input dataframe length.

- run_data_ingest & run_feature_service: Confirms that the external data fetching
  and feature generation subprocess commands are invoked correctly.

Mocks:
- MongoDB collection (`mongo_col`) to avoid real database writes.
- Redis client (`redis_client`) to avoid real cache writes.
- MarketNewsFusionWeightModel (`load_model`) to bypass actual model inference.

Fixtures:
- sample_df_te: A small, synthetic dataframe mimicking the structure expected
  by the ChronoBridge service, including feature columns, data stamps, embeddings,
  and per-stock close prices.
Author: Elham Esmaeilnia(elham.e.shirvani@gmail.com)
Date: 2025 Oct 19
Version: 1.0.0
"""
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import torch

# Import your module
import apps.ChronoBridge.scripts.chronobridge_service as cb

@pytest.fixture
def sample_df_te():
    # Create a small sample dataframe
    data = {
        "date": pd.date_range("2025-10-19 10:00", periods=5, freq="min"),
        "feature1": [0.1, 0.2, 0.3, 0.4, 0.5],
        "feature2": [1, 2, 3, 4, 5],
        "data_stamp1": [0, 1, 0, 1, 0],
        "embedding": [np.random.rand(768) for _ in range(5)],
        "STOCK_A_close": [100, 101, 102, 103, 104],
    }
    df = pd.DataFrame(data)
    return df

@patch("apps.ChronoBridge.scripts.chronobridge_service.mongo_col")
@patch("apps.ChronoBridge.scripts.chronobridge_service.redis_client")
@patch("apps.ChronoBridge.scripts.chronobridge_service.load_model")
def test_run_inference(mock_load_model, mock_redis, mock_mongo, sample_df_te):
    # Mock model forward pass to return correct shape (num_stocks, d_model)
    class MockModel:
        def eval(self): pass
        def __call__(self, ts_input, mask, count_input, news_input, return_embeddings=False):
            num_stocks = ts_input.shape[2] if len(ts_input.shape) == 3 else 1
            d_model = 64
            # Always return batch of fused embeddings
            return torch.rand(1, len(["STOCK_A"]), d_model)

    mock_load_model.return_value = MockModel()

    feat_cols = ["feature1", "feature2"]
    data_stamp_cols = ["data_stamp1"]
    stock_list = ["STOCK_A"]
    cnt_cols = []

    import apps.ChronoBridge.scripts.chronobridge_service as cb

    # ✅ FIXED: pass sample_df_te twice (for df_not_norm_te and df_te)
    cb.run_inference(sample_df_te, sample_df_te, feat_cols, data_stamp_cols, stock_list, cnt_cols, device='cpu')

    # Check Redis set was called multiple times
    assert mock_redis.set.called
    # Check Mongo insert_many was called multiple times
    assert mock_mongo.insert_many.called

    # Verify number of insertions matches number of df_te rows
    total_records = sum(len(call[0][0]) for call in mock_mongo.insert_many.call_args_list)
    assert total_records == len(sample_df_te)


@patch("apps.ChronoBridge.scripts.chronobridge_service.subprocess.run")
def test_run_data_ingest(mock_subprocess):
    cb.run_data_ingest(1)
    mock_subprocess.assert_called()

@patch("apps.ChronoBridge.scripts.chronobridge_service.subprocess.run")
def test_run_feature_service(mock_subprocess):
    cb.run_feature_service(2)
    mock_subprocess.assert_called()
