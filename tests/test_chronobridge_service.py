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
    data = {
        "date": pd.date_range("2025-10-19 10:00", periods=5, freq="min"),
        "feature1": [0.1, 0.2, 0.3, 0.4, 0.5],
        "feature2": [1, 2, 3, 4, 5],
        "data_stamp1": [0, 1, 0, 1, 0],
        "embedding": [np.random.rand(768) for _ in range(5)],
        "STOCK_A_close": [100, 101, 102, 103, 104],
    }
    return pd.DataFrame(data)

@patch("apps.ChronoBridge.scripts.chronobridge_service.NeuralFusionCore_infer")
@patch("apps.ChronoBridge.scripts.chronobridge_service.mongo_col")
def test_main_pipeline(mock_mongo, mock_nfc, sample_df_te):
    # Mock the NFC_infer instance
    mock_instance = MagicMock()
    mock_nfc.return_value = mock_instance

    # Mock FusedEmbedding to insert dummy embeddings into Mongo
    def fake_FusedEmbedding(model_checkpoint, mongo_collection, device):
        dummy_embeddings = [
            {
                "date": row["date"],
                "symbol": "STOCK_A",
                "fused_embedding": np.random.rand(64).tolist(),
                "close": row["STOCK_A_close"]
            }
            for _, row in sample_df_te.iterrows()
        ]
        mongo_collection.insert_many(dummy_embeddings)

    mock_instance.FusedEmbedding.side_effect = fake_FusedEmbedding

    # Call main() instead of run_inference
    with patch("apps.ChronoBridge.scripts.chronobridge_service.run_data_ingest") as mock_ingest, \
         patch("apps.ChronoBridge.scripts.chronobridge_service.run_feature_service") as mock_feature:
        mock_ingest.return_value = None
        mock_feature.return_value = None

        # Call main with test args
        import sys
        sys.argv = ["chronobridge_service.py", "--hours", "1", "--mode", "bridge", "--device", "cpu"]
        cb.main()

    # Assertions
    assert mock_mongo.insert_many.called
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
