"""
test_neuralfusioncore_train_service.py
Description: Unit tests for train_service.py
Author: Elham Esmaeilnia (elham.e.shirvani@gmail.com)
Date: 2025 Oct 14
"""

import pytest
from unittest.mock import patch, MagicMock
import os

# --------------------- Test 1: Missing files ---------------------
@patch("os.path.exists")
def test_missing_files(mock_exists):
    import apps.NeuralFusionCore.scripts.train_service as ts

    # Simulate missing files
    mock_exists.return_value = False

    # main() should exit gracefully without exceptions
    result = ts.main()
    assert result is None

# --------------------- Test 2: Full train_service flow ---------------------
@patch("os.path.exists")
@patch("apps.NeuralFusionCore.scripts.train_service.pd.read_parquet")
@patch("builtins.open")
@patch("json.load")
@patch("apps.NeuralFusionCore.scripts.train_service.make_loaders")
@patch("apps.NeuralFusionCore.scripts.train_service.MarketNewsFusionWeightModel")
@patch("apps.NeuralFusionCore.scripts.train_service.train_loop")
@patch("torch.cuda.is_available")
def test_train_service_flow(mock_cuda, mock_train_loop, mock_model, mock_loaders,
                            mock_json_load, mock_open, mock_read_parquet, mock_exists):
    import pandas as pd
    import apps.NeuralFusionCore.scripts.train_service as ts

    # Dummy meta and data
    dummy_meta = {
        'count_cols': ['BTC', 'ETH'],
        'data_stamp_cols': ['month', 'day', 'hour'],
        'feature_cols': ['feat1', 'feat2'],
        'stock_list': ['BTCUSDT', 'ETHUSDT']
    }
    dummy_df = pd.DataFrame({
        'feat1':[0.1,0.2,0.3],
        'feat2':[1.0,2.0,3.0],
        'dateTime':['2025-10-01','2025-10-02','2025-10-03'],
        'BTC':[0,1,0],
        'ETH':[1,0,1]
    })

    # Mocks
    mock_exists.return_value = True
    mock_read_parquet.side_effect = [dummy_df, dummy_df]
    mock_json_load.return_value = dummy_meta

    mock_tr_loader = MagicMock()
    mock_va_loader = MagicMock()
    mock_te_loader = MagicMock()
    mock_loaders.return_value = (mock_tr_loader, mock_va_loader, mock_te_loader)

    mock_model_instance = MagicMock()
    mock_model.return_value = mock_model_instance

    mock_cuda.return_value = False
    mock_train_loop.return_value = 0.123

    # Run main
    ts.main()

# --------------------- Test 3: Dynamic epoch override ---------------------
from unittest.mock import patch, MagicMock

@patch("sys.argv", ["train_service.py", "--epochs", "5"])
def test_dynamic_epoch_override_corrected():
    import apps.NeuralFusionCore.scripts.train_service as ts_dynamic

    # Override epochs like CLI
    ts_dynamic.TrainCfg.epochs = 5

    dummy_meta = {
        "count_cols": ["BTC", "ETH"],
        "data_stamp_cols": ["month", "day", "hour"],
        "feature_cols": ["feat1", "feat2"],
        "stock_list": ["BTCUSDT", "ETHUSDT"]
    }

    dummy_df = MagicMock()  # pandas DataFrame mock

    with patch("apps.NeuralFusionCore.scripts.train_service.pd.read_parquet", return_value=dummy_df), \
         patch("builtins.open", MagicMock()), \
         patch("apps.NeuralFusionCore.scripts.train_service.json.load", return_value=dummy_meta), \
         patch("os.path.exists", return_value=True), \
         patch("apps.NeuralFusionCore.scripts.train_service.make_loaders", return_value=(MagicMock(), MagicMock(), MagicMock())), \
         patch("apps.NeuralFusionCore.scripts.train_service.MarketNewsFusionWeightModel", return_value=MagicMock()), \
         patch("apps.NeuralFusionCore.scripts.train_service.train_loop", return_value=0.123):
        
        ts_dynamic.main()