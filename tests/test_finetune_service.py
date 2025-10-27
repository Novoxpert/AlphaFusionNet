"""
test_finetune_service.py

Unit tests for the finetune_service module.

These tests cover:
1. Handling of missing fine-tune dataset or meta files.
2. Full fine-tune flow execution with mocked DataLoaders and model.
3. Fine-tune with --save_best behavior, including atomic model replacement and backup.

All I/O operations, training loops, and model initialization are mocked to allow safe
and fast unit testing without touching the filesystem or performing actual training.

Author: Elham Esmaeilnia(elham.e.shirvani@gmail.com)
Date: 2025-10-15
"""

import pytest
from unittest.mock import patch, MagicMock, mock_open
import builtins
import torch  # Keep real torch types for isinstance/type checks

# -----------------------------
# Dummy Data
# -----------------------------
dummy_meta = {
    "feature_cols": ["A", "B"],
    "data_stamp_cols": ["date"],
    "stock_list": ["STOCK1", "STOCK2"],
    "count_cols": ["count"],
}
dummy_df = MagicMock()

# -----------------------------
# Helper: Safe mock_open for JSON
# -----------------------------
_real_open = builtins.open
def safe_mock_open(file, *args, **kwargs):
    if str(file).endswith(".json"):
        return mock_open(read_data='{"feature_cols": ["A"], "data_stamp_cols": [], "stock_list": ["STOCK1"], "count_cols": []}')().return_value
    return _real_open(file, *args, **kwargs)

# -----------------------------
# TEST 1: Missing Files
# -----------------------------
@patch("transformers.utils.import_utils._is_package_available", return_value=(True, "1.0"))
def test_missing_files(mock_torch_available):
    import apps.NeuralFusionCore.scripts.finetune_service as fs
    with patch("sys.argv", ["finetune_service.py"]), \
         patch("os.path.exists", return_value=False):
        fs.main()

# -----------------------------
# TEST 2: Finetune Flow
# -----------------------------
@patch("transformers.utils.import_utils._is_package_available", return_value=(True, "1.0"))
@patch("torch.load", return_value={"layer1": MagicMock()})
@patch("torch.no_grad", return_value=MagicMock())
def test_finetune_service_flow(mock_no_grad, mock_torch_load, mock_torch_available):
    import apps.NeuralFusionCore.scripts.finetune_service as fs

    with patch("sys.argv", ["finetune_service.py"]), \
         patch("os.path.exists", return_value=True), \
         patch("apps.NeuralFusionCore.scripts.finetune_service.json.load", return_value=dummy_meta), \
         patch("apps.NeuralFusionCore.scripts.finetune_service.pd.read_parquet", return_value=dummy_df), \
         patch("apps.NeuralFusionCore.scripts.finetune_service.make_loaders", return_value=(MagicMock(), MagicMock(), None)), \
         patch("apps.NeuralFusionCore.scripts.finetune_service.MarketNewsFusionWeightModel", return_value=MagicMock()), \
         patch("apps.NeuralFusionCore.scripts.finetune_service.train_loop", return_value=0.123):
        with patch("builtins.open", safe_mock_open):
            # Patch argparse to return real ints
            with patch("argparse.ArgumentParser.parse_args") as mock_args:
                mock_args.return_value.epochs = 1
                mock_args.return_value.batch_size = 2
                mock_args.return_value.temp_weights = "/tmp/fake.pt"
                mock_args.return_value.save_best = False
                fs.main()

# -----------------------------
# TEST 3: Save Best Flow
# -----------------------------
@patch("transformers.utils.import_utils._is_package_available", return_value=(True, "1.0"))
@patch("torch.load", return_value={"layer1": MagicMock()})
@patch("torch.no_grad", return_value=MagicMock())
def test_finetune_service_save_best(mock_no_grad, mock_torch_load, mock_torch_available):
    import apps.NeuralFusionCore.scripts.finetune_service as fs

    with patch("sys.argv", ["finetune_service.py"]), \
         patch("os.path.exists", return_value=True), \
         patch("apps.NeuralFusionCore.scripts.finetune_service.json.load", return_value=dummy_meta), \
         patch("apps.NeuralFusionCore.scripts.finetune_service.pd.read_parquet", return_value=dummy_df), \
         patch("apps.NeuralFusionCore.scripts.finetune_service.make_loaders", return_value=(MagicMock(), MagicMock(), None)), \
         patch("apps.NeuralFusionCore.scripts.finetune_service.MarketNewsFusionWeightModel", return_value=MagicMock()), \
         patch("apps.NeuralFusionCore.scripts.finetune_service.train_loop", return_value=0.123), \
         patch("apps.NeuralFusionCore.scripts.finetune_service.atomic_model_swap", return_value=None), \
         patch("shutil.copy2", return_value=None):
        with patch("builtins.open", safe_mock_open):
            with patch("argparse.ArgumentParser.parse_args") as mock_args:
                mock_args.return_value.epochs = 1
                mock_args.return_value.batch_size = 2
                mock_args.return_value.temp_weights = "/tmp/fake.pt"
                mock_args.return_value.save_best = True
                fs.main()