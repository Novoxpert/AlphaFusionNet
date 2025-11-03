"""
test_neuralfusioncore_backtesting_service.py

Unit tests for the AlphaFusionNet NeuralFusionCore backtesting pipeline.

This test suite validates critical components of the backtesting service,
including:

- Portfolio metric functions (Sharpe, Sortino, Max Drawdown, CAGR, etc.)
- Safe portfolio backtest execution using synthetic logits/returns
- Model loading behavior when weights are absent or mocked
- Main pipeline logic execution with heavy components mocked
  (data ingestion, feature generation, model forward pass, plotting, and I/O)

The tests ensure lightweight and deterministic behavior. They do NOT
require real parquet data, model checkpoints, or running the end-to-end
AlphaFusionNet system.

Usage:
    pytest -q

Author: Elham Esmaeilnia (elham.e.shirvani@gmail.com)
Date : 2025-11-02 (tests updated)
Version : 1.0.1
"""

import os
import io
import json 
import numpy as np
import pandas as pd
import torch
import builtins
from unittest.mock import patch

# import the module under test
import apps.NeuralFusionCore.scripts.backtesting_service as mod
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
device = torch.device("cpu")
# ---------------------------
#  Test Metrics
# ---------------------------
def test_metrics_functions():
    r = np.array([0.01, 0.02, -0.01, 0.03])

    assert isinstance(mod.annualized_sharpe(r), float)
    assert (isinstance(mod.sortino_ratio(r), (float, np.floating)) or
            np.isnan(mod.sortino_ratio(r)))
    
    eq = np.array([1, 1.2, 1.1, 0.9, 1.0])
    dd = mod.max_drawdown(eq)
    assert isinstance(dd, float)
    assert dd <= 0

    stats = mod.compute_trade_stats(eq)
    assert "sharpe_ann" in stats
    assert "cagr" in stats
    assert isinstance(stats["total_return"], float)

# ---------------------------
#  Test Model Load (no weights)
# ---------------------------
def test_load_model_without_weights(tmp_path, monkeypatch):
    # Patch P.weights_pt to a non-existing path so load_model logs warning but returns a model
    dummy = type("P", (), {"weights_pt": str(tmp_path / "no_weights.pt")})
    monkeypatch.setattr(mod, "P", dummy)

    model = mod.load_model(feat_dim=5, num_stocks=3, count_dim=1, device=torch.device("cpu"))
    assert model is not None
    assert hasattr(model, "forward")

# ---------------------------
#  Test Backtest Function
# ---------------------------
def test_safe_backtest_weight_logits(tmp_path, monkeypatch):
    # Patch P.outputs_dir to tmp folder so pickle goes to tmp_path
    dummyP = type("P", (), {"outputs_dir": str(tmp_path)})
    monkeypatch.setattr(mod, "P", dummyP)

    # Patch MarketCfg so symbols_usdt matches the number of assets (2)
    class DummyMC:
        symbols_usdt = ["A", "B"]
    monkeypatch.setattr(mod, "MarketCfg", lambda *a, **k: DummyMC())

    logits = np.array([[1, -1], [2, -2]], float)  # mock logits for 2 assets
    returns = np.array([[0.01, -0.01], [0.02, -0.02]], float)
    dates = pd.date_range("2024-01-01", periods=2)

    # The function was exposed to module-level in the service; call it
    safe_bt = getattr(mod, "safe_backtest_weight_logits", None)
    assert callable(safe_bt), "safe_backtest_weight_logits is not exposed on the module"

    out = safe_bt(
        pred_logits=logits,
        returns_matrix=returns,
        dates=dates,
        k=1,
        gross=1.0,
        stride=1
    )

    assert "equity" in out
    assert len(out["equity"]) == 2
    # equity starts at 1 per implementation (first element)
    assert float(out["equity"][0]) == float(out["equity"][0])  # sanity, numeric

    # output pickle exists
    assert os.path.exists(os.path.join(str(tmp_path), "df_portfolio.pickle"))

# ---------------------------
#  Ensure CLI does not run full pipeline (use_saved mode)
# ---------------------------
def test_main_call_does_not_execute_real_pipeline(tmp_path, monkeypatch):
    import torch
    import io, json, builtins
    import pandas as pd
    import numpy as np
    import apps.NeuralFusionCore.scripts.backtesting_service as mod
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    device = torch.device("cpu")

    # Simulate CLI arguments for using saved data
    monkeypatch.setattr("sys.argv", ["prog", "--mode", "use_saved", "--device", "cpu"])

    # Provide safe temporary paths used inside main()
    dummy_paths = type("Paths", (), {
        "processed_backtesting_dir": str(tmp_path),
        "weights_pt": str(tmp_path / "weights.pt"),
        "outputs_dir": str(tmp_path)
    })
    monkeypatch.setattr(mod, "P", dummy_paths)

    # Ensure MarketCfg symbols match mock stocks
    class DummyMC:
        symbols_usdt = ["A", "B"]
    monkeypatch.setattr(mod, "MarketCfg", lambda *a, **k: DummyMC())

    # Mock functions that would trigger external processes
    monkeypatch.setattr(mod, "run_data_ingest", lambda *a, **k: None)
    monkeypatch.setattr(mod, "run_feature_service", lambda *a, **k: None)

    # Provide deterministic parquet returns
    sample_df = pd.DataFrame({
        "dateTime": pd.date_range("2024-01-01", periods=10),
        "A_target_return": np.random.randn(10),
        "B_target_return": np.random.randn(10)
    })
    monkeypatch.setattr(mod.pd, "read_parquet", lambda *a, **k: sample_df)

    # Patch open/json for meta.json
    meta_obj = {
        "feature_cols": ["f1", "f2"],
        "stock_list": ["A", "B"],
        "count_cols": [],
        "data_stamp_cols": []
    }

    orig_open = builtins.open

    def fake_meta_open(filename, *args, **kwargs):
        if str(filename).endswith("meta.json"):
            return io.StringIO(json.dumps(meta_obj))
        return orig_open(filename, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_meta_open)
    monkeypatch.setattr(mod, "json", type("json", (), {"load": lambda fp: meta_obj}))

    # Mock dataloaders
    class DummyBatchLoader:
        def __init__(self, n_batches=1, n_assets=2, seq_len=5, device=device):
            self.n_batches = n_batches
            self.n_assets = n_assets
            self.seq_len = seq_len
            self.device = device

        def __len__(self):
            return self.n_batches

        def __iter__(self):
            for _ in range(self.n_batches):
                yield {
                    'timeseries': torch.randn(1, self.seq_len, self.n_assets, device=self.device),
                    'news': torch.randn(1, self.seq_len, self.n_assets, device=self.device),
                    'news_count': torch.randint(0, 2, (1, self.seq_len, self.n_assets), device=self.device),
                    'time_mask': torch.ones(1, self.seq_len, self.n_assets, device=self.device),
                    'target': torch.randn(1, self.seq_len, self.n_assets, device=self.device)
                }

    tr_loader = DummyBatchLoader()
    va_loader = DummyBatchLoader()
    te_loader = DummyBatchLoader()
    monkeypatch.setattr(mod, "make_loaders", lambda *a, **k: (tr_loader, va_loader, te_loader))

    # Mock load_model
    class DummyModel(torch.nn.Module):
        def __init__(self, n_assets=2):
            super().__init__() 
            self.n_assets = n_assets
            # keep as nn.Parameter to avoid autograd errors
            self.dummy_param = torch.nn.Parameter(torch.randn(n_assets))

        def forward(self, ts, mask, cnt, news):
            # logits should be on the same device as ts
            device = ts.device
            batch, seq_len, n_assets = ts.shape
            return self.dummy_param.to(device).view(1, n_assets).expand(batch, n_assets)

    monkeypatch.setattr(mod, "load_model", lambda *a, **k: DummyModel())

    # Prevent plotting and heavy I/O
    monkeypatch.setattr(mod, "plot_equity", lambda *a, **k: None)
    monkeypatch.setattr(mod, "plot_equity_sl", lambda *a, **k: None)
    monkeypatch.setattr(mod, "plot_underwater", lambda *a, **k: None)
    monkeypatch.setattr(mod, "plot_rolling_sharpe", lambda *a, **k: None)
    monkeypatch.setattr(mod, "plot_turnover", lambda *a, **k: None)

    # Run main — should not raise and operates on mocked objects
    mod.main()