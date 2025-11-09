"""
test_neuralfusioncore_model.py
Unit tests for lib/model.py
Covers TimesNet-based MarketNews fusion model.
Author: Elham Esmaeilnia(elham.e.shirvani@gmail.com)
Date: 2025-10-15
"""

import pytest
import torch
from unittest.mock import patch
import apps.NeuralFusionCore.lib.model as m

# ---------------------------------------------------------------------
# FIXED CONFIGS: include all keys that your model.py expects
# ---------------------------------------------------------------------
dummy_configs = {
    "seq_len": 5,
    "pred_len": 1,       # You said pred_len = 1
    "label_len": 1,
    "d_model": 8,
    "d_ff": 16,
    "num_kernels": 2,
    "e_layers": 1,
    "enc_in": 8,
    "embed": "fixed",
    "freq": "h",
    "dropout": 0.0,
    "top_k": 2,          # required by TimesBlock
    "n_heads": 2,        # to keep consistency
    "num_layers": 1
}


# ---------------------------------------------------------------------
# TimesBlock
# ---------------------------------------------------------------------
@patch("apps.NeuralFusionCore.lib.model.FFT_for_Period")
def test_timesblock_forward(mock_fft):
    """Ensure TimesBlock forward works with consistent seq_len/pred_len."""

    dummy_cfg = {
        "seq_len": 6,
        "pred_len": 1,
        "d_model": 8,
        "d_ff": 16,
        "dropout": 0.1,
        "num_kernels": 4,
        "top_k": 2,
    }

    total_len = dummy_cfg["seq_len"] + dummy_cfg["pred_len"]
    valid_period = total_len  # ensures no padding issues

    mock_fft.return_value = ([valid_period] * dummy_cfg["top_k"], torch.ones(2, 2))

    tb = m.TimesBlock(dummy_cfg)
    # Include pred_len in input length
    x = torch.rand(2, dummy_cfg["seq_len"] + dummy_cfg["pred_len"], dummy_cfg["d_model"])
    y = tb(x)

    assert isinstance(y, torch.Tensor)
    assert y.shape[0] == x.shape[0]
    assert y.shape[-1] == x.shape[-1]
# ---------------------------------------------------------------------
# GatedCrossAttentionFusion2D
# ---------------------------------------------------------------------
def test_gated_cross_attention_fusion2d_forward():
    """Ensure GatedCrossAttentionFusion2D runs properly."""
    d_model = 8
    n_heads = 2
    fusion = m.GatedCrossAttentionFusion2D(d_model, n_heads)  # ✅ positional args
    primary = torch.rand(2, d_model)
    auxiliary = torch.rand(2, d_model)
    out = fusion(primary, auxiliary)
    assert out.shape == primary.shape


# ---------------------------------------------------------------------
# MSGCAFusion
# ---------------------------------------------------------------------
def test_msgca_fusion_forward():
    """Ensure MSGCAFusion forward works properly."""
    d_model = 8
    n_heads = 2
    fusion = m.MSGCAFusion(d_model, n_heads)  # ✅ positional args
    indicators = torch.rand(2, d_model)
    documents = torch.rand(2, d_model)
    out = fusion(indicators, documents)
    assert out.shape == indicators.shape


# ---------------------------------------------------------------------
# MarketNewsFusionWeightModel - forward test
# ---------------------------------------------------------------------
def test_marketnews_fusion_weight_model_forward():
    """Test main forward pass output shape."""
    num_stocks = 2
    ts_input_dim = dummy_configs["enc_in"]
    news_embed_dim = dummy_configs["d_model"]
    count_dim = 2

    model = m.MarketNewsFusionWeightModel(
        configs=dummy_configs,
        ts_input_dim=ts_input_dim,
        num_stocks=num_stocks,
        d_model=dummy_configs["d_model"],
        nhead=2,
        num_layers=1,
        news_embed_dim=news_embed_dim,
        hidden_dim=8,
        count_dim=count_dim,
    )

    B, L = 2, dummy_configs["seq_len"]
    ts_input = torch.rand(B, L, ts_input_dim)
    x_mark_enc = torch.rand(B, L, ts_input_dim)
    count_input = torch.rand(B, L, count_dim)
    news_input = torch.rand(B, L, news_embed_dim)

    out = model(ts_input, x_mark_enc, count_input, news_input)
    assert out.shape == (B, num_stocks)


# ---------------------------------------------------------------------
# MarketNewsFusionWeightModel - embeddings mode
# ---------------------------------------------------------------------
def test_marketnews_fusion_weight_model_embeddings():
    """Test embeddings extraction mode (return_embeddings=True)."""
    num_stocks = 2
    ts_input_dim = dummy_configs["enc_in"]
    news_embed_dim = dummy_configs["d_model"]
    count_dim = 2

    model = m.MarketNewsFusionWeightModel(
        configs=dummy_configs,
        ts_input_dim=ts_input_dim,
        num_stocks=num_stocks,
        d_model=dummy_configs["d_model"],
        nhead=2,
        num_layers=1,
        news_embed_dim=news_embed_dim,
        hidden_dim=8,
        count_dim=count_dim,
    )

    B, L = 2, dummy_configs["seq_len"]
    ts_input = torch.rand(B, L, ts_input_dim)
    x_mark_enc = torch.rand(B, L, ts_input_dim)
    count_input = torch.rand(B, L, count_dim)
    news_input = torch.rand(B, L, news_embed_dim)

    fused = model(ts_input, x_mark_enc, count_input, news_input, return_embeddings=True)
    assert fused.shape == (B, num_stocks, dummy_configs["d_model"])