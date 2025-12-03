"""
test_chronobridge_features_service.py
------------------------
Unit tests for features_service.py.
Focuses on logic, structure, and correct calls — mocks heavy I/O, Redis, and model ops.

Author: Elham Esmaeilnia (elham.e.shirvani@gmail.com)
Date: 2025 Oct 14
Version: 1.0.7
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from apps.ChronoBridge.scripts import features_service as fs


# ============================================================
# Helper fixtures
# ============================================================
@pytest.fixture
def dummy_df():
    return pd.DataFrame({
        "dateTime": pd.date_range("2025-10-01", periods=5, freq="1h", tz="UTC"),
        "BINANCE:BTCUSDT_close": np.random.rand(5),
        "BINANCE:BTCUSDT_volume": np.random.rand(5),
    })


@pytest.fixture(autouse=True)
def mock_config(tmp_path):
    """Automatically mock config objects for all tests."""
    with patch.object(fs, "P", MagicMock()) as mock_P, \
         patch.object(fs, "MC", MagicMock()):
        mock_P.processed_dir = str(tmp_path)
        mock_P.normalizer_pkl = str(tmp_path / "norm.pkl")
        mock_P.__dict__["processed_dir"] = str(tmp_path)
        fs.P = mock_P

        fs.MC.symbols_usdt = ["BINANCE:BTCUSDT", "BINANCE:ETHUSDT"]
        yield


# ============================================================
# Test: parse_time_args
# ============================================================
def test_parse_time_args_valid():
    s, e = fs.parse_time_args("2025-10-01T00:00", "2025-10-02T00:00")
    assert isinstance(s, pd.Timestamp)
    assert isinstance(e, pd.Timestamp)
    assert s < e


def test_parse_time_args_none():
    s, e = fs.parse_time_args(None, None)
    assert s is None
    assert e is None


# ============================================================
# Test: make_features_from_redis
# ============================================================
def test_make_features_from_redis(monkeypatch, dummy_df):
    # ---------- Patch any NewsCfg instance so it has embedding config (defensive) ----------
    for name in dir(fs):
        obj = getattr(fs, name)
        if getattr(obj, "__class__", None) and obj.__class__.__name__ == "NewsCfg":
            obj.embedding_dim = getattr(obj, "embedding_dim", 3)
            obj.max_len = getattr(obj, "max_len", 2048)
            obj.batch_size = getattr(obj, "batch_size", 32)
            obj.pooling = getattr(obj, "pooling", "mean")

    # ---------- Dummy news DataFrame in realistic shape ----------
    now = datetime(2025, 10, 1, tzinfo=timezone.utc)
    df_news_raw = pd.DataFrame({
        "releasedAt": [now] * 5,
        "content": ["Market moves up"] * 5,
        "news_count": [1] * 5,
        "assets": [[{"symbol": "BINANCE:BTCUSDT"}, {"symbol": "BINANCE:ETHUSDT"}]] * 5,
    })

    # ---------- Patch Redis loader functions on fs (no real Redis calls) ----------
    monkeypatch.setattr(
        fs,
        "load_ohlcv_from_redis",
        lambda symbols, start_time=None, end_time=None: {"BINANCE:BTCUSDT": dummy_df},
        raising=False,
    )
    monkeypatch.setattr(
        fs,
        "load_news_range_from_redis",
        lambda start_time, end_time: df_news_raw,
        raising=False,
    )

    # ---------- Patch F (feature utils) ----------
    mock_F = MagicMock()
    mock_F.add_targets_and_features.return_value = dummy_df
    mock_F.merge_assets.return_value = dummy_df.copy()
    mock_F.make_time_cols.return_value = pd.DataFrame({"hour": [1, 2, 3, 4, 5]})
    mock_F.attach_news.return_value = dummy_df.assign(
        embedding=[[0.1, 0.2, 0.3]] * len(dummy_df)
    )
    monkeypatch.setattr(fs, "F", mock_F, raising=False)

    # ---------- Patch N (news utils) ----------
    mock_N = MagicMock()

    def fake_add_onehot_columns(df, symbols, *args, **kwargs):
        df2 = df.copy()
        for s in symbols:
            df2[s] = 1
        return df2

    mock_N.add_onehot_columns.side_effect = fake_add_onehot_columns
    mock_N.load_text_encoder.return_value = ("tok", "mdl", "cpu", 32)
    mock_N.embed_texts.side_effect = lambda texts, *a, **k: [[0.0, 0.0, 0.0]] * len(texts)
    mock_N.resample_news_3m.side_effect = lambda df, *a, **k: df
    monkeypatch.setattr(fs, "N", mock_N, raising=False)

    # ---------- Call function under test ----------
    merged, news, no_news_vec = fs.make_features_from_redis()

    # ---------- Assertions: only public contract ----------
    assert isinstance(merged, pd.DataFrame)
    assert not merged.empty
    assert "BINANCE:BTCUSDT_close" in merged.columns

    assert isinstance(news, pd.DataFrame)
    assert isinstance(no_news_vec, (list, np.ndarray))


# ============================================================
# Test: time_split_and_save (train mode)
# ============================================================
@patch("apps.ChronoBridge.scripts.features_service.F.normalize_train_val_test_stream")
@patch("apps.ChronoBridge.scripts.features_service.json.dump")
@patch("apps.ChronoBridge.scripts.features_service.pd.DataFrame.to_parquet")
def test_time_split_and_save_train(mock_to_parquet, mock_json_dump, mock_norm, dummy_df):
    mock_norm.return_value = (
        dummy_df.iloc[:3],
        dummy_df.iloc[3:],
        pd.DataFrame(),
        {"mean": 0},
    )
    fs.time_split_and_save(dummy_df, val_frac=0.2, mode="train")

    assert mock_to_parquet.call_count == 2
    mock_json_dump.assert_called_once()


# ============================================================
# Test: time_split_and_save (finetune mode)
# ============================================================
@patch("apps.ChronoBridge.scripts.features_service.F.apply_existing_normalizer")
@patch("apps.ChronoBridge.scripts.features_service.pd.DataFrame.to_parquet")
def test_time_split_and_save_finetune(mock_to_parquet, mock_apply_norm, dummy_df):
    mock_apply_norm.return_value = dummy_df
    fs.time_split_and_save(dummy_df, val_frac=0.2, mode="finetune")

    assert mock_to_parquet.call_count == 2
    mock_apply_norm.assert_called_once()


# ============================================================
# Test: time_split_and_save (inference mode)
# ============================================================
@patch("apps.ChronoBridge.scripts.features_service.F.apply_existing_normalizer")
@patch("apps.ChronoBridge.scripts.features_service.pd.DataFrame.to_parquet")
def test_time_split_and_save_inference(mock_to_parquet, mock_apply_norm, dummy_df):
    mock_apply_norm.return_value = dummy_df
    fs.time_split_and_save(dummy_df, val_frac=0.2, mode="inference")

    mock_apply_norm.assert_called_once()
    mock_to_parquet.assert_called_once()


# ============================================================
# Test: main (mock CLI args)
# ============================================================
@patch("apps.ChronoBridge.scripts.features_service.make_features_from_redis")
@patch("apps.ChronoBridge.scripts.features_service.time_split_and_save")
@patch("argparse.ArgumentParser.parse_args")
def test_main(mock_parse_args, mock_time_split, mock_make_features, dummy_df):
    mock_parse_args.return_value = MagicMock(
        mode="train",
        latest_hours=1,
        history_days=None,
        start_time=None,
        end_time=None,
        val_frac=0.2,
    )
    mock_make_features.return_value = (dummy_df, None, [0.0])
    fs.main()

    mock_make_features.assert_called_once()
    mock_time_split.assert_called_once()
