# run_realtime.py
import os, time, json, pickle, math
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timezone, timedelta

from config import Paths, NewsCfg, MarketCfg, FeatureCfg, TrainCfg, PortfolioCfg
from lib import market as M
from lib import features as F
from lib import news as N
from lib.model import MarketNewsFusionModel
from lib.portfolio import method0_mvo, method1_small_l2, method2_subset_bagging

# ---------------------- Helpers ----------------------

def ceil_to_next_3min(ts_utc: datetime) -> datetime:
    # round up to next multiple of 3 minutes
    minute = (ts_utc.minute // 3) * 3
    base = ts_utc.replace(second=0, microsecond=0, minute=minute)
    if base < ts_utc:
        base += timedelta(minutes=3)
    return base

def fetch_recent_1m(client, symbol, minutes=720):
    # Fetch last N minutes (Binance lets you specify start/end as strings; we’ll use ms)
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=minutes+5)
    start_str = str(int(start.timestamp() * 1000))
    end_str = str(int(end.timestamp() * 1000))
    df = M.fetch_klines(client, symbol, "1m", start_str, end_str)
    return df

def resample_all_to_3m(symbols, client, agg_map):
    per_asset = []
    for sym in symbols:
        # df1m = fetch_recent_1m(client, sym, minutes=24*60)  # last 24h (adjust if you want more)
        df1m = pd.read_pickle(Paths().ohlcv_realtime_dir+'/'+sym+'.pickle')
        df3m = M.resample_to_3m(df1m, agg_map)
        per_asset.append(df3m.assign(symbol=sym))
    return per_asset

def build_per_asset_features(df3m, sym, fwd_h, seq_len):
    """Same feature engineering as training, but we won't use forward targets."""
    d = df3m.copy()
    # Past stats on close for prev_*; fwd targets are computed but not used here
    d[f"return"] = 100*((d["close"].shift(-fwd_h)/d["close"]) - 1)
    d[f"prev_return"] = 100*((d["close"]/d["close"].shift(seq_len)) - 1)
    d[f"volatility"] = 100*d["close"].rolling(seq_len).std().shift(-seq_len)
    d[f"prev_volatility"] = 100*d["close"].rolling(seq_len).std()

    keep = ["close","volume","numberOfTrades","prev_return","prev_volatility","return","volatility"]
    out = d[["dateTime"] + keep].rename(columns={c:f"{sym}_{c}" for c in keep})
    # also keep instantaneous returns for “past realized” stats (not as model inputs)
    return out

def load_artifacts():
    P = Paths()
    T = TrainCfg()
    Fcfg = FeatureCfg()

    # splits meta and normalizer
    meta = json.load(open(P.processed_dir + "/meta.json"))
    feat_cols = meta["feature_cols"]
    cnt_cols = meta["count_cols"]
    with open(P.normalizer_pkl, "rb") as f:
        norm_stats = pickle.load(f)

    # model
    n_stocks = len([c for c in meta["target_cols"]])  # target per asset
    ts_in_dim = len(feat_cols)
    count_dim = len(cnt_cols) if cnt_cols else 0
    device = torch.device(T.device if torch.cuda.is_available() else "cpu")

    model = MarketNewsFusionModel(
        ts_input_dim=ts_in_dim, num_stocks=n_stocks,
        d_model=T.d_model, nhead=T.nhead, num_layers=T.num_layers,
        news_embed_dim=768, hidden_dim=T.hidden_dim, count_dim=count_dim,
        max_len=Fcfg.seq_len
    ).to(device)
    model.load_state_dict(torch.load(P.weights_pt, map_location=device))
    model.eval()

    return feat_cols, cnt_cols, norm_stats, model, device

def normalize_df(df, norm_stats, cols):
    d = df.copy()
    for c in cols:
        mu = norm_stats[c]["mean"]
        sd = norm_stats[c]["std"] if norm_stats[c]["std"] > 1e-12 else 1.0
        d[c] = (pd.to_numeric(d[c], errors="coerce").fillna(0) - mu) / sd
    return d

def embed_news_incremental(news_csv_path, news_cfg: NewsCfg, tradable_syms, tok=None, mdl=None, device=None):
    """
    Load CSV, embed all rows (or cache embeddings within the CSV by a helper column).
    For simplicity here we (re)embed recent 24h rows each tick.
    Expected columns: releasedAt, content, asset_symbols
    """
    df = pd.read_csv(news_csv_path)
    # filter plausible columns
    need = {"releasedAt","content","asset_symbols"}
    if not need.issubset(set(df.columns)):
        raise ValueError(f"News CSV must have columns {need}")

    syms = [s.replace("USDT","") for s in tradable_syms]
    df["asset_symbols"] = (df["asset_symbols"]
                           .fillna("None")
                           .apply(lambda x: [t for t in str(x).replace(" ","").split(",") if t in syms]))
    df = df[df["asset_symbols"].apply(len) > 0].copy()

    # recent slice (24h)
    now = datetime.now(timezone.utc)
    df["releasedAt"] = pd.to_datetime(df["releasedAt"], utc=True, errors="coerce")
    # recent = df[df["releasedAt"] >= (now - timedelta(hours=24))].copy()
    recent = df.copy()

    # one-hot
    onehot = (recent["asset_symbols"].explode().str.get_dummies().groupby(level=0).sum())
    recent = recent.join(onehot)

    # encoder
    if tok is None or mdl is None:
        tok, mdl, device2, _ = N.load_text_encoder(news_cfg.model_path)
        device = device or device2
    with torch.no_grad():
        embs = N.embed_texts(recent["content"].tolist(), tok, mdl, device,
                             news_cfg.max_len, news_cfg.pooling, news_cfg.batch_size)
    recent["embedding"] = [e for e in embs]
    # add count
    recent["news_count"] = 1
    # build no-news vec (static)
    no_news_vec = N.embed_texts([news_cfg.no_news_token], tok, mdl, device,
                                news_cfg.max_len, news_cfg.pooling, batch_size=1)[0]
    # resample to 3T
    news_3m = N.resample_news_3m(recent[["releasedAt","embedding","news_count"] +
                            [s.replace("USDT","") for s in tradable_syms]].copy(),
                            np.asarray(no_news_vec), rule=news_cfg.rule)
    return news_3m, no_news_vec, tok, mdl, device

def build_window_tensor(df_merged, feat_cols, cnt_cols, seq_len):
    """
    Prepare a single sample with the last seq_len rows:
      - timeseries: [1, T, F]
      - news:       [1, T, 768]
      - news_count: [1, T, |cnt_cols|]
    """
    d = df_merged.copy().fillna(0)
    # keep last seq_len rows that have embeddings
    d = d.dropna(subset=["embedding"]).tail(seq_len)
    if len(d) < seq_len:
        return None

    X_ts = d[feat_cols].astype("float32").values[None, :, :]           # [1,T,F]
    X_news = np.stack(d["embedding"].values).astype("float32")[None]   # [1,T,768]
    if cnt_cols:
        X_cnt = d[cnt_cols].astype("float32").values[None, :, :]        # [1,T,C]
    else:
        X_cnt = np.zeros((1, seq_len, 1), dtype="float32")
    return X_ts, X_news, X_cnt

def muL_to_std_corr(mu, L):
    # mu: [S], L: [S,S]
    Sigma = L @ L.T
    std = np.sqrt(np.clip(np.diag(Sigma), 1e-12, None))
    D_inv = np.diag(1.0 / std)
    corr = D_inv @ Sigma @ D_inv
    return std, corr

# ---------------------- Main loop ----------------------

def main():

    P = Paths(); NC = NewsCfg(); MC = MarketCfg(); FC = FeatureCfg(); TC = TrainCfg(); PC = PortfolioCfg()

    # load artifacts
    feat_cols, cnt_cols, norm_stats, model, device = load_artifacts()

    # Binance client
    client = M.get_client("", "")  # put keys if needed

    # News encoder state
    tok = mdl = enc_device = None
    no_news_vec = None

    print("Realtime loop starting… (Ctrl+C to stop)")


    # Align to the next 3-min boundary
    # now = datetime.now(timezone.utc)
    # next_tick = ceil_to_next_3min(now)
    # sleep_s = max(1, (next_tick - now).total_seconds())
    # time.sleep(sleep_s)

    tick_ts = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    print(f"\n=== Tick at {tick_ts.isoformat()} ===")

    # 1) Market: fetch recent 1m, resample to 3m, engineer features, merge across assets
    per3m = resample_all_to_3m(MC.symbols_usdt, client, FC.agg_cols)
    feat_frames = []
    for d in per3m:
        sym = d["symbol"].iloc[0]
        feat_frames.append(build_per_asset_features(d.drop(columns=["symbol"]), sym, FC.fwd_h, FC.seq_len))
    merged = feat_frames[0][["dateTime"]].copy()
    for f in feat_frames:
        merged = merged.merge(f, on="dateTime", how="outer")
    merged = merged.sort_values("dateTime").reset_index(drop=True)

    # 2) News: read CSV, embed recent, resample 3T and merge
    news_csvs = [os.path.join(P.data_realtime_dir, "news", f) for f in os.listdir(P.news_realtime_dir) if f.endswith(".csv")]
    if not news_csvs:
        print("No news CSV found in data/raw_news/. Using no-news vectors.")
        # construct constant no-news frame on market timestamps
        if no_news_vec is None:
            # lazy init encoder once to get a dimension; safer: load from model dim 768
            tok, mdl, enc_device, _ = N.load_text_encoder(NC.model_path)
            no_news_vec = N.embed_texts([NC.no_news_token], tok, mdl, enc_device,
                                        NC.max_len, NC.pooling, 1)[0]
        news_3m = pd.DataFrame(index=pd.to_datetime(merged["dateTime"]).dt.floor("3T").unique())
        news_3m["embedding"] = [no_news_vec] * len(news_3m)
        # count columns
        cnt_cols_full = [s[:-4] for s in MC.symbols_usdt]
        for c in cnt_cols_full: news_3m[c] = 0
        news_3m["news_count"] = 0
    else:
        # concatenate all provided CSVs (assume they append over time)
        # For simplicity, use the first CSV if you keep one file rolling
        news_3m, no_news_vec, tok, mdl, enc_device = embed_news_incremental(
            news_csvs[0], NC, MC.symbols_usdt, tok, mdl, enc_device
        )

    # Exact merge on 3T anchor
    news_3m.index = news_3m.index.tz_localize(None)
    merged = F.attach_news(merged, news_3m)
    mask = merged["embedding"].isna()
    merged.loc[mask, "embedding"] = merged.loc[mask, "embedding"].apply(
        lambda _: np.asarray(no_news_vec, dtype=float).copy()
    )
    # Add simple time cols (not used in model here; kept for parity)
    tcols = F.make_time_cols(merged)
    merged = pd.concat([merged, tcols], axis=1)

    # 3) Normalize features (train stats)
    merged = merged.fillna(0)
    merged_norm = merged.copy()
    merged_norm = normalize_df(merged_norm, norm_stats, feat_cols)

    # 4) Build window tensors (last seq_len rows)
    win = build_window_tensor(merged_norm, feat_cols, cnt_cols, FC.seq_len)
    # if win is None:
    #     print("Not enough rows yet to form a full window; waiting for more data…")
    #     continue
    X_ts, X_news, X_cnt = win

    # 5) Inference → μ, L
    with torch.no_grad():
        ts_t = torch.tensor(X_ts, device=device)
        news_t = torch.tensor(X_news, device=device)
        cnt_t = torch.tensor(X_cnt, device=device)
        mu_t, L_t = model(ts_t, cnt_t, news_t)
        mu = mu_t.squeeze(0).cpu().numpy()
        L  = L_t.squeeze(0).cpu().numpy()

    std, corr = muL_to_std_corr(mu, L)

    # 6) Portfolio weights (blend past + predicted, like backtest)
    # Build past realized returns on 3-min bars from the same merged (close columns)
    # We map asset order from target columns used at train: they were one per symbol (return).
    assets = [c.split("_")[0] for c in feat_cols if c.endswith("_prev_return")]
    # realized returns (1-step ahead) approximation: 100 * pct_change(next bar); here just use pct_change
    # since we are at the tick, we only have realized up to current
    close_cols = [f"{a}_close" for a in assets if f"{a}_close" in merged.columns]
    rets = merged[["dateTime"] + close_cols].copy().set_index("dateTime").pct_change()*100.0
    rets = rets.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    lookback = 240  # minutes (i.e., 80 three-minute bars)
    if len(rets) < lookback + 1:
        print("Not enough realized history for portfolio blending; skipping weights this tick.")
        weights0 = weights1 = weights2 = np.zeros_like(mu)
    else:
        past = rets.tail(lookback)
        past_mu = past.sum().to_numpy()
        past_vol = past.std().replace(0, 1e-8).to_numpy()
        past_corr = past.corr().fillna(0).to_numpy()

        alpha = PC.alpha
        mu_mix = (1 - alpha) * past_mu + alpha * mu
        std_mix = (1 - alpha) * past_vol + alpha * std
        corr_mix = (1 - alpha) * past_corr + alpha * corr

        stats = {"mean": mu_mix, "std": std_mix, "corr": corr_mix}
        weights0 = method0_mvo(stats, gamma=PC.gamma)
        weights1 = method1_small_l2(stats, gamma=PC.gamma, theta=PC.theta)
        weights2 = method2_subset_bagging(stats, gamma=PC.gamma,
                                          num_resamples=PC.num_resamples,
                                          subset_size=PC.subset_size)

    # 7) Print results
    print("mu (first 5):", np.round(mu[:5], 6))
    print("diag(std) (first 5):", np.round(std[:5], 6))
    print("weights (Method0) first 5:", np.round(weights0[:5], 4))
    print("weights (Method1) first 5:", np.round(weights1[:5], 4))
    print("weights (Method2) first 5:", np.round(weights2[:5], 4))

    # (Optional) persist each tick
    out_row = {
        "timestamp": tick_ts.isoformat(),
        "mu": mu.tolist(),
        "L_flat": L[np.tril_indices(L.shape[0])].tolist(),
        "std": std.tolist(),
        "weights_m0": weights0.tolist(),
        "weights_m1": weights1.tolist(),
        "weights_m2": weights2.tolist(),
    }
    os.makedirs(Paths().outputs_dir, exist_ok=True)
    with open(os.path.join(Paths().outputs_dir, "realtime_log.jsonl"), "a") as f:
        f.write(json.dumps(out_row) + "\n")



if __name__ == "__main__":
    main()
