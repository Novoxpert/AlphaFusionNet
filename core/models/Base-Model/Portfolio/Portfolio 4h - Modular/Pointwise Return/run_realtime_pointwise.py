
# Realtime runner for pointwise mu model using historical vol & covariance
import os, time, json, pickle, numpy as np, pandas as pd, torch
from datetime import datetime, timezone, timedelta

from config import Paths, NewsCfg, MarketCfg, FeatureCfg, TrainCfg, PortfolioCfg
from lib import market as M
from lib import features as F
from lib import news as N
from lib.model import MarketNewsFusionModel
from lib.portfolio import method0_mvo, method1_small_l2, method2_subset_bagging

def round_to_next_3m(ts):
    minute = (ts.minute // 3) * 3
    base = ts.replace(minute=minute, second=0, microsecond=0)
    if base < ts: base += timedelta(minutes=3)
    return base

def fetch_recent_1m(client, symbol, minutes=24*60):
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=minutes+5)
    return M.fetch_klines(client, symbol, "1m", str(int(start.timestamp()*1000)), str(int(end.timestamp()*1000)))

def resample_all_to_3m(symbols, client, agg_map):
    frames = []
    for sym in symbols:
        # df1m = fetch_recent_1m(client, sym)
        df1m = pd.read_pickle(Paths().ohlcv_realtime_dir+'/'+sym+'.pickle')
        df3m = M.resample_to_3m(df1m, agg_map)
        frames.append(df3m.assign(symbol=sym))
    return frames

def build_per_asset_features(df3m, sym, fwd_h, seq_len):
    d = df3m.copy()
    d[f"return"] = 100*((d["close"].shift(-fwd_h)/d["close"]) - 1)
    d[f"prev_return"] = 100*((d["close"]/d["close"].shift(seq_len)) - 1)
    d[f"volatility"] = 100*d["close"].rolling(seq_len).std().shift(-seq_len)
    d[f"prev_volatility"] = 100*d["close"].rolling(seq_len).std()
    keep = ["close","volume","numberOfTrades","prev_return","prev_volatility","return","volatility"]
    return d[["dateTime"] + keep].rename(columns={c:f"{sym}_{c}" for c in keep})

def load_artifacts():
    P = Paths(); T = TrainCfg(); Fcfg = FeatureCfg()
    meta = json.load(open(P.processed_dir + "/meta.json"))
    feat_cols = meta["feature_cols"]; cnt_cols = meta["count_cols"]
    with open(P.normalizer_pkl, "rb") as f: norm_stats = pickle.load(f)

    n_stocks = len([c for c in meta["target_cols"]])
    ts_in_dim = len(feat_cols)
    count_dim = len(cnt_cols) if cnt_cols else 0
    device = torch.device(T.device if torch.cuda.is_available() else "cpu")

    model = MarketNewsFusionModel(ts_input_dim=ts_in_dim, num_stocks=n_stocks,
                                  d_model=T.d_model, nhead=T.nhead, num_layers=T.num_layers,
                                  news_embed_dim=768, hidden_dim=T.hidden_dim, count_dim=count_dim,
                                  max_len=Fcfg.seq_len).to(device)
    model.load_state_dict(torch.load(P.weights_pt, map_location=device))
    model.eval()
    return feat_cols, cnt_cols, norm_stats, model, device

def build_window_tensor(df_merged, feat_cols, cnt_cols, seq_len):
    d = df_merged.copy().fillna(0)
    d = d.dropna(subset=["embedding"]).tail(seq_len)
    if len(d) < seq_len: return None
    X_ts = d[feat_cols].astype("float32").values[None, :, :]
    X_news = np.stack(d["embedding"].values).astype("float32")[None]
    if cnt_cols:
        X_cnt = d[cnt_cols].astype("float32").values[None, :, :]
    else:
        X_cnt = np.zeros((1, seq_len, 1), dtype="float32")
    return X_ts, X_news, X_cnt

def normalize_inplace(df, norm_stats):
    for c, st in norm_stats.items():
        if c in df.columns:
            mu = st["mean"]; sd = st["std"] if st["std"] > 1e-12 else 1.0
            df[c] = (pd.to_numeric(df[c], errors="coerce").fillna(0) - mu) / sd

def main():
    P = Paths(); NC = NewsCfg(); MC = MarketCfg(); FC = FeatureCfg(); T = TrainCfg(); PC = PortfolioCfg()

    feat_cols, cnt_cols, norm_stats, model, device = load_artifacts()
    client = M.get_client("", "")
    # client = ''
    tok = mdl = enc_device = None
    no_news_vec = None

    print("Realtime (pointwise) starting… Ctrl+C to stop.")


    # now = datetime.now(timezone.utc)
    # next_tick = round_to_next_3m(now)
    # time.sleep(max(1, (next_tick - now).total_seconds()))

    tick_ts = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    print(f"\n=== Tick at {tick_ts.isoformat()} ===")

    # Market
    per3m = resample_all_to_3m(MC.symbols_usdt, client, FC.agg_cols)
    feat_frames = []
    for d in per3m:
        sym = d["symbol"].iloc[0]
        feat_frames.append(build_per_asset_features(d.drop(columns=["symbol"]), sym, FC.fwd_h, FC.seq_len))
    merged = feat_frames[0][["dateTime"]].copy()
    for f in feat_frames: merged = merged.merge(f, on="dateTime", how="outer")
    merged = merged.sort_values("dateTime").reset_index(drop=True)

    # News
    news_csvs = [os.path.join(P.data_realtime_dir, "news", f) for f in os.listdir(P.news_realtime_dir) if f.endswith(".csv")]
    if not news_csvs:
        if no_news_vec is None:
            tok, mdl, enc_device, _ = N.load_text_encoder(NC.model_path)
            no_news_vec = N.embed_texts([NC.no_news_token], tok, mdl, enc_device,
                                        NC.max_len, NC.pooling, 1)[0]
        news_3m = pd.DataFrame(index=pd.to_datetime(merged["dateTime"]).dt.floor("3T").unique())
        news_3m["embedding"] = [no_news_vec] * len(news_3m)
        cnt_cols_full = [s[:-4] for s in MC.symbols_usdt]
        for c in cnt_cols_full: news_3m[c] = 0
        news_3m["news_count"] = 0
    else:
        df = pd.read_csv(news_csvs[0])
        need = {"releasedAt","content","asset_symbols"}
        if not need.issubset(set(df.columns)):
            raise ValueError(f"News CSV must have columns {need}")
        syms = [s.replace("USDT","") for s in MC.symbols_usdt]
        df["asset_symbols"] = (df["asset_symbols"].fillna("None").apply(
            lambda x: [t for t in str(x).replace(" ","").split(",") if t in syms]))
        df = df[df["asset_symbols"].apply(len) > 0].copy()
        now2 = datetime.now(timezone.utc)
        df["releasedAt"] = pd.to_datetime(df["releasedAt"], utc=True, errors="coerce")
        # recent = df[df["releasedAt"] >= (now2 - timedelta(hours=24))].copy()
        recent = df.copy()
        onehot = (recent["asset_symbols"].explode().str.get_dummies().groupby(level=0).sum())
        recent = recent.join(onehot)
        if tok is None or mdl is None:
            tok, mdl, enc_device, _ = N.load_text_encoder(NC.model_path)
        embs = N.embed_texts(recent["content"].tolist(), tok, mdl, enc_device,
                              NC.max_len, NC.pooling, NC.batch_size)
        recent["embedding"] = [e for e in embs]
        recent["news_count"] = 1
        no_news_vec = N.embed_texts([NC.no_news_token], tok, mdl, enc_device,
                                    NC.max_len, NC.pooling, 1)[0]
        news_3m = N.resample_news_3m(recent[["releasedAt","embedding","news_count"] +
                                [s.replace("USDT","") for s in MC.symbols_usdt]].copy(),
                                np.asarray(no_news_vec), rule=NC.rule)

    news_3m.index = news_3m.index.tz_localize(None)
    merged = F.attach_news(merged, news_3m)
    mask = merged["embedding"].isna()
    merged.loc[mask, "embedding"] = merged.loc[mask, "embedding"].apply(
        lambda _: np.asarray(no_news_vec, dtype=float).copy()
    )
    tcols = F.make_time_cols(merged); merged = pd.concat([merged, tcols], axis=1)

    merged = merged.fillna(0)
    normalize_inplace(merged, norm_stats)

    win = build_window_tensor(merged, feat_cols, cnt_cols, FC.seq_len)


    X_ts, X_news, X_cnt = win

    with torch.no_grad():
        ts_t = torch.tensor(X_ts, device=device)
        news_t = torch.tensor(X_news, device=device)
        cnt_t = torch.tensor(X_cnt, device=device)
        mu_t = model(ts_t, cnt_t, news_t)
        mu = mu_t.squeeze(0).cpu().numpy()

    # Historical moments
    assets = [c.split("_")[0] for c in feat_cols if c.endswith("_prev_return")]
    close_cols = [f"{a}_close" for a in assets if f"{a}_close" in merged.columns]
    rets = merged[["dateTime"] + close_cols].copy().set_index("dateTime").pct_change()*100.0
    rets = rets.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    lookback = 240
    if len(rets) < lookback + 1:
        print("Insufficient history for correlation; skipping weights.")
        w0 = w1 = w2 = np.zeros_like(mu)
    else:
        past = rets.tail(lookback)
        past_mu  = past.sum().to_numpy()
        past_vol = past.std().replace(0, 1e-8).to_numpy()
        hist_corr = past.corr().fillna(0).to_numpy()

        mu_mix  = (1 - PortfolioCfg().alpha_mu)*past_mu + PortfolioCfg().alpha_mu*mu
        stats = {"mean": mu_mix, "std": past_vol, "corr": hist_corr}
        w0 = method0_mvo(stats, gamma=PortfolioCfg().gamma)
        w1 = method1_small_l2(stats, gamma=PortfolioCfg().gamma, theta=PortfolioCfg().theta)
        w2 = method2_subset_bagging(stats, gamma=PortfolioCfg().gamma,
                                    num_resamples=PortfolioCfg().num_resamples,
                                    subset_size=PortfolioCfg().subset_size)

    print("mu[:5]:", np.round(mu[:5], 6))
    print("w0[:5]:", np.round(w0[:5], 4), "| w1[:5]:", np.round(w1[:5], 4), "| w2[:5]:", np.round(w2[:5], 4))

    out_row = {"timestamp": tick_ts.isoformat(), "mu": mu.tolist(),
                "weights_m0": w0.tolist(), "weights_m1": w1.tolist(), "weights_m2": w2.tolist()}
    os.makedirs(Paths().outputs_dir, exist_ok=True)
    with open(os.path.join(Paths().outputs_dir, "realtime_pointwise_log.jsonl"), "a") as f:
        f.write(json.dumps(out_row) + "\n")


if __name__ == "__main__":
    main()
