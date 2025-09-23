
import os, json, pickle, numpy as np, pandas as pd

def add_targets_and_features(df3m, fwd_h, seq_len, per_asset_features, symbol):
    df = df3m.copy()
    df["return"] = 100*((df["close"].shift(-fwd_h)/df["close"]) - 1)
    df["prev_return"] = 100*((df["close"]/df["close"].shift(seq_len)) - 1)
    df["volatility"] = 100*df["close"].rolling(seq_len).std().shift(-seq_len)
    df["prev_volatility"] = 100*df["close"].rolling(seq_len).std()
    df["close_main"] = df["close"].copy()
    keep = {"close_main", "close","volume","numberOfTrades","prev_return","prev_volatility","return","volatility"}
    out = (df[["dateTime"] + list(keep)].rename(columns={c:f"{symbol}_{c}" for c in keep}))
    out[f"{symbol}_target_return"] = out[f"{symbol}_return"]
    return out

def merge_assets(dfs: list):
    base = dfs[0][["dateTime"]].copy()
    for d in dfs:
        base = base.merge(d, on="dateTime", how="outer")
    base = base.sort_values("dateTime").reset_index(drop=True)
    return base

def attach_news(merged, news_3m):
    merged["t3"] = pd.to_datetime(merged["dateTime"]).dt.floor("3T")
    news_for_mkt = news_3m.reset_index().rename(columns={"t3":"t3_news"})
    out = merged.merge(news_for_mkt, left_on="t3", right_on="t3_news", how="left")
    out = out.drop(columns=["t3_news"])
    return out

def make_time_cols(df):
    dt = pd.to_datetime(df["dateTime"])
    return pd.DataFrame({
        "month": dt.dt.month.values,
        "day": dt.dt.day.values,
        "weekday": dt.dt.weekday.values,
        "hour": dt.dt.hour.values,
        "minute": dt.dt.minute.values
    })

def train_val_test_by_day(df, train=0.77, val=0.11):
    day = pd.to_datetime(df["dateTime"]).dt.date
    uniq = sorted(day.unique())
    n = len(uniq)
    tr = uniq[:int(train*n)]
    va = uniq[int(train*n):int((train+val)*n)]
    te = uniq[int((train+val)*n):]
    return tr, va, te

def normalize_train_val_test(df, feature_cols, split_days, save_path):
    tr_days, va_days, te_days = split_days
    day = pd.to_datetime(df["dateTime"]).dt.date
    df_tr = df[day.isin(tr_days)].copy()
    df_va = df[day.isin(va_days)].copy()
    df_te = df[day.isin(te_days)].copy()

    stats = {}
    for c in feature_cols:
        mu = float(pd.to_numeric(df_tr[c], errors="coerce").fillna(0).mean())
        sd = float(pd.to_numeric(df_tr[c], errors="coerce").fillna(0).std())
        sd = sd if sd > 1e-12 else 1.0
        stats[c] = {"mean":mu, "std":sd}
        for d in (df_tr, df_va, df_te):
            d[c] = (pd.to_numeric(d[c], errors="coerce").fillna(0) - mu) / sd

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f: pickle.dump(stats, f)
    return df_tr, df_va, df_te, stats
