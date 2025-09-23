
import os, numpy as np, pandas as pd, torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

def read_and_onehot_news(csv_paths: list, tradable_symbols: list) -> pd.DataFrame:
    df_list = [pd.read_pickle(p) for p in csv_paths]
    df = pd.concat(df_list, ignore_index=True)
    syms = [s.replace('USDT','') for s in tradable_symbols]
    df["asset_symbols"] = (df["asset_symbols"]
                            .fillna("None")
                            .apply(lambda x: x if type(df["asset_symbols"].iloc[0])==type([]) else [t for t in str(x).replace(" ","").split(",") if t in syms]))
    df = df[df["asset_symbols"].apply(len) > 0].copy()
    onehot = (df["asset_symbols"].explode().str.get_dummies().groupby(level=0).sum())
    for x in onehot.columns:
      if x in df.columns:
        df = df.drop(x,axis=1)
    df = df.join(onehot)
    return df.reset_index(drop=True)

def load_text_encoder(model_path: str, device: str = None, max_len: int = 2048):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_path)
    mdl = AutoModel.from_pretrained(model_path).to(device).eval()
    return tok, mdl, device, max_len

@torch.no_grad()
def embed_texts(texts, tok, mdl, device, max_len=2048, pooling="mean", batch_size=32):
    out = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding news"):
        batch = texts[i:i+batch_size]
        enc = tok(batch, truncation=True, padding=True, max_length=max_len, return_tensors="pt").to(device)
        hs = mdl(**enc).last_hidden_state
        if pooling == "mean":
            mask = enc["attention_mask"].unsqueeze(-1)
            pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            pooled = hs[:, 0, :]
        out.append(pooled.detach().cpu().numpy())
        if device == "cuda": torch.cuda.empty_cache()
    return np.vstack(out)

def resample_news_3m(df_news: pd.DataFrame, no_news_vec: np.ndarray, rule: str = "3T") -> pd.DataFrame:
    df = df_news.copy()
    df["releasedAt"] = pd.to_datetime(df["releasedAt"], errors="coerce")
    df = df.dropna(subset=["releasedAt"])
    df["t3"] = df["releasedAt"].dt.floor(rule)

    base = {"releasedAt","content","embedding","news_count","asset_symbols","t3"}
    asset_cols = [c for c in df.columns if c not in base and df[c].dtype != "O"]

    if "news_count" not in df.columns: df["news_count"] = 1

    g = df.groupby("t3", sort=True)
    agg = g[asset_cols + ["news_count"]].sum()

    def _avg(arr):
        arr = [np.asarray(x, dtype=float) for x in arr if isinstance(x, (list, np.ndarray))]
        if not arr: return None
        Ls = {a.shape for a in arr}
        if len(Ls) != 1: raise ValueError(f"Embedding length mismatch {Ls}")
        return np.mean(np.stack(arr, 0), 0)

    emb = g["embedding"].apply(_avg)
    news_3m = agg.join(emb.rename("embedding"))

    idx = pd.date_range(news_3m.index.min(), news_3m.index.max(), freq=rule)
    news_3m = news_3m.reindex(idx)
    news_3m["news_count"] = news_3m["news_count"].fillna(0).astype(int)
    for c in asset_cols: news_3m[c] = news_3m[c].fillna(0).astype(int)
    # mask = news_3m["embedding"].isna()
    # news_3m.loc[mask, "embedding"] = [np.asarray(no_news_vec, dtype=float)] * mask.sum()
  
    mask = news_3m["embedding"].isna()
    news_3m.loc[mask, "embedding"] = news_3m.loc[mask, "embedding"].apply(
        lambda _: np.asarray(no_news_vec, dtype=float).copy()
    )

    news_3m.index.name = "t3"
    return news_3m
