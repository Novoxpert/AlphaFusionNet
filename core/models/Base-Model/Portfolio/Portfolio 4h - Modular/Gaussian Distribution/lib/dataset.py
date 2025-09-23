
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader

class NewsTimeSeriesDataset(Dataset):
    def __init__(self, df, seq_len, feature_cols, target_cols, news_vec_col="embedding", news_count_cols=None):
        self.df = df.reset_index(drop=True)
        self.seq_len = seq_len
        self.fcols = feature_cols
        self.tcols = target_cols
        self.news_col = news_vec_col
        self.count_cols = news_count_cols or []

        self.df[self.news_col] = self.df[self.news_col].apply(
            lambda x: np.asarray(x, dtype="float32") if not isinstance(x, np.ndarray) else x
        )
        self.N = len(self.df) - self.seq_len - 1
        self.N = max(self.N, 0)

    def __len__(self): return self.N

    def __getitem__(self, idx):
        lo, hi = idx, idx + self.seq_len
        X_ts = self.df.loc[lo:hi-1, self.fcols].astype("float32").values
        X_news = np.stack(self.df.loc[lo:hi-1, self.news_col].values).astype("float32")
        if self.count_cols:
            X_cnt = self.df.loc[lo:hi-1, self.count_cols].astype("float32").values
        else:
            X_cnt = np.zeros((self.seq_len, 1), dtype="float32")
        y = self.df.loc[hi-1, self.tcols].astype("float32").values
        return {"timeseries": torch.tensor(X_ts),
                "news": torch.tensor(X_news),
                "news_count": torch.tensor(X_cnt),
                "target": torch.tensor(y)}

def make_loaders(df_tr, df_va, df_te, seq_len, feature_cols, target_cols, news_count_cols, bs):
    ds_tr = NewsTimeSeriesDataset(df_tr, seq_len, feature_cols, target_cols, news_count_cols=news_count_cols)
    ds_va = NewsTimeSeriesDataset(df_va, seq_len, feature_cols, target_cols, news_count_cols=news_count_cols)
    ds_te = NewsTimeSeriesDataset(df_te, seq_len, feature_cols, target_cols, news_count_cols=news_count_cols)
    return (DataLoader(ds_tr, batch_size=bs, shuffle=True),
            DataLoader(ds_va, batch_size=bs, shuffle=False),
            DataLoader(ds_te, batch_size=bs, shuffle=False))
