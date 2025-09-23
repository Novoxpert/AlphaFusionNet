
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader

class WeightDataset(Dataset):
    def __init__(self, df, seq_len, horizon_steps, feature_cols, stock_list, news_vec_col='embedding', news_count_cols=None):
        self.df = df.reset_index(drop=True)
        self.seq_len = seq_len
        self.h = horizon_steps
        self.fcols = feature_cols
        self.news_col = news_vec_col
        self.count_cols = news_count_cols or []
        self.stock_ret_cols = [f'{s}_target_return' for s in stock_list]
        self.df[self.news_col] = self.df[self.news_col].apply(
            lambda x: np.asarray(x, dtype='float32') if not isinstance(x, np.ndarray) else x
        )
        self.N = max(len(self.df) - self.seq_len - self.h, 0)

    def __len__(self): return self.N

    def __getitem__(self, idx):
        lo, hi = idx, idx + self.seq_len
        fut_lo, fut_hi = hi, hi + self.h
        X_ts = self.df.loc[lo:hi-1, self.fcols].astype('float32').values
        X_news = np.stack(self.df.loc[lo:hi-1, self.news_col].values).astype('float32')
        if self.count_cols:
            X_cnt = self.df.loc[lo:hi-1, self.count_cols].astype('float32').values
        else:
            X_cnt = np.zeros((self.seq_len, 1), dtype='float32')
        Y = self.df.loc[fut_lo:fut_hi-1, self.stock_ret_cols].astype('float32').values
        return {'timeseries': torch.tensor(X_ts),
                'news': torch.tensor(X_news),
                'news_count': torch.tensor(X_cnt),
                'target': torch.tensor(Y)}

def make_loaders(df_tr, df_va, df_te, seq_len, horizon_steps, feature_cols, stock_list, news_count_cols, bs):
    ds_tr = WeightDataset(df_tr, seq_len, horizon_steps, feature_cols, stock_list, news_count_cols=news_count_cols)
    ds_va = WeightDataset(df_va, seq_len, horizon_steps, feature_cols, stock_list, news_count_cols=news_count_cols)
    ds_te = WeightDataset(df_te, seq_len, horizon_steps, feature_cols, stock_list, news_count_cols=news_count_cols)
    return (DataLoader(ds_tr, batch_size=bs, shuffle=True),
            DataLoader(ds_va, batch_size=bs, shuffle=False),
            DataLoader(ds_te, batch_size=bs, shuffle=False))
