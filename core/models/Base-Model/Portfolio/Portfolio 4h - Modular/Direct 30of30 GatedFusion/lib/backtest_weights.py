import numpy as np, pandas as pd
from lib.loss_weights import weights_long_short_topk_abs
import torch
from config import MarketCfg

def backtest_weight_logits(pred_logits, returns_matrix, dates, k=30, gross=1.0, stride=80):
    symbols = MarketCfg().symbols_usdt

    M, N = pred_logits.shape
    T = returns_matrix.shape[0]
    assert T == len(dates)
    with torch.no_grad():
        logits_t = torch.tensor(pred_logits, dtype=torch.float32)
        w_t = weights_long_short_topk_abs(logits_t, k=k, gross=gross).cpu().numpy()
    w_series = np.zeros((T, N), dtype=float)
    idxs = list(range(0, T, stride))
    for j, t0 in enumerate(idxs):
        w_series[t0:, :] = w_t[min(j, M-1)]
    rp = (w_series * returns_matrix).sum(axis=1)
    equity = rp.cumsum()

    df_portfolio = pd.DataFrame()
    df_portfolio['dateTime'] = dates
    for sym in symbols:
        df_portfolio[sym+'_return'] = returns_matrix[:, symbols.index(sym)]
        df_portfolio[sym+'_weight'] = w_series[:, symbols.index(sym)]
    df_portfolio.to_pickle('data/outputs/df_portfolio.pickle')

    return {'dates': dates, 'equity': equity, 'rp': rp, 'weights': w_series}

