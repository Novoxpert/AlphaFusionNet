
import json, numpy as np, pandas as pd, torch
from config import Paths, TrainCfg, FeatureCfg, PortfolioCfg, MarketCfg
from lib.model import MarketNewsFusionModel
from lib.dataset import make_loaders
from lib.inference import infer_full
from lib.portfolio import rolling_backtest_univariate
from lib.utils import plot_portfolios

def main():
    P = Paths(); T = TrainCfg(); F = FeatureCfg(); PC = PortfolioCfg()
    MC = MarketCfg() 
    FC = FeatureCfg()
    
    df_tr = pd.read_parquet(P.processed_dir + "/train.parquet")
    df_va = pd.read_parquet(P.processed_dir + "/val.parquet")
    df_te = pd.read_parquet(P.processed_dir + "/test.parquet")
    meta = json.load(open(P.processed_dir + "/meta.json"))
    feat_cols = meta["feature_cols"]; tgt_cols = meta["target_cols"]; cnt_cols = meta["count_cols"]
    symbols = MC.symbols_usdt
    
    
    _, _, te_loader = make_loaders(df_tr, df_va, df_te, F.seq_len, feat_cols, tgt_cols, cnt_cols, bs=T.batch_size)
    device = torch.device(T.device if torch.cuda.is_available() else "cpu")

    n_stocks = len(tgt_cols)
    ts_in_dim = len(feat_cols)
    count_dim = len(cnt_cols) if cnt_cols else 0

    model = MarketNewsFusionModel(ts_input_dim=ts_in_dim, num_stocks=n_stocks,
                                  d_model=T.d_model, nhead=T.nhead, num_layers=T.num_layers,
                                  news_embed_dim=768, hidden_dim=T.hidden_dim, count_dim=count_dim,
                                  max_len=F.seq_len).to(device)

    model.load_state_dict(torch.load(P.weights_pt, map_location=device))

    mu, sigma, y = infer_full(model, te_loader, device=device)   # [T,S], [T,S], [T,S]

    te_idx = pd.to_datetime(df_te["dateTime"]).reset_index(drop=True)
    dates = te_idx.iloc[F.seq_len+1 : F.seq_len+1 + mu.shape[0]].values
    y_main = 100*df_te[[x+'_close_main' for x in symbols]].pct_change().fillna(0).iloc[F.seq_len+1 : F.seq_len+1 + mu.shape[0]].values

    curves = rolling_backtest_univariate(pred_mu=mu, pred_std=sigma,
                                         realized_returns=y_main, dates=pd.DatetimeIndex(dates),
                                         alpha=PC.alpha, gamma=PC.gamma, theta=PC.theta,
                                         num_resamples=PC.num_resamples, subset_size=PC.subset_size,
                                         lookback_minutes=FC.fwd_h, step=FC.fwd_h)

    plot_portfolios(curves, out_png=Paths.outputs_dir + "/equity_curves_univariate.png")
    print("Done. Curves plotted to data/outputs/equity_curves_univariate.png")

if __name__ == "__main__":
    main()
