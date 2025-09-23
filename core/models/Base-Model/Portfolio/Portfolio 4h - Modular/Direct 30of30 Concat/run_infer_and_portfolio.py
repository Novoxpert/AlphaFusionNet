
import json, numpy as np, pandas as pd, torch
from config import Paths, TrainCfg, FeatureCfg, BacktestCfg
from lib.model import MarketNewsFusionWeightModel
from lib.dataset import make_loaders
from lib.backtest_weights import backtest_weight_logits
from lib.utils import plot_equity

def main():
    P = Paths(); T = TrainCfg(); F = FeatureCfg(); B = BacktestCfg()
    df_tr = pd.read_parquet(P.processed_dir + '/train.parquet')
    df_va = pd.read_parquet(P.processed_dir + '/val.parquet')
    df_te = pd.read_parquet(P.processed_dir + '/test.parquet')
    meta = json.load(open(P.processed_dir + '/meta.json'))
    feat_cols = meta['feature_cols']; stock_list = meta['stock_list']; cnt_cols = meta['count_cols']
    _, _, te_loader = make_loaders(df_tr, df_va, df_te, F.seq_len, F.horizon_steps, feat_cols, stock_list, cnt_cols, bs=T.batch_size)
    device = torch.device(T.device if torch.cuda.is_available() else 'cpu')
    n_stocks = len(stock_list); ts_in_dim = len(feat_cols); count_dim = len(cnt_cols) if cnt_cols else 0
    model = MarketNewsFusionWeightModel(ts_input_dim=ts_in_dim, num_stocks=n_stocks,
                                        d_model=T.d_model, nhead=T.nhead, num_layers=T.num_layers,
                                        news_embed_dim=768, hidden_dim=T.hidden_dim, count_dim=count_dim,
                                        max_len=F.seq_len).to(device)
    model.load_state_dict(torch.load(P.weights_pt, map_location=device)); model.eval()
    all_logits, all_future = [], []
    with torch.no_grad():
        for b in te_loader:
            ts = b['timeseries'].to(device); news = b['news'].to(device); cnt = b['news_count'].to(device)
            Y = b['target'].cpu().numpy()
            logits = model(ts, cnt, news).cpu().numpy()
            all_logits.append(logits); all_future.append(Y)
    logits = np.vstack(all_logits)
    ret_cols = [f'{s}_target_return' for s in stock_list]
    rets_full = df_te[ret_cols].astype(float).to_numpy()
    dates = pd.to_datetime(df_te['dateTime'])
    curves = backtest_weight_logits(logits, rets_full, dates, k=T.top_k, gross=T.gross, stride=B.stride)
    plot_equity(curves['dates'], curves['equity'], out_png=Paths.outputs_dir + '/equity_weights.png')
    print('Backtest complete. Plot saved to data/outputs/equity_weights.png')

if __name__ == '__main__':
    main()
