
import json, pandas as pd, torch
from config import Paths, TrainCfg, FeatureCfg
from lib.dataset import make_loaders
from lib.model import MarketNewsFusionModel
from lib.train import train_loop

def main():
    P = Paths(); T = TrainCfg(); F = FeatureCfg()

    df_tr = pd.read_parquet(P.processed_dir + "/train.parquet")
    df_va = pd.read_parquet(P.processed_dir + "/val.parquet")
    df_te = pd.read_parquet(P.processed_dir + "/test.parquet")
    meta = json.load(open(P.processed_dir + "/meta.json"))
    feat_cols = meta["feature_cols"]; tgt_cols = meta["target_cols"]; cnt_cols = meta["count_cols"]

    tr_loader, va_loader, te_loader = make_loaders(df_tr, df_va, df_te,
                                                   F.seq_len, feat_cols, tgt_cols, cnt_cols,
                                                   bs=T.batch_size)
    device = torch.device(T.device if torch.cuda.is_available() else "cpu")

    n_stocks = len(tgt_cols)
    ts_in_dim = len(feat_cols)
    count_dim = len(cnt_cols) if cnt_cols else 0

    model = MarketNewsFusionModel(ts_input_dim=ts_in_dim, num_stocks=n_stocks,
                                  d_model=T.d_model, nhead=T.nhead, num_layers=T.num_layers,
                                  news_embed_dim=768, hidden_dim=T.hidden_dim, count_dim=count_dim,
                                  max_len=F.seq_len).to(device)

    best = train_loop(model, (tr_loader, va_loader, te_loader),
                      device=device, epochs=T.epochs, patience=T.patience,
                      lr=T.lr, save_path=P.weights_pt)
    print("Best val loss:", best)

if __name__ == "__main__":
    main()
