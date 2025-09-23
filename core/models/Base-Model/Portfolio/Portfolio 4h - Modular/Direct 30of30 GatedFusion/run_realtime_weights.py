
import os, time, json, pickle, numpy as np, pandas as pd, torch
from datetime import datetime, timezone, timedelta
from config import Paths, NewsCfg, MarketCfg, FeatureCfg, TrainCfg, BacktestCfg
from lib import market as M, features as F, news as N
from lib.model import MarketNewsFusionWeightModel
from lib.loss_weights import weights_long_short_topk_abs
import warnings
warnings.filterwarnings('ignore')
def round_to_next_3m(ts):
    minute = (ts.minute // 3) * 3
    base = ts.replace(minute=minute, second=0, microsecond=0)
    if base < ts: base += timedelta(minutes=3)
    return base

def fetch_recent_1m(client, symbol, minutes=24*60):
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=minutes+5)
    return M.fetch_klines(client, symbol, '1m', str(int(start.timestamp()*1000)), str(int(end.timestamp()*1000)))

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
    d[f'return'] = 100*((d['close'].shift(-1)/d['close']) - 1)
    d[f'prev_return'] = 100*((d['close']/d['close'].shift(seq_len)) - 1)
    d[f'volatility'] = 100*d['close'].rolling(seq_len).std().shift(-seq_len)
    d[f'prev_volatility'] = 100*d['close'].rolling(seq_len).std()
    keep = ['close','volume','numberOfTrades','prev_return','prev_volatility','return','volatility']
    return d[['dateTime'] + keep].rename(columns={c:f'{sym}_{c}' for c in keep})

def load_artifacts():
    P = Paths(); T = TrainCfg(); Fcfg = FeatureCfg()
    meta = json.load(open(P.processed_dir + '/meta.json'))
    feat_cols = meta['feature_cols']; cnt_cols = meta['count_cols']; stock_list = meta['stock_list']
    with open(P.normalizer_pkl, 'rb') as f: norm_stats = pickle.load(f)
    n_stocks = len(stock_list); ts_in_dim = len(feat_cols); count_dim = len(cnt_cols) if cnt_cols else 0
    device = torch.device(T.device if torch.cuda.is_available() else 'cpu')
    model = MarketNewsFusionWeightModel(ts_input_dim=ts_in_dim, num_stocks=n_stocks,
                                        d_model=T.d_model, nhead=T.nhead, num_layers=T.num_layers,
                                        news_embed_dim=768, hidden_dim=T.hidden_dim, count_dim=count_dim,
                                        max_len=Fcfg.seq_len).to(device)
    model.load_state_dict(torch.load(P.weights_pt, map_location=device)); model.eval()
    return feat_cols, cnt_cols, stock_list, norm_stats, model, device

def normalize_inplace(df, norm_stats):
    for c, st in norm_stats.items():
        if c in df.columns:
            mu = st['mean']; sd = st['std'] if st['std'] > 1e-12 else 1.0
            df[c] = (pd.to_numeric(df[c], errors='coerce').fillna(0) - mu) / sd


def main():
    P = Paths(); NC = NewsCfg(); MC = MarketCfg(); FC = FeatureCfg(); T = TrainCfg(); B = BacktestCfg()
    feat_cols, cnt_cols, stock_list, norm_stats, model, device = load_artifacts()
    client = M.get_client('', '')
    tok = mdl = enc_device = None; no_news_vec = None
    print('Realtime (direct weights) starting… Ctrl+C to stop.')


    # now = datetime.now(timezone.utc); next_tick = round_to_next_3m(now)
    # time.sleep(max(1, (next_tick - now).total_seconds()))
    tick_ts = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    print(f'\n=== Tick at {tick_ts.isoformat()} ===')
    per3m = resample_all_to_3m(MC.symbols_usdt, client, FC.agg_cols)
    feat_frames = []
    for d in per3m:
        sym = d['symbol'].iloc[0]
        feat_frames.append(build_per_asset_features(d.drop(columns=['symbol']), sym, FC.horizon_steps, FC.seq_len))
    merged = feat_frames[0][['dateTime']].copy()
    for f in feat_frames: merged = merged.merge(f, on='dateTime', how='outer')
    merged = merged.sort_values('dateTime').reset_index(drop=True)
    # News
    news_csvs = [os.path.join(P.data_realtime_dir, 'news', f) for f in os.listdir(P.news_realtime_dir) if f.endswith('.csv')]
    if not news_csvs:
        if no_news_vec is None:
            tok, mdl, enc_device, _ = N.load_text_encoder(NC.model_path)
            no_news_vec = N.embed_texts([NC.no_news_token], tok, mdl, enc_device, NC.max_len, NC.pooling, 1)[0]
        news_3m = pd.DataFrame(index=pd.to_datetime(merged['dateTime']).dt.floor('3T').unique())
        news_3m['embedding'] = [no_news_vec] * len(news_3m)
        cnt_cols_full = [s[:-4] for s in MC.symbols_usdt]
        for c in cnt_cols_full: news_3m[c] = 0
        news_3m['news_count'] = 0
    else:
        df = pd.read_csv(news_csvs[0])
        need = {'releasedAt','content','asset_symbols'}
        if not need.issubset(set(df.columns)): raise ValueError(f'News CSV must have columns {need}')
        syms = [s.replace('USDT','') for s in MC.symbols_usdt]
        df['asset_symbols'] = (df['asset_symbols'].fillna('None').apply(lambda x: [t for t in str(x).replace(' ','').split(',') if t in syms]))
        df = df[df['asset_symbols'].apply(len) > 0].copy()
        now2 = datetime.now(timezone.utc)
        df['releasedAt'] = pd.to_datetime(df['releasedAt'], utc=True, errors='coerce')
        # recent = df[df['releasedAt'] >= (now2 - timedelta(hours=24))].copy()
        recent = df.copy()
        onehot = (recent['asset_symbols'].explode().str.get_dummies().groupby(level=0).sum())
        recent = recent.join(onehot)
        if tok is None or mdl is None:
            tok, mdl, enc_device, _ = N.load_text_encoder(NC.model_path)
        embs = N.embed_texts(recent['content'].tolist(), tok, mdl, enc_device, NC.max_len, NC.pooling, NC.batch_size)
        recent['embedding'] = [e for e in embs]
        recent['news_count'] = 1
        no_news_vec = N.embed_texts([NC.no_news_token], tok, mdl, enc_device, NC.max_len, NC.pooling, 1)[0]
        news_3m = N.resample_news_3m(recent[['releasedAt','embedding','news_count'] +
                                [s.replace('USDT','') for s in MC.symbols_usdt]].copy(),
                                np.asarray(no_news_vec), rule=NC.rule)
    # merged = F.attach_news(merged, news_3m)
    news_3m.index = news_3m.index.tz_localize(None)
    merged = F.attach_news(merged, news_3m)
    mask = merged["embedding"].isna()
    merged.loc[mask, "embedding"] = merged.loc[mask, "embedding"].apply(
        lambda _: np.asarray(no_news_vec, dtype=float).copy()
    )

    tcols = F.make_time_cols(merged); merged = pd.concat([merged, tcols], axis=1)
    merged = merged.fillna(0); normalize_inplace(merged, norm_stats)
    d = merged.copy().dropna(subset=['embedding']).tail(FC.seq_len)
    # if len(d) < FC.seq_len:
    #     print('Not enough rows yet for a window…'); continue
    X_ts = d[feat_cols].astype('float32').values[None, :, :]
    X_news = np.stack(d['embedding'].values).astype('float32')[None]
    if cnt_cols:
        X_cnt = d[cnt_cols].astype('float32').values[None, :, :]
    else:
        X_cnt = np.zeros((1, FC.seq_len, 1), dtype='float32')
    with torch.no_grad():
        ts_t = torch.tensor(X_ts, device=device); news_t = torch.tensor(X_news, device=device); cnt_t = torch.tensor(X_cnt, device=device)
        logits_t = model(ts_t, cnt_t, news_t)
        w = weights_long_short_topk_abs(logits_t, k=TrainCfg().top_k, gross=TrainCfg().gross).squeeze(0).cpu().numpy()
    print('Top-5 abs weights preview:', np.round(w[np.argsort(-np.abs(w))[:5]], 4))
    out_row = {'timestamp': tick_ts.isoformat(), 'weights': w.tolist()}
    os.makedirs(Paths().outputs_dir, exist_ok=True)
    with open(os.path.join(Paths().outputs_dir, 'realtime_weights_log.jsonl'), 'a') as f:
        f.write(json.dumps(out_row) + '\n')


if __name__ == '__main__':
    main()
