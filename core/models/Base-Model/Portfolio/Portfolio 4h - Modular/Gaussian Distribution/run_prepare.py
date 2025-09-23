
import os, json, numpy as np, pandas as pd
from config import Paths, NewsCfg, MarketCfg, FeatureCfg
from lib import news as N, market as M, features as F, utils as U

def main():
    U.set_seed(123)
    P = Paths(); NC = NewsCfg(); MC = MarketCfg(); FC = FeatureCfg()

    tradable = MC.symbols_usdt
    news_dir = os.path.join(P.data_dir, "raw_news")
    csvs = [os.path.join(news_dir, f) for f in os.listdir(news_dir) if f.endswith(".pickle")]
    if not csvs:
        raise FileNotFoundError(f"No CSVs found in {news_dir}. Place your news CSVs there.")

    df_news = N.read_and_onehot_news(csvs, tradable)

    if os.path.exists(P.news_pickle):
        df_news = pd.read_pickle(P.news_pickle)
        print("Loaded cached news embeddings.")
        no_news_vec = df_news["embedding"].iloc[0] * 0
    else:
        tok, mdl, device, max_len = N.load_text_encoder(NC.model_path)
        emb = N.embed_texts(df_news["content"].tolist(), tok, mdl, device, max_len, NC.pooling, NC.batch_size)
        df_news["embedding"] = [e for e in emb]
        no_news_vec = N.embed_texts([NC.no_news_token], tok, mdl, device, max_len, NC.pooling, batch_size=1)[0]
        df_news["news_count"] = 1
        os.makedirs(os.path.dirname(P.news_pickle), exist_ok=True)
        df_news.to_pickle(P.news_pickle)
        print("Saved news embeddings.")

    news_3m = N.resample_news_3m(df_news[["releasedAt","embedding","news_count"] +
                            [s.replace("USDT","") for s in tradable if s.endswith("USDT")]].copy(),
                            np.asarray(no_news_vec), rule=NC.rule)

    # client = M.get_client("", "")
    per_asset = []
    for sym in MC.symbols_usdt:
        fpath = os.path.join(P.ohlcv_dir, f"{sym}.pickle")
        if os.path.exists(fpath):
            df1m = pd.read_pickle(fpath)
        else:
            df1m = M.fetch_klines(client, sym, MC.timeframe, MC.start_date, MC.end_date)
            M.cache_symbol(df1m, fpath)
        df3m = M.resample_to_3m(df1m, FC.agg_cols)
        per_asset.append(F.add_targets_and_features(df3m, FC.fwd_h, FC.seq_len, FC.per_asset_features, sym))

    merged = F.merge_assets(per_asset)
    merged = F.attach_news(merged, news_3m)

    tcols = F.make_time_cols(merged)
    merged = pd.concat([merged, tcols], axis=1)
    os.makedirs(os.path.dirname(P.merged_parquet), exist_ok=True)
    merged.to_parquet(P.merged_parquet, index=False)
    print("Saved merged 3m parquet.")

    tr_days, va_days, te_days = F.train_val_test_by_day(merged)
    with open(P.splits_json, "w") as f: json.dump({"train": [str(x) for x in tr_days],
                                                  "val": [str(x) for x in va_days],
                                                  "test": [str(x) for x in te_days]}, f)

    all_cols = [c for c in merged.columns if any(c.endswith(x) for x in ["close","volume","numberOfTrades",
                                                                         "prev_return","prev_volatility","return","volatility"])]
    target_cols = [c for c in merged.columns if c.endswith("_target_return")]
    feat_cols = [c for c in all_cols if (("return" not in c) or ("prev_return" in c))]
    feat_cols = [c for c in feat_cols if (("volatility" not in c) or ("prev_volatility" in c))]

    count_cols = [c for c in merged.columns if c in [s[:-4] for s in MC.symbols_usdt]]

    df_tr, df_va, df_te, stats = F.normalize_train_val_test(merged.fillna(0),
                                                            feature_cols=feat_cols,
                                                            split_days=(tr_days, va_days, te_days),
                                                            save_path=P.normalizer_pkl)
    df_tr.to_parquet(P.processed_dir + "/train.parquet", index=False)
    df_va.to_parquet(P.processed_dir + "/val.parquet", index=False)
    df_te.to_parquet(P.processed_dir + "/test.parquet", index=False)
    with open(P.processed_dir + "/meta.json","w") as f:
        json.dump({"feature_cols":feat_cols, "target_cols":target_cols, "count_cols":count_cols}, f)

if __name__ == "__main__":
    main()
