
from dataclasses import dataclass, field

@dataclass
class Paths:
    data_dir: str = "data"
    raw_news_dir: str = "data/raw_news"
    ohlcv_dir: str = "data/ohlcv"
    data_realtime_dir: str = "data_realtime"
    ohlcv_realtime_dir: str = "data_realtime/ohlcv"
    news_realtime_dir: str = "data_realtime/news"
    processed_dir: str = "data/processed"
    outputs_dir: str = "data/outputs"
    news_pickle: str = "data/processed/news_embeddings.pkl"
    merged_parquet: str = "data/processed/merged_3m.parquet"
    normalizer_pkl: str = "data/processed/normalizer.pkl"
    splits_json: str = "data/processed/splits.json"
    weights_pt: str = "data/outputs/model_weights.pt"

@dataclass
class NewsCfg:
    model_path: str = "/content/drive/MyDrive/ColabModels/bigbird-2048-final"
    batch_size: int = 32
    max_len: int = 2048
    pooling: str = "mean"
    rule: str = "3T"
    no_news_token: str = "no news at this time"

@dataclass
class MarketCfg:
    symbols_usdt: list = field(default_factory=lambda: [
        'BTCUSDT','ETHUSDT','XRPUSDT','SOLUSDT','DOGEUSDT','ADAUSDT','TRUMPUSDT',
        'SHIBUSDT','BNBUSDT','USDCUSDT','PEPEUSDT','LINKUSDT','AVAXUSDT','SUIUSDT',
        'XLMUSDT','LTCUSDT','HBARUSDT','PENGUUSDT','DOTUSDT','UNIUSDT','OPUSDT',
        'TRXUSDT','ARBUSDT','APTUSDT','TONUSDT','ATOMUSDT','BONKUSDT','AAVEUSDT',
        'BCHUSDT','ONDOUSDT'
    ])
    timeframe: str = "1m"
    start_date: str = "2025-02-26"
    end_date: str = "2025-08-13"

@dataclass
class FeatureCfg:
    agg_cols: dict = field(default_factory=lambda: {
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum", "quoteAssetVolume": "sum", "numberOfTrades": "sum",
        "takerBuyBaseVol": "sum", "takerBuyQuoteVol": "sum", "ignore": "last"
    })
    seq_len: int = 80
    horizon_steps: int = 80
    news_rule: str = "3T"

@dataclass
class TrainCfg:
    batch_size: int = 32
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    hidden_dim: int = 64
    lr: float = 1e-4
    epochs: int = 50
    patience: int = 5
    device: str = "cuda"
    top_k: int = 30
    gross: float = 1.0

@dataclass
class LossCfg:
    use_cov: bool = True
    lambda_div: float = 0.01
    lambda_net: float = 0.0
    lambda_turnover: float = 0.0

@dataclass
class BacktestCfg:
    stride: int = 80
    stoploss: int = 5
    takeprofit: int = 5
