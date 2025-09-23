
import os, pandas as pd
from binance.client import Client
from binance.enums import HistoricalKlinesType

def get_client(api_key: str = "", api_secret: str = ""):
    return Client(api_key, api_secret)

def fetch_klines(client, symbol, interval, start, end):
    kl = client.get_historical_klines(symbol, interval, start, end,
                                      klines_type=HistoricalKlinesType.SPOT)
    cols = ['dateTime','open','high','low','close','volume','closeTime',
            'quoteAssetVolume','numberOfTrades','takerBuyBaseVol','takerBuyQuoteVol','ignore']
    df = pd.DataFrame(kl, columns=cols)
    df["dateTime"] = pd.to_datetime(df["dateTime"], unit="ms")
    df["closeTime"] = pd.to_datetime(df["closeTime"], unit="ms")
    num_cols = ['open','high','low','close','volume','quoteAssetVolume',
                'numberOfTrades','takerBuyBaseVol','takerBuyQuoteVol']
    for c in num_cols: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def cache_symbol(df, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_pickle(out_path)

def resample_to_3m(df, agg_map):
    return (df.set_index("dateTime")
              .resample("3T")
              .agg(agg_map)
              .reset_index())
