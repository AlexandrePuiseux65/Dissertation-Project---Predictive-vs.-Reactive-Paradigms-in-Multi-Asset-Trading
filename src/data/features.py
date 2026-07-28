"""
    This script has for goal to compute most of the indicator needed.
"""

import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FILE_PATH_STOCKS = os.path.join(BASE_DIR, "data", "raw", "stocks")
FILE_PATH_CRYPTO = os.path.join(BASE_DIR, "data", "raw", "crypto")
FILE_PATH_BONDS = os.path.join(BASE_DIR, "data", "raw", "bonds")

FILE_PATH_STOCKS_PROCESSED = os.path.join(BASE_DIR, "data", "processed", "stocks")
FILE_PATH_CRYPTO_PROCESSED = os.path.join(BASE_DIR, "data", "processed", "crypto")
FILE_PATH_BONDS_PROCESSED  = os.path.join(BASE_DIR, "data", "processed", "bonds")

def TemporalSerie(link, timeframe='1h'):
    '''
        Recalulate data on a different timeframe of 1h.
        Usfulle since the bots are trading on a 1h timeframe.
    '''
    df = pd.read_parquet(link)
    df = df.set_index('timestamp')
    df = df.sort_index()
    
    df = df.resample(timeframe).agg({
        'open':        'first',
        'high':        'max',
        'low':         'min',
        'close':       'last',
        'volume':      'sum',
        'trade_count': 'sum',
        'vwap':        'mean'
    }).dropna()
    
    return df

def wma(series, n):
    '''
        Compute the WMA, in order to find the HMA.
    '''
    weights = pd.Series(range(1, n + 1))
    return series.rolling(n).apply(lambda x: (x * weights).sum() / weights.sum(), raw=True)

def evwma(df, n):
    '''
        Compute the EVWMA for one candle.
    '''
    result = [float('nan')] * len(df['close'])
    capsum = df['volume'].rolling(n).sum()
    for t in range(n, len(df['close'])):
        cap = capsum.iloc[t]
        if t == n:
            result[t] = df['close'].iloc[t]
        else:
            result[t] = (df['volume'].iloc[t] * df['close'].iloc[t] + (cap - df['volume'].iloc[t]) * result[t-1]) / cap
    return pd.Series(result, index=df['close'].index)

def CalculateInputFeature(df, output_path):
    # SMA 20 and 50 periods
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()

    # RSI (14 periods)
    delta = df['close'].diff()
    gain  = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss  = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs    = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9)
    ema_12       = df['close'].ewm(span=12, adjust=False).mean()
    ema_26       = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD']   = ema_12 - ema_26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist']   = df['MACD'] - df['MACD_signal']

    # Bollinger Bands (20 periods, 2 std)
    sma_20          = df['close'].rolling(window=20).mean()
    std_20          = df['close'].rolling(window=20).std()
    df['BB_middle'] = sma_20
    df['BB_upper']  = sma_20 + 2 * std_20
    df['BB_lower']  = sma_20 - 2 * std_20

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.reset_index().to_parquet(output_path, index=False)
    print(f"Success: {output_path} - done")

if __name__ == "__main__":
    # Get raw data.
    df = TemporalSerie(os.path.join(FILE_PATH_STOCKS, "AAPL.parquet"))
    df_crypto = TemporalSerie(os.path.join(FILE_PATH_CRYPTO, "BTC-USD.parquet"))
    df_bond  = TemporalSerie(os.path.join(FILE_PATH_BONDS, "TLT.parquet"))

    # Compute the feature.
    CalculateInputFeature(df, os.path.join(FILE_PATH_STOCKS_PROCESSED, "AAPL.parquet"))
    CalculateInputFeature(df_bond, os.path.join(FILE_PATH_BONDS_PROCESSED, "TLT.parquet"))
    CalculateInputFeature(df_crypto, os.path.join(FILE_PATH_CRYPTO_PROCESSED, "BTC-USD.parquet"))

    # Verification of the data and add the type of assets (for the trainning in multi-asset).
    df = pd.read_parquet(os.path.join(FILE_PATH_STOCKS_PROCESSED, "AAPL.parquet"))
    df_crypto = pd.read_parquet(os.path.join(FILE_PATH_CRYPTO_PROCESSED, "BTC-USD.parquet"))
    df_bonds = pd.read_parquet(os.path.join(FILE_PATH_BONDS_PROCESSED, "TLT.parquet"))
    df['types'] = 0
    df_crypto['types'] = 1
    df_bonds['types'] = 2

    print(f"Stocks; {df.size}",df.columns)
    print(f"Crypto; {df_crypto.size}",df_crypto.columns)
    print(f"Bonds; {df_bonds.size}",df_bonds.columns)
    
