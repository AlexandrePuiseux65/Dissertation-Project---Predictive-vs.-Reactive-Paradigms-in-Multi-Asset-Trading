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

def CalculateInputFeature(df, n, output_path):
    # EMA 
    df['EMA'] = df['close'].ewm(span=n, adjust=False).mean() 

    # HMA
    half  = wma(df['close'], n // 2)
    full  = wma(df['close'], n)
    raw   = 2 * half - full
    df['HMA'] = wma(raw, int(n ** 0.5))

    #EVWMA
    df['EVWMA'] = evwma(df, n)

    # ROC
    df['ROC'] = df['close'].pct_change(periods=n) * 100

    # RSI
    delta = df['close'].diff()
    gain= delta.where(delta > 0, 0).rolling(window=n).mean() # See if we hardcode 14 in it.
    loss= -delta.where(delta < 0, 0).rolling(window=n).mean()
    rs = gain/loss
    df['RSI']=100-(100/(1+rs))

    # William%R
    highest_high = df['high'].rolling(n).max()
    lowest_low   = df['low'].rolling(n).min()
    df['Williams_R'] = ((highest_high - df['close']) / (highest_high - lowest_low)) * -100

    # Save in the file.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=True)
    print(f"Success: {output_path} - done")

if __name__ == "__main__":
    df = TemporalSerie(os.path.join(FILE_PATH_STOCKS, "AAPL.parquet"))
    df_crypto = TemporalSerie(os.path.join(FILE_PATH_CRYPTO, "BTC-USD.parquet"))
    df_bond  = TemporalSerie(os.path.join(FILE_PATH_BONDS, "TLT.parquet"))

    # Calculate the feature.
    CalculateInputFeature(df, 14, os.path.join(FILE_PATH_STOCKS_PROCESSED, "AAPL.parquet"))
    CalculateInputFeature(df_bond, 14, os.path.join(FILE_PATH_BONDS, "TLT.parquet"))
    CalculateInputFeature(df_crypto, 14, os.path.join(FILE_PATH_CRYPTO, "BTC-USD.parquet"))

    # Test
    # df = pd.read_parquet(os.path.join(FILE_PATH_STOCKS, "AAPL.parquet"))
    # df_crypto = pd.read_parquet(os.path.join(FILE_PATH_CRYPTO, "BTC-USD.parquet"))
    # df_bonds = pd.read_parquet(os.path.join(FILE_PATH_BONDS, "TLT.parquet"))
    #print(df)
    #print(df_crypto)
    #print(df_bonds)
    
