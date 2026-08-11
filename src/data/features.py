"""
    This script computes the technical indicators required for the
    processed dataset (SMA, RSI, MACD, Bollinger Bands), starting from
    raw OHLCV data resampled to an hourly timeframe.
"""

import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FILE_PATH_STOCKS = os.path.join(BASE_DIR, "data", "raw", "stocks")
FILE_PATH_CRYPTO = os.path.join(BASE_DIR, "data", "raw", "crypto")
FILE_PATH_BONDS = os.path.join(BASE_DIR, "data", "raw", "bonds")

FILE_PATH_STOCKS_PROCESSED = os.path.join(BASE_DIR, "data", "processed", "stocks")
FILE_PATH_CRYPTO_PROCESSED = os.path.join(BASE_DIR, "data", "processed", "crypto")
FILE_PATH_BONDS_PROCESSED  = os.path.join(BASE_DIR, "data", "processed", "bonds")

def ResampleTimeframe(link, timeframe='1h'):
    '''
        Resample raw OHLCV data to the given timeframe (default: 1h),
        using proper OHLC aggregation (first/max/min/last) and summing volume/trade_count.
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

def CalculateInputFeature(df, output_path):
    '''
        Compute the technical indicators used as model features (SMA_20,
        SMA_50, RSI(14), MACD(12,26,9), Bollinger Bands(20, 2 std)) and
        save the resulting DataFrame to output_path as a parquet file.
    '''
    # SMA 20 and 50 periods
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()

    # RSI (14 periods)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9)
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']

    # Bollinger Bands (20 periods, 2 std)
    sma_20 = df['close'].rolling(window=20).mean()
    std_20 = df['close'].rolling(window=20).std()
    df['BB_middle'] = sma_20
    df['BB_upper'] = sma_20 + 2 * std_20
    df['BB_lower'] = sma_20 - 2 * std_20

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.reset_index().to_parquet(output_path, index=False)
    print(f"Success: {output_path} - done")

if __name__ == "__main__":
    # Get raw data
    df = ResampleTimeframe(os.path.join(FILE_PATH_STOCKS, "AAPL.parquet"))
    df_crypto = ResampleTimeframe(os.path.join(FILE_PATH_CRYPTO, "BTC-USD.parquet"))
    df_bond  = ResampleTimeframe(os.path.join(FILE_PATH_BONDS, "TLT.parquet"))

    # Compute the feature
    CalculateInputFeature(df, os.path.join(FILE_PATH_STOCKS_PROCESSED, "AAPL.parquet"))
    CalculateInputFeature(df_bond, os.path.join(FILE_PATH_BONDS_PROCESSED, "TLT.parquet"))
    CalculateInputFeature(df_crypto, os.path.join(FILE_PATH_CRYPTO_PROCESSED, "BTC-USD.parquet"))

    # Sanity check: reload processed files and inspect shape/columns
    df = pd.read_parquet(os.path.join(FILE_PATH_STOCKS_PROCESSED, "AAPL.parquet"))
    df_crypto = pd.read_parquet(os.path.join(FILE_PATH_CRYPTO_PROCESSED, "BTC-USD.parquet"))
    df_bonds = pd.read_parquet(os.path.join(FILE_PATH_BONDS_PROCESSED, "TLT.parquet"))
    df['types'] = 0
    df_crypto['types'] = 1
    df_bonds['types'] = 2

    print(f"Stocks; {df.size}",df.columns)
    print(f"Crypto; {df_crypto.size}",df_crypto.columns)
    print(f"Bonds; {df_bonds.size}",df_bonds.columns)

    print(f"Stocks len; {len(df)}", df.columns)
    print(f"Crypto; {len(df_crypto)}",df_crypto.columns)
    print(f"Bonds; {len(df_bonds)}",df_bonds.columns)
    
