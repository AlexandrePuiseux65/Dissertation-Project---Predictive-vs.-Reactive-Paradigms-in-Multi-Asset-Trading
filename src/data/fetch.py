'''
    This script fetches historical market data (stocks, bonds, crypto)
    from the Alpaca API and saves it as raw parquet files, to be later
    processed by features.py.
'''

# --- Lib --- #
import os
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pyarrow as pa
import pyarrow.parquet as pq
from alpaca.data import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.common.exceptions import APIError
from dotenv import load_dotenv
load_dotenv()

# --- Global Variable --- #
KEY = os.getenv("API_ALPACA_KEY")
SECRET = os.getenv("API_ALPACA_SECRET")

print("Alpaca credentials loaded." if KEY and SECRET else "Warning: missing Alpaca credentials.")

START_DT = datetime.strptime("2015-01-01", "%Y-%m-%d")
END_DT = datetime.strptime("2025-12-31", "%Y-%m-%d")

TIMEFRAME = TimeFrame(3, TimeFrameUnit.Minute)  # Raw fetch granularity; resampled to 1h in features.py

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FILE_PATH_STOCKS = os.path.join(BASE_DIR, "data", "raw", "stocks")
FILE_PATH_CRYPTO = os.path.join(BASE_DIR, "data", "raw", "crypto")
FILE_PATH_BONDS = os.path.join(BASE_DIR, "data", "raw", "bonds")

# --- Class --- #
class FetchData:
    def __init__(self, key, secret) -> None:
        self.KEY = key
        self.SECRET = secret
        self.client_stock = StockHistoricalDataClient(key, secret)
        self.client_crypto = CryptoHistoricalDataClient()

    def FetchStocksHistorical(self, ticker: str):
        """
            Fetch historical stock bars for the given ticker between
            START_DT and END_DT, chunked year by year (Alpaca API limits),
            and save the result as a parquet file under FILE_PATH_STOCKS.
            Columns: [timestamp, open, high, low, close, volume, trade_count, vwap].
        """
        try:
            all_chunks = []
            current = START_DT

            while current < END_DT:
                chunk_end = min(current.replace(year=current.year + 1), END_DT)

                request = StockBarsRequest(
                    symbol_or_symbols=ticker,
                    timeframe=TIMEFRAME,
                    start=current,
                    end=chunk_end,
                    adjustment=Adjustment.ALL,
                    feed=DataFeed.IEX
                )

                bars = self.client_stock.get_stock_bars(request)
                df = bars.df.reset_index()

                if not df.empty:
                    all_chunks.append(df)
                    print(f"Success: {ticker} {current.strftime('%Y-%m')} => {len(df)} rows")

                current = chunk_end

            if all_chunks:
                final_df = pd.concat(all_chunks, ignore_index=True)
                os.makedirs(FILE_PATH_STOCKS, exist_ok=True)
                output_path = os.path.join(FILE_PATH_STOCKS, f"{ticker}.parquet")
                final_df.to_parquet(output_path, index=False)
                print(f"Success: {ticker} rows saved to file {output_path}.")

        except APIError as e:
            print(f"Error API; FetchStocksHistorical: {e}.")
        except Exception as e:
            print(f"Error Exception; FetchStocksHistorical: {e}.")

    def FetchBondsHistorical(self, ticker: str):
        """
            Fetch historical bars for a bond ETF (e.g. TLT) using the
            stock bars endpoint, chunked year by year, and save the
            result as a parquet file under FILE_PATH_BONDS.
            Columns: [timestamp, open, high, low, close, volume, trade_count, vwap].
        """
        try:
            all_chunks = []
            current = START_DT

            while current < END_DT:
                chunk_end = min(current.replace(year=current.year + 1), END_DT)

                request = StockBarsRequest(
                    symbol_or_symbols=ticker,
                    timeframe=TIMEFRAME,
                    start=current,
                    end=chunk_end,
                    adjustment=Adjustment.ALL,
                    feed=DataFeed.IEX
                )

                bars = self.client_stock.get_stock_bars(request)
                df = bars.df.reset_index()

                if not df.empty:
                    all_chunks.append(df)
                    print(f"Success: {ticker} {current.strftime('%Y-%m')} => {len(df)} rows")

                current = chunk_end

            if all_chunks:
                final_df = pd.concat(all_chunks, ignore_index=True)
                os.makedirs(FILE_PATH_BONDS, exist_ok=True)
                output_path = os.path.join(FILE_PATH_BONDS, f"{ticker}.parquet")
                final_df.to_parquet(output_path, index=False)
                print(f"Success: {ticker} rows saved to file {output_path}.")

        except APIError as e:
            print(f"Error API; FetchBondsHistorical: {e}.")
        except Exception as e:
            print(f"Error Exception; FetchBondsHistorical: {e}.")

    def FetchCryptosHistorical(self, name_crypto: str):
        """
            Fetch historical crypto bars for the given pair between
            START_DT and END_DT, chunked month by month, writing
            incrementally to a parquet file under FILE_PATH_CRYPTO
            (avoids holding the entire history in memory at once).
            Columns: [timestamp, open, high, low, close, volume, trade_count, vwap].
        """
        try:
            convert_name_crypto = name_crypto.replace("/", "-")
            os.makedirs(FILE_PATH_CRYPTO, exist_ok=True)
            output_path = os.path.join(FILE_PATH_CRYPTO, f"{convert_name_crypto}.parquet")
            current = START_DT
            writer = None

            while current < END_DT:
                chunk_end = min(current + relativedelta(months=1), END_DT)

                request = CryptoBarsRequest(
                    symbol_or_symbols=name_crypto,
                    timeframe=TIMEFRAME,
                    start=current,
                    end=chunk_end,
                )

                bars = self.client_crypto.get_crypto_bars(request)
                df = bars.df.reset_index()

                if not df.empty:
                    table = pa.Table.from_pandas(df)
                    if writer is None:
                        writer = pq.ParquetWriter(output_path, table.schema)
                    writer.write_table(table)
                    print(f"Success: {name_crypto} {current.strftime('%Y-%m')} => {len(df)} rows")

                current = chunk_end

            if writer:
                writer.close()
                print(f"Success: {name_crypto} - done")

        except APIError as e:
            print(f"Error API; FetchCryptosHistorical: {e}.")
        except Exception as e:
            print(f"Error Exception; FetchCryptosHistorical: {e}.")

if __name__ == "__main__":
    # Fetch the data.
    fetch = FetchData(KEY, SECRET)
    fetch.FetchStocksHistorical("AAPL")
    fetch.FetchCryptosHistorical("BTC/USD")
    fetch.FetchBondsHistorical("TLT")

    # Verification of the data
    df = pd.read_parquet(os.path.join(FILE_PATH_STOCKS, "AAPL.parquet"))
    df_crypto = pd.read_parquet(os.path.join(FILE_PATH_CRYPTO, "BTC-USD.parquet"))
    df_bonds = pd.read_parquet(os.path.join(FILE_PATH_BONDS, "TLT.parquet"))
    print(f"Stocks; {df.size}", df.columns)
    print(f"Crypto; {df_crypto.size}", df_crypto.columns)
    print(f"Bonds; {df_bonds.size}", df_bonds.columns)

    print(f"Stocks len; {len(df)}", df.columns)
    print(f"Crypto; {len(df_crypto)}",df_crypto.columns)
    print(f"Bonds; {len(df_bonds)}",df_bonds.columns)