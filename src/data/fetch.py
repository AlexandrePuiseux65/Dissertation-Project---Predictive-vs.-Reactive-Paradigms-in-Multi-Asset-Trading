'''
This file has for goal to fetch the data that the API from alpaca has to give. 
'''

# --- Lib --- #
import os
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pyarrow as pa
import pyarrow.parquet as pq
from alpaca.data import CryptoHistoricalDataClient, StockHistoricalDataClient, OptionHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.common.exceptions import APIError
from dotenv import load_dotenv
load_dotenv()


# --- Global Variable --- #
KEY = os.getenv("API_ALPACA_KEY")
SECRET = os.getenv("API_ALPACA_SECRET")

print(f"KEY: {KEY[:5] if KEY else 'None'}")
print(f"SECRET: {SECRET[:5] if SECRET else 'None'}")

START_DT = datetime.strptime("2015-01-01", "%Y-%m-%d")
END_DT = datetime.strptime("2025-12-31", "%Y-%m-%d")
TIMEFRAME = TimeFrame(3, TimeFrameUnit.Minute)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FILE_PATH_STOCKS = os.path.join(BASE_DIR, "data", "raw", "stocks")
FILE_PATH_CRYPTO = os.path.join(BASE_DIR, "data", "raw", "crypto")
FILE_PATH_BONDS = os.path.join(BASE_DIR, "data", "raw", "bonds")

# --- Class --- # 
class FetchData:
    def __init__(self, KEY, SECRET) -> None:
        self.KEY = KEY
        self.SECRET = SECRET
        self.client_stock = StockHistoricalDataClient(KEY, SECRET)
        self.client_option = OptionHistoricalDataClient(KEY, SECRET)
        self.client_crypto = CryptoHistoricalDataClient()
    
    def FetchStocksHistorical(self, ticker: str):
        """
            Function that fetch the stocks data, from 2015 to 2025. 
            Create a .parquet file, with the following columns : 
                - [timestamp, open, high, low, close, volume, trade_count, vwap], of size (180337, 9).
        """
        try:
            all_chunks = []
            current = START_DT

            while current < END_DT:
                chunk_end = min(current.replace(year=current.year +1), END_DT)

                request = StockBarsRequest(
                    symbol_or_symbols=ticker,
                    timeframe=TIMEFRAME,
                    start=current,
                    end=chunk_end,
                    adjustment=Adjustment.ALL,
                    feed=DataFeed.IEX
                    )

                bars=self.client_stock.get_stock_bars(request)
                df=bars.df.reset_index()

                if not df.empty:
                    all_chunks.append(df)
                    print(f"Success: {ticker} {current.strftime('%Y-%m')} => {len(df)} rows")

                current = chunk_end

            if all_chunks:
                final_df = pd.concat(all_chunks, ignore_index=True)
                os.makedirs(FILE_PATH_STOCKS, exist_ok=True)
                output_path=os.path.join(FILE_PATH_STOCKS, f"{ticker}.parquet")
                final_df.to_parquet(output_path, index=False)
                print(f"Success: {ticker} rows saved to file {output_path}.")

        except APIError as e:
            print(f"Error API; FetchStocksHistorical: {e}.")
        except Exception as e:
            print(f"Error Exception; FetchStocksHistorical: {e}.")

    def FetchOptionsHistorical(self, options: str):
        """
            Same but with a different file folder.
        """
        try:
            all_chunks = []
            current = START_DT

            while current < END_DT:
                chunk_end = min(current.replace(year=current.year +1), END_DT)

                request = StockBarsRequest(
                    symbol_or_symbols=options,
                    timeframe=TIMEFRAME,
                    start=current,
                    end=chunk_end,
                    adjustment=Adjustment.ALL,
                    feed=DataFeed.IEX
                    )

                bars=self.client_stock.get_stock_bars(request)
                df=bars.df.reset_index()

                if not df.empty:
                    all_chunks.append(df)
                    print(f"Success: {options} {current.strftime('%Y-%m')} => {len(df)} rows")

                current = chunk_end

            if all_chunks:
                final_df = pd.concat(all_chunks, ignore_index=True)
                os.makedirs(FILE_PATH_BONDS, exist_ok=True)
                output_path=os.path.join(FILE_PATH_BONDS, f"{options}.parquet")
                final_df.to_parquet(output_path, index=False)
                print(f"Success: {options} rows saved to file {output_path}.")

        except APIError as e:
            print(f"Error API; FetchStocksHistorical: {e}.")
        except Exception as e:
            print(f"Error Exception; FetchStocksHistorical: {e}.")

    def FetchCryptosHistorical(self, name_crypto: str):
        """
            Function that fetch the crypto data, from 2015 to 2025. 
            Create a .parquet file, with the following columns : 
                - [timestamp, open, high, low, close, volume, trade_count, vwap], of size (848726, 9).
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
    # test
    fetch=FetchData(KEY, SECRET)
    # df_stocks = fetch.FetchStocksHistorical("AAPL")
    #df_crypto = fetch.FetchCryptosHistorical("BTC/USD")
    # df_bonds = fetch.FetchOptionsHistorical("TLT")
    # verif
    df = pd.read_parquet(os.path.join(FILE_PATH_STOCKS, "AAPL.parquet"))
    df_crypto = pd.read_parquet(os.path.join(FILE_PATH_CRYPTO, "BTC-USD.parquet"))
    print(df)
    print(df_crypto)