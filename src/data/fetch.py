'''
This file has for goal to fetch the data that the API from alpaca has to give. 
'''

# --- Lib --- #
import os
import pandas as pd
from datetime import datetime
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

START_DT = datetime.strptime("2025-01-01", "%Y-%m-%d")
END_DT = datetime.strptime("2025-01-31", "%Y-%m-%d")
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
        try:
            all_chunks = []
            current = START_DT

            while current < END_DT:
                chunk_end = min(current.replace(year=current.year +1), END_DT)

                request = StockBarsRequest(
                    symbol_or_symbols=ticker,
                    timeframe=TIMEFRAME,
                    start=START_DT,
                    end=chunk_end,
                    adjustment=Adjustment.ALL,
                    feed=DataFeed.IEX
                    )
                print(request)

                bars=self.client_stock.get_stock_bars(request)
                df=bars.df.reset_index()

                if not df.empty:
                    all_chunks.append(df)
                    print(f"Added {len(df)} from {current.year} to dataframe.")

                current = chunk_end

            if all_chunks:
                final_df = pd.concat(all_chunks, ignore_index=True)
                os.makedirs(FILE_PATH_STOCKS, exist_ok=True)
                output_path=os.path.join(FILE_PATH_STOCKS, f"{ticker}.parquet")
                final_df.to_parquet(output_path, index=False)
                print(f"SUCCESS: {ticker} rows saved to file {output_path}.")

        except APIError as e:
            print(f"ERROR API; 'FetchStocksHistorical': {e}.")
        except Exception as e:
            print(f"ERROR; 'FetchStocksHistorical': {e}.")

    def FetchOptionsHistorical(self, options: str):
        # Process

        return

    def FetchCryptosHistorical(self, name_crypto):
        # Process

        return

if __name__ == "__main__":
    # test
    fetch=FetchData(KEY, SECRET)
    df_stocks = fetch.FetchStocksHistorical("AAPL")