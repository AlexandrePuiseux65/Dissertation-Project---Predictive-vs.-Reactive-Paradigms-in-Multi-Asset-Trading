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

# --- Global Variable --- #
KEY = os.getenv("API_ALPACA_KEY")
SECRET = os.getenv("API_ALPACA_SECRET")

if not KEY or not SECRET:
    raise EnvironmentError("API_ALPACA_KEY or API_ALPACA_SECRET not found. Check your .env file.")

START_DT = datetime.strptime("2015-01-01", "%Y-%m-%d")
END_DT = datetime.strptime("2025-12-31", "%Y-%m-%d")
TIMEFRAME = TimeFrame(3, TimeFrameUnit.Minute) 
FILE_PATH_STOCKS = "../data/raw/stocks/"
FILE_PATH_CRYPTO = "../data/raw/crypto/"
FILE_PATH_BONDS = "../data/raw/bonds/"

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
            request = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TIMEFRAME,
                start=START_DT,
                end=END_DT,
                adjustment=Adjustment.ALL,
                feed=DataFeed.SIP
            )

            bars=self.client_stock.get_stock_bars(request)
            df=bars.df.reset_index()

            if df.empty: return None

            os.makedirs(FILE_PATH_STOCKS, exist_ok=True)
            output_path=os.path.join(FILE_PATH_STOCKS, f"{ticker}.parquet")
            df.to_parquet(output_path, index=False)
            print(f"SUCCESS: {ticker} rows saved to file {output_path}.")
            return df
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