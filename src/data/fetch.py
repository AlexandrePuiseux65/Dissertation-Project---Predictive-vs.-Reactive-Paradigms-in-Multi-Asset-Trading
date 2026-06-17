'''
This file has for goal to fetch the data that the API from alpaca has to give. 
'''

# --- Lib --- #
import os
from dotenv import load_dotenv

load_dotenv()

import pandas as pd
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import Adjustment, DataFeed

# --- Global Variable --- #
KEY = os.getenv("API_ALPACA_KEY")
SECRET = os.getenv("API_ALPACA_SECRET")

# --- Class --- # 
class FetchData:
    def __init__(self, KEY, SECRET) -> None:
        pass