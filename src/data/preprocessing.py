'''
This code have for goal to preprocessing the data.
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