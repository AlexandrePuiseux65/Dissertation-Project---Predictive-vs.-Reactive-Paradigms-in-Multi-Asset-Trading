'''
    Code for the RL v1 model, also have the class fro the tranning of any LSTM model types.
'''
# --- Lib --- #
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy as np
import copy
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --- Global Variable --- #
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FILE_PATH_STOCKS_PROCESSED = os.path.join(BASE_DIR, "data", "processed", "stocks")
FILE_PATH_CRYPTO_PROCESSED = os.path.join(BASE_DIR, "data", "processed", "crypto")
FILE_PATH_BONDS_PROCESSED  = os.path.join(BASE_DIR, "data", "processed", "bonds")

FILE_SAVE_MODEL = os.path.join(BASE_DIR, "model")

FEATURE_COLS = ['open', 'high', 'low', 'close', 'volume', 'trade_count',
                'vwap', 'EMA', 'HMA', 'EVWMA', 'ROC', 'RSI', 'Williams_R']