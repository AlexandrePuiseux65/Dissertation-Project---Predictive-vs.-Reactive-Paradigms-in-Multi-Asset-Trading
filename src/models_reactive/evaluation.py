# --- Lib --- #
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import gymnasium
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

# --- Global Variable --- #
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FILE_PATH_STOCKS_PROCESSED = os.path.join(BASE_DIR, "data", "processed", "stocks")
FILE_PATH_CRYPTO_PROCESSED = os.path.join(BASE_DIR, "data", "processed", "crypto")
FILE_PATH_BONDS_PROCESSED  = os.path.join(BASE_DIR, "data", "processed", "bonds")

FILE_SAVE_MODEL = os.path.join(BASE_DIR, "model")

FEATURE_COLS = ['open', 'high', 'low', 'volume', 'trade_count', 'vwap',
                'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
                'BB_middle', 'BB_upper', 'BB_lower']

