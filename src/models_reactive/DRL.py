'''
    Code for the RL v1 model, also have the class fro the tranning of any LSTM model types.
'''
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

ALL_COLS = FEATURE_COLS + ['asset_type']

class TradingEnv(gymnasium.Env):
    '''
        Trading environment for Deep Reinforcement Learning.
        Implements the Gymnasium interface for PPO training.
        The agent observes a 24h window of market features and outputs
        a continuous position size in [-1, 1] (short to long).
    '''
    def __init__(self, data):
        super().__init__()
        self.close_prices = data['close'].values.astype(np.float32)
        self.features = data[ALL_COLS].values.astype(np.float32)
        self.n_steps  = len(self.features)
        self.current_step = 24
        self.returns_history = []
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(24 * len(ALL_COLS),),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        '''
            Resets the environment to a random starting point in the first half
            of the data. Called at the beginning of each training episode.
            Returns the initial observation and an empty info dict.
        '''
        super().reset(seed=seed)
        self.current_step = np.random.randint(24, self.n_steps // 2)
        self.returns_history = []
        return self._get_observation(), {}

    def _get_observation(self):
        '''
            Returns the current observation: a window of the last 24 timesteps
            of market features, shaped (24, n_features).
        '''
        return self.features[self.current_step - 24:self.current_step].flatten()

    def step(self, action):
        '''
            Executes one step in the environment.
            - Computes the log-return between current and next price
            - Scales it by the agent's position (action)
            - Returns (observation, reward, done, truncated, info)
        '''
        current_price = self.close_prices[self.current_step]
        next_price    = self.close_prices[self.current_step + 1]

        log_return       = np.log(next_price / current_price)
        reward           = action[0] * log_return

        self.current_step += 1
        done = self.current_step >= self.n_steps - 1
        obs  = self._get_observation()

        return obs, reward, done, False, {}

def PreparationData(link, file_name, asset_type):
    '''
        Loads and prepares a processed parquet file for RL training.
    '''
    df = pd.read_parquet(os.path.join(link, file_name))
    df = df.dropna()

    n = len(df)
    train_end = int(n * 0.75)
    val_end   = int(n * 0.90)

    train_df = df.iloc[:train_end].copy()
    val_df   = df.iloc[train_end:val_end].copy()
    test_df  = df.iloc[val_end:].copy()

    scaler = StandardScaler()
    train_df[FEATURE_COLS] = scaler.fit_transform(train_df[FEATURE_COLS])
    val_df[FEATURE_COLS]   = scaler.transform(val_df[FEATURE_COLS])
    test_df[FEATURE_COLS]  = scaler.transform(test_df[FEATURE_COLS])
    train_df['asset_type'] = asset_type
    val_df['asset_type']   = asset_type
    test_df['asset_type']  = asset_type

    return train_df, val_df, test_df

# --- Main --- # 
if __name__ == "__main__":

    train_stocks, val_stocks, test_stocks = PreparationData(FILE_PATH_STOCKS_PROCESSED, "AAPL.parquet", asset_type=0)
    train_bonds,  val_bonds,  test_bonds  = PreparationData(FILE_PATH_BONDS_PROCESSED,  "TLT.parquet",  asset_type=1)
    train_crypto, val_crypto, test_crypto = PreparationData(FILE_PATH_CRYPTO_PROCESSED, "BTC-USD.parquet", asset_type=2)

    train_all = pd.concat([train_stocks, train_bonds, train_crypto]).reset_index(drop=True)
    env = TradingEnv(train_all)

    val_all = pd.concat([val_stocks, val_bonds, val_crypto]).reset_index(drop=True)
    val_env = TradingEnv(val_all)

    obs, info = env.reset()
    print("obs shape:", obs.shape)
    print("obs NaN:", np.isnan(obs).sum())
    action = env.action_space.sample()
    obs, reward, done, trunc, info = env.step(action)
    print("reward:", reward, "done:", done)

    checkpoint_cb = CheckpointCallback(
        save_freq=10000,
        save_path=FILE_SAVE_MODEL,
        name_prefix="drl_v1"
    )

    model = PPO("MlpPolicy", env, verbose=1)
    eval_cb = EvalCallback(val_env, 
                       best_model_save_path=FILE_SAVE_MODEL,
                       log_path=FILE_SAVE_MODEL,
                       eval_freq=10000, verbose=1)

    model.learn(total_timesteps=1_000_000, reset_num_timesteps=True, callback=[checkpoint_cb, eval_cb])
    model.save(os.path.join(FILE_SAVE_MODEL, "drl_v1"))
    print("Model saved.")