'''
    Code for the LSTM v1 model, also have the class fro the tranning of any LSTM model types.
'''

import pandas as pd
import torch
import torch.nn as nn
import os
import numpy as np
from sklearn.preprocessing import StandardScaler


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FILE_PATH_STOCKS_PROCESSED = os.path.join(BASE_DIR, "data", "processed", "stocks")
FILE_PATH_CRYPTO_PROCESSED = os.path.join(BASE_DIR, "data", "processed", "crypto")
FILE_PATH_BONDS_PROCESSED  = os.path.join(BASE_DIR, "data", "processed", "bonds")

FEATURE_COLS = ['open', 'high', 'low', 'close', 'volume', 'trade_count',
                'vwap', 'EMA', 'HMA', 'EVWMA', 'ROC', 'RSI', 'Williams_R']

'''
    Class that produce the model LSTM, depending on the input size, the hidden layer size
    and the output size.
'''
class MyLSTM_V1(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input):
        lstm_out, _ = self.lstm(input)
        predictions = self.linear(lstm_out[-1])
        return predictions

'''
    Loads and prepares a processed parquet file for LSTM training.
    Computes the log-return target, splits chronologically (75/15/10),
    normalizes features using StandardScaler fitted on train only,
    and returns tensors for features and targets.
'''
def PreparationData(link, file_name):
    df = pd.read_parquet(os.path.join(link, file_name))

    df['target'] = np.log(df['close'].shift(-1) / df['close'])
    df = df.dropna(subset=['target'])

    n = len(df)
    train_end = int(n * 0.75)
    val_end   = int(n * 0.90)

    train_df = df.iloc[:train_end]
    val_df   = df.iloc[train_end:val_end]
    test_df  = df.iloc[val_end:]

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df[FEATURE_COLS])
    val_scaled   = scaler.transform(val_df[FEATURE_COLS])
    test_scaled  = scaler.transform(test_df[FEATURE_COLS])

    train_data_normalized = torch.FloatTensor(train_scaled)
    val_data_normalized = torch.FloatTensor(val_scaled)
    test_data_normalized = torch.FloatTensor(test_scaled)

    train_target = torch.FloatTensor(train_df['target'].values)
    val_target   = torch.FloatTensor(val_df['target'].values)
    test_target  = torch.FloatTensor(test_df['target'].values)

    return train_data_normalized, val_data_normalized, test_data_normalized, train_target, val_target, test_target


if __name__ == "__main__":
    train_stocks, verif_stocks, test_stocks = PreparationData(FILE_PATH_STOCKS_PROCESSED, file_name="AAPL.parquet")
    train_donds, verif_donds, test_donds = PreparationData(FILE_PATH_BONDS_PROCESSED, file_name="TLT.parquet")
    train_crypto, verif_crypto, test_crypto = PreparationData(FILE_PATH_CRYPTO_PROCESSED, file_name="BTC-USD.parquet")

    # Train the model.
    model_v1 = MyLSTM_V1()
    loss_function = nn.MSELoss() # A changer MSE/MAE pour pertes dirrectionnelles ou voir comment on mets
    optimiser = torch.optim.Adam(model_v1.parameters, lr=0.001)
    epochs = 50


    try:
        for epoch in range(epochs):
            pass
        # model = MonLSTM().to(device)
        # X_train = X_train.to(device)
    except Exception as e:
        print(f"Error GPU detection: {e}")