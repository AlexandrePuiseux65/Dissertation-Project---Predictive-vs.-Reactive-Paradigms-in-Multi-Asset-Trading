'''
    This is the test model 
'''

import pandas as pd
import torch
import torch.nn as nn
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FILE_PATH_STOCKS_PROCESSED = os.path.join(BASE_DIR, "data", "processed", "stocks")
FILE_PATH_CRYPTO_PROCESSED = os.path.join(BASE_DIR, "data", "processed", "crypto")
FILE_PATH_BONDS_PROCESSED  = os.path.join(BASE_DIR, "data", "processed", "bonds")

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

if __name__ == "__main__":
    # get data.
    df = pd.read_parquet(os.path.join(FILE_PATH_STOCKS_PROCESSED, "AAPL.parquet"))
    df_crypto = pd.read_parquet(os.path.join(FILE_PATH_CRYPTO_PROCESSED, "BTC-USD.parquet"))
    df_bonds = pd.read_parquet(os.path.join(FILE_PATH_BONDS_PROCESSED, "TLT.parquet"))

    # Find the log return.
    df['target']= np.log(df['close'].shift(-1)/df['close'])
    df_bonds['target']= np.log(df_bonds['close'].shift(-1)/df_bonds['close'])
    df_crypto['target']= np.log(df_crypto['close'].shift(-1)/df_crypto['close'])

    df['types'] = 0
    df_bonds['types'] = 1
    df_crypto['types'] = 2

    # Separating the data
    df = df.sort_values('timestamp').reset_index(drop=True)
    n = len(df)
    train_end = int(n * 0.75)
    val_end   = int(n * 0.90)

    train_data = df.iloc[:train_end]
    verif_data = df.iloc[train_end:val_end]
    test_data  = df.iloc[val_end:]

    # Normalisation of the feature.


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