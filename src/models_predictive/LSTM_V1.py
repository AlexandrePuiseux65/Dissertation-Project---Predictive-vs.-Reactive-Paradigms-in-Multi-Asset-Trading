'''
    This is the test model 
'''

import pandas as pd
import torch
import torch.nn as nn
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FILE_PATH_STOCKS = os.path.join(BASE_DIR, "data", "raw", "stocks")
FILE_PATH_CRYPTO = os.path.join(BASE_DIR, "data", "raw", "crypto")
FILE_PATH_BONDS = os.path.join(BASE_DIR, "data", "raw", "bonds")

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

def FindTargetLSTM(df):
    df['target']= np.log(df['close'].shift(-1)/df['close'])
    return df

if __name__ == "__main__":
    # df = 

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