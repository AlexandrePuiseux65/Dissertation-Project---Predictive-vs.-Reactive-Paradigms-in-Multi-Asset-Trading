'''
    Code for the LSTM v1 model, also have the class fro the tranning of any LSTM model types.
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

FEATURE_COLS =  ['open', 'high', 'low', 'close', 'volume', 'trade_count',
       'vwap', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
       'BB_middle', 'BB_upper', 'BB_lower']

# --- Class --- #
'''
    Class that produce the model LSTM, depending on the input size, the hidden layer size
    and the output size.
'''
class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input):
        lstm_out, _ = self.lstm(input)
        lstm_out = self.dropout(lstm_out)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

# --- Functions --- # 
'''
    Train the LSTM model, with early stopping, in order to not overfeed the model.
    Base and modify on ths code "https://codesignal.com/learn/courses/time-series-forecasting-with-lstms-2/lessons/optimizing-lstm-models-for-time-series-forecasting-with-pytorch"
'''
def TrainModel(model, model_name ,train_loader, val_loader, loss_function, optimiser, device, epochs, patience):
    best_loss = np.inf
    patience_c = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    train_losses = []
    val_losses   = []

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for seq, label in train_loader:
            optimiser.zero_grad()
            y_pred = model(seq.to(device))
            loss = loss_function(y_pred.squeeze(), label.to(device))
            loss.backward()
            optimiser.step()
            train_loss += loss.item()
        train_losses.append(train_loss / len(train_loader))

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for seq, label in val_loader:
                y_pred = model(seq.to(device))
                loss = loss_function(y_pred.squeeze(), label.to(device))
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f'Epoch {epoch+1} -> T-Loss: {train_losses[-1]:.6f} | V-Loss: {val_loss:.6f}')

        if val_loss < best_loss:
            best_loss = val_loss
            patience_c = 0
            best_model_wts = copy.deepcopy(model.state_dict())
        else:
            patience_c += 1

        if patience_c >= patience:
            print(f"Early Stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), os.path.join(FILE_SAVE_MODEL, f"{model_name}.pth"))
    print(f"Model saved -> best val_loss: {best_loss:.6f}")
    return train_losses, val_losses

'''
    Loads and prepares a processed parquet file for LSTM training.
    Computes the log-return target, splits chronologically (75/15/10),
    normalizes features using StandardScaler fitted on train only,
    and returns tensors for features and targets.
'''
def PreparationData(link, file_name):
    df = pd.read_parquet(os.path.join(link, file_name))

    df['target'] = np.log(df['close'].shift(-1) / df['close'])
    df = df.dropna()

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

    train_target = torch.FloatTensor(train_df['target'].values.copy())
    val_target   = torch.FloatTensor(val_df['target'].values.copy())
    test_target  = torch.FloatTensor(test_df['target'].values.copy())

    return train_data_normalized, val_data_normalized, test_data_normalized, train_target, val_target, test_target
    
'''
    Builds sliding window sequences from features and targets.
    Each sequence is a window of seq_len timesteps (features) paired
    with the log-return at the next timestep (label).
'''
def CreateInOutSequence(features, targets, seq_len):
    sequences = []
    for i in range(len(features) - seq_len):
        seq   = features[i:i + seq_len]# (seq_len, 13) — features
        label = targets[i + seq_len]# scalaire — log-return suivant
        sequences.append((seq, label))
    return sequences

# --- Main --- # 
if __name__ == "__main__":
    # Prepare the data before trainning
    train_stocks, val_stocks, test_stocks, train_y_stocks, val_y_stocks, test_y_stocks= PreparationData(FILE_PATH_STOCKS_PROCESSED, file_name="AAPL.parquet")
    train_bonds, val_bonds, test_bonds, train_y_bonds, val_y_bonds, test_y_bonds = PreparationData(FILE_PATH_BONDS_PROCESSED, file_name="TLT.parquet")
    train_crypto, val_crypto, test_crypto, train_y_crypto, val_y_crypto, test_y_crypto = PreparationData(FILE_PATH_CRYPTO_PROCESSED, file_name="BTC-USD.parquet")

    # Add asset type as categorical feature (0=stocks, 1=bonds, 2=crypto)
    train_stocks = torch.cat([train_stocks, torch.full((len(train_stocks), 1), 0.0)], dim=1)
    train_bonds  = torch.cat([train_bonds,  torch.full((len(train_bonds),  1), 1.0)], dim=1)
    train_crypto = torch.cat([train_crypto, torch.full((len(train_crypto), 1), 2.0)], dim=1)
    val_stocks = torch.cat([val_stocks, torch.full((len(val_stocks), 1), 0.0)], dim=1)
    val_bonds  = torch.cat([val_bonds,  torch.full((len(val_bonds),  1), 1.0)], dim=1)
    val_crypto = torch.cat([val_crypto, torch.full((len(val_crypto), 1), 2.0)], dim=1)

    # Build sliding window sequences of length 24h for each asset (same for validation set)
    all_train = (
        CreateInOutSequence(train_stocks, train_y_stocks, 24) +
        CreateInOutSequence(train_bonds,  train_y_bonds,  24) +
        CreateInOutSequence(train_crypto, train_y_crypto, 24)
    )

    all_val = (
        CreateInOutSequence(val_stocks, val_y_stocks, 24) +
        CreateInOutSequence(val_bonds,  val_y_bonds,  24) +
        CreateInOutSequence(val_crypto, val_y_crypto, 24)
    )

    # Stack sequences into tensors for DataLoader
    train_seqs   = torch.stack([s for s, _ in all_train])
    train_labels = torch.stack([l for _, l in all_train])
    val_seqs     = torch.stack([s for s, _ in all_val])
    val_labels   = torch.stack([l for _, l in all_val])

    # Wrap in DataLoader for mini-batch training (batch_size=32)
    train_loader = DataLoader(TensorDataset(train_seqs, train_labels), batch_size=32, shuffle=False)
    val_loader   = DataLoader(TensorDataset(val_seqs,   val_labels),   batch_size=32, shuffle=False)  

    # Trainning information
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_v1 = MyLSTM(input_size=17, hidden_layer_size=32, output_size=1).to(device)
    loss_function = nn.MSELoss() # A changer MSE/MAE pour pertes dirrectionnelles ou voir comment on mets
    optimiser = torch.optim.Adam(model_v1.parameters(), lr=0.0005)
    model_name='lstm_v1'
    
    # Trainning of the model
    train_loss, val_loss = TrainModel(model_v1, model_name, train_loader, val_loader, loss_function, optimiser, device, epochs=100, patience=10)

    # Visualisation of the progress :
    plt.figure(figsize=(12, 6))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss,   label='Val Loss')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(FILE_SAVE_MODEL, "loss_curve.png"))