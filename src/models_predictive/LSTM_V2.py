'''
    Code for the LSTM v2 model, also have the class for the training of any LSTM model types.

    # Experimental / not used in the current dissertation results
'''
# --- Lib --- #
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
from LSTM_V1 import MyLSTM, PreparationData, CreateInOutSequence, TrainModel, ProgressVisualisations
from LSTM_V1 import FILE_PATH_STOCKS_PROCESSED, FILE_PATH_BONDS_PROCESSED, FILE_PATH_CRYPTO_PROCESSED

# --- Global Variable --- #
SEQ_LEN = 48

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

    # Build sliding window sequences of length 48h for each asset (same for validation set)
    all_train = (
        CreateInOutSequence(train_stocks, train_y_stocks, SEQ_LEN) +
        CreateInOutSequence(train_bonds,  train_y_bonds, SEQ_LEN) +
        CreateInOutSequence(train_crypto, train_y_crypto, SEQ_LEN)
    )

    all_val = (
        CreateInOutSequence(val_stocks, val_y_stocks, SEQ_LEN) +
        CreateInOutSequence(val_bonds,  val_y_bonds, SEQ_LEN) +
        CreateInOutSequence(val_crypto, val_y_crypto, SEQ_LEN)
    )

    # Stack sequences into tensors for DataLoader
    train_seqs   = torch.stack([s for s, _ in all_train])
    train_labels = torch.stack([l for _, l in all_train])
    val_seqs     = torch.stack([s for s, _ in all_val])
    val_labels   = torch.stack([l for _, l in all_val])

    # Wrap in DataLoader for mini-batch training (batch_size=32)
    train_loader = DataLoader(TensorDataset(train_seqs, train_labels), batch_size=32, shuffle=False)
    val_loader   = DataLoader(TensorDataset(val_seqs,   val_labels),   batch_size=32, shuffle=False)  

    # Training information
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_v2 = MyLSTM(input_size=17, hidden_layer_size=64, output_size=1).to(device)
    loss_function = nn.MSELoss()
    optimiser = torch.optim.Adam(model_v2.parameters(), lr=0.0005)  
    model_name = "lstm_v2"
    
    # Training of the model
    train_loss, val_loss = TrainModel(model_v2, model_name, train_loader, val_loader, loss_function, optimiser, device, epochs=100, patience=10)

    # Visualisation of the progress :
    ProgressVisualisations(train_loss, val_loss)
    