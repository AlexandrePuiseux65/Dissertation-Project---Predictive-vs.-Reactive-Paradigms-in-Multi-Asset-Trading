# --- Lib --- #
import torch
import torch.nn as nn
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from LSTM_V1 import MyLSTM, PreparationData, CreateInOutSequence
from LSTM_V1 import FILE_PATH_STOCKS_PROCESSED, FILE_PATH_BONDS_PROCESSED
from LSTM_V1 import FILE_PATH_CRYPTO_PROCESSED, FILE_SAVE_MODEL

# --- General Variable --#
HIDDEN_LAYER_SIZE = 32
SEQUENCE = 48

# --- Load model --- #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MyLSTM(input_size=17, hidden_layer_size=HIDDEN_LAYER_SIZE, output_size=1)
model.load_state_dict(torch.load(os.path.join(FILE_SAVE_MODEL, "lstm_v1.pth"), weights_only=True))
model.to(device)
model.eval()

# --- Prepare test data --- #
_, _, test_stocks, _, _, test_y_stocks = PreparationData(FILE_PATH_STOCKS_PROCESSED, "AAPL.parquet")
_, _, test_bonds,  _, _, test_y_bonds  = PreparationData(FILE_PATH_BONDS_PROCESSED,  "TLT.parquet")
_, _, test_crypto, _, _, test_y_crypto = PreparationData(FILE_PATH_CRYPTO_PROCESSED, "BTC-USD.parquet")

test_stocks = torch.cat([test_stocks, torch.full((len(test_stocks), 1), 0.0)], dim=1)
test_bonds  = torch.cat([test_bonds,  torch.full((len(test_bonds),  1), 1.0)], dim=1)
test_crypto = torch.cat([test_crypto, torch.full((len(test_crypto), 1), 2.0)], dim=1)

all_test = (
    CreateInOutSequence(test_stocks, test_y_stocks, SEQUENCE) +
    CreateInOutSequence(test_bonds,  test_y_bonds,  SEQUENCE) +
    CreateInOutSequence(test_crypto, test_y_crypto,SEQUENCE )
)

test_seqs   = torch.stack([s for s, _ in all_test])
test_labels = torch.stack([l for _, l in all_test])

# --- Evaluate --- #
with torch.no_grad():
    y_pred = model(test_seqs.to(device)).squeeze().cpu()

mse = nn.MSELoss()(y_pred, test_labels).item()
print(f"Test MSE:             {mse:.6f}")
print(f"Predicted mean: {y_pred.mean():.6f}, std: {y_pred.std():.6f}")
print(f"Actual mean:    {test_labels.mean():.6f}, std: {test_labels.std():.6f}")

correct  = (y_pred > 0) == (test_labels > 0)
accuracy = correct.float().mean().item()
print(f"Directional Accuracy: {accuracy:.2%}")

plt.figure(figsize=(12, 4))
plt.plot(test_labels[:200].numpy(), label='Actual', alpha=0.7)
plt.plot(y_pred[:200].numpy(),      label='Predicted', alpha=0.7)
plt.axhline(0, color='black', linewidth=0.5)
plt.legend()
plt.title('Predicted vs Actual log-returns')
plt.savefig(os.path.join(FILE_SAVE_MODEL, "predictions.png"))