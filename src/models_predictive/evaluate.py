# --- Lib --- #
import torch
import torch.nn as nn
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from LSTM_V1 import MyLSTM, PreparationData, CreateInOutSequence
from LSTM_V1 import FILE_PATH_STOCKS_PROCESSED, FILE_PATH_BONDS_PROCESSED, BASE_DIR
from LSTM_V1 import FILE_PATH_CRYPTO_PROCESSED, FILE_SAVE_MODEL

# --- General Variable --#
HIDDEN_LAYER_SIZE = 32
SEQUENCE = 24

FILE_SAVE_IMG = os.path.join(BASE_DIR, "src", "benchmark", "img", "LSTM")

# --- Load model --- #
def LoadModel():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MyLSTM(input_size=17, hidden_layer_size=HIDDEN_LAYER_SIZE, output_size=1)
    model.load_state_dict(torch.load(os.path.join(FILE_SAVE_MODEL, "lstm_v1.pth"), weights_only=True))
    model.to(device)
    model.eval()
    return model, device 

# --- Prepare test data --- #
def PrepareTestData(FILE_STOCKS, FILE_BONDS, FILE_CRYPTO):
    '''
        FIX: returns a list of (asset_name, test_seqs, test_labels) instead
        of one concatenated tensor pair, so that evaluation and net-return
        computation can be done per asset (position resets between assets).
    '''
    _, _, test_stocks, _, _, test_y_stocks = PreparationData(FILE_PATH_STOCKS_PROCESSED, FILE_STOCKS)
    _, _, test_bonds,  _, _, test_y_bonds  = PreparationData(FILE_PATH_BONDS_PROCESSED, FILE_BONDS)
    _, _, test_crypto, _, _, test_y_crypto = PreparationData(FILE_PATH_CRYPTO_PROCESSED, FILE_CRYPTO)

    test_stocks = torch.cat([test_stocks, torch.full((len(test_stocks), 1), 0.0)], dim=1)
    test_bonds  = torch.cat([test_bonds,  torch.full((len(test_bonds),  1), 1.0)], dim=1)
    test_crypto = torch.cat([test_crypto, torch.full((len(test_crypto), 1), 2.0)], dim=1)

    per_asset_raw = [
        ("AAPL", test_stocks, test_y_stocks),
        ("TLT",  test_bonds,  test_y_bonds),
        ("BTC",  test_crypto, test_y_crypto),
    ]

    result = []
    for name, feats, targets in per_asset_raw:
        seqs = CreateInOutSequence(feats, targets, SEQUENCE)
        test_seqs   = torch.stack([s for s, _ in seqs])
        test_labels = torch.stack([l for _, l in seqs])
        result.append((name, test_seqs, test_labels))

    return result

# --- Financial metrics --- #
def max_drawdown(returns):
    cumulative  = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    return (cumulative - running_max).min()

def sortino_ratio(returns, target=0.0):
    excess       = returns - target
    downside     = np.where(excess < 0, excess, 0)
    downside_std = np.sqrt(np.mean(downside**2)) + 1e-8
    return excess.mean() / downside_std

# --- LSTM testing (not in the benchmark) --- #
def TestingLSTM(y_pred, test_labels):
    '''
        Note: plots the first 200 points, which after the per-asset fix
        corresponds to the start of the AAPL test set (first asset
        evaluated), not a mix of all three assets.
    '''
    plt.figure(figsize=(12, 4))
    plt.plot(test_labels[:200].numpy(), label='Actual', alpha=0.7)
    plt.plot(y_pred[:200].numpy(), label='Predicted', alpha=0.7)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.legend()
    plt.title('Predicted vs Actual log-returns (AAPL test set)')
    plt.savefig(os.path.join(FILE_SAVE_IMG, "predictions.png"))

# --- Evaluate --- #
if __name__ == "__main__":
    model, device = LoadModel()
    per_asset_test = PrepareTestData(FILE_STOCKS="AAPL.parquet", FILE_BONDS="TLT.parquet", FILE_CRYPTO="BTC-USD.parquet")

    all_actions, all_portfolio, all_market, all_net = [], [], [], []
    y_pred_list, label_list = [], []

    for name, test_seqs, test_labels in per_asset_test:
        with torch.no_grad():
            y_pred = model(test_seqs.to(device)).squeeze().cpu()

        actions           = torch.sign(y_pred).numpy()
        portfolio_returns = actions * test_labels.numpy()
        market_returns    = test_labels.numpy()

        # FIX: reset position at the start of each asset instead of
        # carrying prev_action over from the previous asset's last step.
        net_returns = []
        prev_action = 0.0
        for i, act in enumerate(actions):
            cost = abs(act - prev_action) * 0.001
            net_returns.append(portfolio_returns[i] - cost)
            prev_action = act

        print(f"{name}: evaluated {len(actions)} sequences")

        all_actions.append(actions)
        all_portfolio.append(portfolio_returns)
        all_market.append(market_returns)
        all_net.append(np.array(net_returns))
        y_pred_list.append(y_pred)
        label_list.append(test_labels)

    actions           = np.concatenate(all_actions)
    portfolio_returns = np.concatenate(all_portfolio)
    market_returns    = np.concatenate(all_market)
    net_returns       = np.concatenate(all_net)
    y_pred            = torch.cat(y_pred_list)
    test_labels       = torch.cat(label_list)

    TestingLSTM(y_pred, test_labels)

    print(f"\n--- LSTM Without costs ---")
    print(f"Cumulative return : {portfolio_returns.sum():.4f}")
    print(f"Sharpe ratio : {portfolio_returns.mean() / (portfolio_returns.std() + 1e-8):.4f}")
    print(f"Sortino ratio : {sortino_ratio(portfolio_returns):.4f}")
    print(f"Max Drawdown : {max_drawdown(portfolio_returns):.4f}")

    print(f"\n--- LSTM With 10 bps ---")
    print(f"Cumulative return : {net_returns.sum():.4f}")
    print(f"Sharpe ratio : {net_returns.mean() / (net_returns.std() + 1e-8):.4f}")
    print(f"Sortino ratio : {sortino_ratio(net_returns):.4f}")
    print(f"Max Drawdown : {max_drawdown(net_returns):.4f}")