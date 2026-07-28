# --- Lib --- #
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import torch
import pandas as pd

# --- Path setup --- #
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # to src/
sys.path.append(SRC_DIR)

# --- Imports from models --- #
from models_reactive.DRL import PreparationData as PrepDRL, TradingEnv, FILE_SAVE_MODEL
from models_reactive.DRL import FILE_PATH_STOCKS_PROCESSED, FILE_PATH_BONDS_PROCESSED, FILE_PATH_CRYPTO_PROCESSED

from models_predictive.evaluate import LoadModel as LoadLSTM, PrepareTestData as PrepLSTM
from models_predictive.evaluate import max_drawdown, sortino_ratio

FILE_SAVE_IMG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img")
FILE_SAVE_TLB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tlb")

# --- Metrics --- #
def Sharpe(returns):
    return returns.mean() / (returns.std() + 1e-8)

def Sortino(returns, target=0.0):
    excess = returns - target
    downside = np.where(excess < 0, excess, 0)
    downside_std = np.sqrt(np.mean(downside**2)) + 1e-8
    return excess.mean() / downside_std

def MaxDrawdown(returns):
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    return (cumulative - running_max).min()

def CumulativeReturn(returns):
    return np.sum(returns)

def ComputeNetReturns(returns, actions, cost_rate=0.001):
    net = []
    prev = 0.0
    for i, act in enumerate(actions):
        cost = abs(act - prev) * cost_rate
        net.append(returns[i] - cost)
        prev = act
    return np.array(net)

# --- Graphs ---#
def cumulative_return(strategies):
    plt.figure(figsize=(12,8))
    for name, (gross, net) in strategies.items():
        plt.plot(np.cumsum(gross), label=f"{name} (gross)")
        plt.plot(np.cumsum(net), label=f"{name} (10bps)", linestyle='--', alpha=0.6)
    plt.title("Cumulative Returns (Gross vs Net of Costs)")
    plt.legend(fontsize=8)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.grid()
    plt.savefig(os.path.join(FILE_SAVE_IMG, "cumulative_returns.png"))
    plt.close()
    print("Graph printed -> 'Cumulative Returns'")

def sharpe_graph(strategies, names):
    sharpe = [Sharpe(g) for g, _ in strategies.values()]
    plt.figure(figsize=(10, 6))
    plt.bar(names, sharpe, color='steelblue')
    plt.title("Sharpe Ratio")
    plt.axhline(0, color='black', linewidth=0.5)
    plt.grid()
    plt.savefig(os.path.join(FILE_SAVE_IMG, "sharpe.png"))
    plt.close()
    print("Graph printed -> 'Sharpe Ratio'")

def sortino_graph(strategies, names):
    sortino = [Sortino(g) for g, _ in strategies.values()]
    plt.figure(figsize=(10, 6))
    plt.bar(names, sortino, color='seagreen')
    plt.title("Sortino Ratio")
    plt.axhline(0, color='black', linewidth=0.5)
    plt.grid()
    plt.savefig(os.path.join(FILE_SAVE_IMG, "sortino.png"))
    plt.close()
    print("Graph printed -> 'Sortino Ratio'")

def max_drawdown_graph(strategies, names):
    mdd = [MaxDrawdown(g) for g, _ in strategies.values()]
    plt.figure(figsize=(10, 6))
    plt.bar(names, mdd, color='tomato')
    plt.title("Max Drawdown")
    plt.grid()
    plt.savefig(os.path.join(FILE_SAVE_IMG, "max_drawdown.png"))
    plt.close()
    print("Graph printed -> 'Max Drawdown'")

def GraphAndMetric(strategies):
    lines = []
    header = f"{'Strategy':<12} {'Return':>8} {'Return(10bps)':>14} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8}"
    lines.append(header)
    lines.append("-" * 65)

    for name, (gross, net) in strategies.items():
        line = (f"{name:<12} {CumulativeReturn(gross):>8.4f} {CumulativeReturn(net):>14.4f} "
                f"{Sharpe(gross):>8.4f} {Sortino(gross):>8.4f} {MaxDrawdown(gross):>8.4f}")
        lines.append(line)

    output = "\n".join(lines)
    print(f"\n{output}")

    with open(os.path.join(FILE_SAVE_TLB, "benchmark_results.txt"), "w") as f:
        f.write(output)

    names = list(strategies.keys())

    # --- Graphs --- #
    cumulative_return(strategies)
    sharpe_graph(strategies, names)
    sortino_graph(strategies, names)
    max_drawdown_graph(strategies, names)

# --- Main --- # 
if __name__ == "__main__":
    train_stocks, val_stocks, test_stocks = PrepDRL(FILE_PATH_STOCKS_PROCESSED, "AAPL.parquet", asset_type=0)
    train_bonds,  val_bonds,  test_bonds  = PrepDRL(FILE_PATH_BONDS_PROCESSED,  "TLT.parquet",  asset_type=1)
    train_crypto, val_crypto, test_crypto = PrepDRL(FILE_PATH_CRYPTO_PROCESSED, "BTC-USD.parquet", asset_type=2)

    test_all = pd.concat([test_stocks, test_bonds, test_crypto]).reset_index(drop=True)
    env = TradingEnv(test_all)

    # DRL model
    model_DRL = PPO.load(os.path.join(FILE_SAVE_MODEL, "drl_v1_1M.zip"), env=env, device='cpu')

    # LSTM model
    model_LSTM, device = LoadLSTM()
    test_seqs, test_labels = PrepLSTM(FILE_STOCKS="AAPL.parquet", FILE_BONDS="TLT.parquet", FILE_CRYPTO="BTC-USD.parquet")

    # --- DRL Evaluation --- #
    obs, _ = env.reset()
    env.current_step = 24
    drl_portfolio = []
    drl_actions = []
    drl_market = []
    prev_action = 0.0
    done = False

    while not done:
        action, _ = model_DRL.predict(obs, deterministic=True)

        current_price = env.close_prices[env.current_step]
        next_price = env.close_prices[env.current_step + 1]
        log_return = np.log(next_price / current_price)
        portfolio_return = action[0] * log_return

        obs, reward, done, trunc, info = env.step(action)
        drl_portfolio.append(portfolio_return)
        drl_actions.append(action[0])
        drl_market.append(log_return)
        prev_action = action[0]

    drl_portfolio = np.array(drl_portfolio)
    drl_market = np.array(drl_market)
    drl_net = ComputeNetReturns(drl_portfolio, drl_actions)

    # --- LSTM Evaluation --- #
    with torch.no_grad():
        y_pred = model_LSTM(test_seqs.to(device)).squeeze().cpu()

    lstm_actions = torch.sign(y_pred).numpy()
    lstm_portfolio = lstm_actions * test_labels.numpy()
    lstm_net = ComputeNetReturns(lstm_portfolio, lstm_actions)

    # --- Random Strategy --- #
    np.random.seed(42)
    random_actions = np.random.uniform(-1, 1, len(drl_market))
    random_portfolio = random_actions * drl_market
    random_net = ComputeNetReturns(random_portfolio, random_actions)

    # --- Momentum Strategy --- #
    window = 24
    momentum_actions = []
    for i in range(len(drl_market)):
        if i < window:
            momentum_actions.append(0.0)
        else:
            past = drl_market[i-window:i]
            momentum_actions.append(1.0 if past.sum() > 0 else -1.0)
    momentum_actions = np.array(momentum_actions)
    momentum_portfolio = momentum_actions * drl_market
    momentum_net = ComputeNetReturns(momentum_portfolio, momentum_actions)
    
    # --- Graphs & Metrics --- #
    strategies = {
        'DRL': (drl_portfolio, drl_net),
        'LSTM': (lstm_portfolio, lstm_net),
        'Momentum': (momentum_portfolio, momentum_net),
        'Random': (random_portfolio, random_net),
        'Buy&Hold': (drl_market, drl_market),
    }

    GraphAndMetric(strategies)