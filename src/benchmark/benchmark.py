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
from models_reactive.DRL import PreparationData as PrepDRL, TradingEnv, FILE_SAVE_MODEL, FEATURE_COLS
from models_reactive.DRL import FILE_PATH_STOCKS_PROCESSED, FILE_PATH_BONDS_PROCESSED, FILE_PATH_CRYPTO_PROCESSED

from models_predictive.evaluate import LoadModel as LoadLSTM, PrepareTestData as PrepLSTM
from models_predictive.evaluate import max_drawdown, sortino_ratio

FILE_SAVE_IMG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img")
FILE_SAVE_TLB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tlb")

NOISE_STD = 10 #Realy big

SHOCK_WINDOWS = [
    (500, 50, "AAPL"),
    (1500, 50, "TLT"),
    (3500, 50, "BTC"),
]

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

def Turnover(actions):
    actions = np.array(actions)
    return np.abs(np.diff(actions)).mean()

# --- Graphs ---#
def cumulative_return(strategies):
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab10.colors

    for i, (name, (gross, net, actions)) in enumerate(strategies.items()):
        color = colors[i]
        plt.plot(np.cumsum(gross), label=f"{name} (gross)", color=color, linestyle='-')
        plt.plot(np.cumsum(net), label=f"{name} (10bps)", color=color, linestyle='--', alpha=0.6)

    plt.title("Cumulative Returns (Gross vs Net of Costs)")
    plt.xlabel("Time steps") #(hourly, concatenated: AAPL + TLT + BTC-USD)
    plt.ylabel("Cumulative log-return") #(sum, not %)
    plt.legend(fontsize=8)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.savefig(os.path.join(FILE_SAVE_IMG, "cumulative_returns.png"))
    plt.close()
    print("Graph printed -> 'Cumulative Returns'")

def sharpe_graph(strategies, names):
    sharpe = [Sharpe(g) for g, _, _ in strategies.values()]
    plt.figure(figsize=(10, 6))
    plt.bar(names, sharpe, color='steelblue')
    plt.title("Sharpe Ratio")
    plt.axhline(0, color='black', linewidth=0.5)
    plt.savefig(os.path.join(FILE_SAVE_IMG, "sharpe.png"))
    plt.close()
    print("Graph printed -> 'Sharpe Ratio'")

def sortino_graph(strategies, names):
    sortino = [Sortino(g) for g, _, _ in strategies.values()]
    plt.figure(figsize=(10, 6))
    plt.bar(names, sortino, color='seagreen')
    plt.title("Sortino Ratio")
    plt.axhline(0, color='black', linewidth=0.5)
    plt.savefig(os.path.join(FILE_SAVE_IMG, "sortino.png"))
    plt.close()
    print("Graph printed -> 'Sortino Ratio'")

def max_drawdown_graph(strategies, names):
    mdd = [MaxDrawdown(g) for g, _, _ in strategies.values()]
    plt.figure(figsize=(10, 6))
    plt.bar(names, mdd, color='tomato')
    plt.title("Max Drawdown")
    plt.savefig(os.path.join(FILE_SAVE_IMG, "max_drawdown.png"))
    plt.close()
    print("Graph printed -> 'Max Drawdown'")

def GraphAndMetric(strategies):
    lines = []
    header = f"{'Strategy':<12} {'Return':>8} {'Return(10bps)':>14} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'Turnover':>10}"
    lines.append(header)
    lines.append("-" * 75)

    for name, (gross, net, actions) in strategies.items():
        line = (f"{name:<12} {CumulativeReturn(gross):>8.4f} {CumulativeReturn(net):>14.4f} "
                f"{Sharpe(gross):>8.4f} {Sortino(gross):>8.4f} {MaxDrawdown(gross):>8.4f} "
                f"{Turnover(actions):>10.4f}")
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

# --- Stress-test --- #
def add_noise(data, std, seed=42):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, std, size=data.shape)
    return data + noise

def evaluate_drl(model, env):
    obs, _ = env.reset()
    env.current_step = 24
    portfolio, actions_list, market = [], [], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        current_price = env.close_prices[env.current_step]
        next_price = env.close_prices[env.current_step + 1]
        log_return = np.log(next_price / current_price)
        portfolio.append(action[0] * log_return)
        actions_list.append(action[0])
        market.append(log_return)
        obs, reward, done, trunc, info = env.step(action)
    return np.array(portfolio), np.array(actions_list), np.array(market)

def GraphAndMetric_noisy(stress_strategies):
    lines = []
    header = f"{'Model':<14} {'Return':>8} {'Return(10bps)':>14} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'Turnover':>10}"
    lines.append(header)
    lines.append("-" * 75)
    for name, (gross, net, actions) in stress_strategies.items():
        lines.append(f"{name:<14} {CumulativeReturn(gross):>8.4f} {CumulativeReturn(net):>14.4f} "
                      f"{Sharpe(gross):>8.4f} {Sortino(gross):>8.4f} {MaxDrawdown(gross):>8.4f} "
                      f"{Turnover(actions):>10.4f}")

    stress_output = "\n".join(lines)
    print(f"\n{stress_output}")

    with open(os.path.join(FILE_SAVE_TLB, "stress_test_results.txt"), "w") as f:
        f.write(stress_output)

    # --- Graph: cumulative returns clean vs noisy --- #
    plt.figure(figsize=(12, 8))
    for name, (gross, _, _) in stress_strategies.items():
        plt.plot(np.cumsum(gross), label=name)
    plt.title("Stress Test: Clean vs Noisy Features")
    plt.xlabel("Time steps")
    plt.ylabel("Cumulative log-return")
    plt.legend()
    plt.axhline(0, color='black', linewidth=0.5)
    plt.savefig(os.path.join(FILE_SAVE_IMG, "stress_test.png"))
    plt.close()

    # --- Graph: illustration of noise on a single feature (zoomed) --- #
    window_start, window_end = 1000, 1200
    plt.figure(figsize=(12, 4))
    plt.plot(test_all['SMA_20'].values[window_start:window_end], label='Clean (SMA_20)', alpha=0.8)
    plt.plot(test_all_noisy['SMA_20'].values[window_start:window_end], label='Noisy (SMA_20)', alpha=0.8)
    plt.title("Illustration: Effect of Gaussian Noise on SMA_20")
    plt.legend()
    plt.savefig(os.path.join(FILE_SAVE_IMG, "noise_illustration.png"))
    plt.close()

    print("Stress test complete.")

# --- Volatility Shock Test --- #
def apply_shock(returns, start, duration, shock_value=-0.05):
    shocked = np.array(returns).copy()
    shocked[start:start+duration] = shock_value
    return shocked

def adversarial_shock_value(actions, start, duration, magnitude=0.05):
    # Find moment when the bot has a long position.
    avg_action = np.mean(actions[start:start+duration])
    return -magnitude if avg_action > 0 else magnitude

def run_shock_test(actions, market_or_labels, shock_windows, magnitude=0.05):
    results = []
    for start, duration, asset_name in shock_windows:
        shock_val = adversarial_shock_value(actions, start, duration, magnitude)
        shocked_returns = apply_shock(market_or_labels, start, duration, shock_val)
        shocked_portfolio = actions * shocked_returns
        results.append((asset_name, start, duration, shocked_portfolio))
    return results

def GraphAndMetric_shock(shock_strategies, shock_windows):
    lines = []
    header = f"{'Model':<20} {'Return':>8} {'Return(10bps)':>14} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'Turnover':>10}"
    lines.append(header)
    lines.append("-" * 82)
    for name, (gross, net, actions) in shock_strategies.items():
        lines.append(f"{name:<20} {CumulativeReturn(gross):>8.4f} {CumulativeReturn(net):>14.4f} "
                      f"{Sharpe(gross):>8.4f} {Sortino(gross):>8.4f} {MaxDrawdown(gross):>8.4f} "
                      f"{Turnover(actions):>10.4f}")

    shock_output = "\n".join(lines)
    print(f"\n{shock_output}")

    with open(os.path.join(FILE_SAVE_TLB, "shock_test_results.txt"), "w") as f:
        f.write(shock_output)

    plt.figure(figsize=(12, 8))
    for name, (gross, _, _) in shock_strategies.items():
        plt.plot(np.cumsum(gross), label=name)
    for start, duration, asset_name in shock_windows:
        plt.axvspan(start, start + duration, color='red', alpha=0.15)
    plt.title("Volatility Shock Test (adversarial, multi-window)")
    plt.xlabel("Time steps")
    plt.ylabel("Cumulative log-return")
    plt.legend(fontsize=8)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.savefig(os.path.join(FILE_SAVE_IMG, "shock_test.png"))
    plt.close()
    print("Shock test complete.")

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
        'DRL':      (drl_portfolio, drl_net, drl_actions),
        'LSTM':     (lstm_portfolio, lstm_net, lstm_actions),
        'Momentum': (momentum_portfolio, momentum_net, momentum_actions),
        'Random':   (random_portfolio, random_net, random_actions),
        'Buy&Hold': (drl_market, drl_market, np.ones(len(drl_market))),
    }

    GraphAndMetric(strategies)

    # --- Stress-test --- #
    test_all_noisy = test_all.copy()

    # Add noise to the data and revaluate DRL
    test_all_noisy[FEATURE_COLS] = add_noise(test_all_noisy[FEATURE_COLS].values, std=NOISE_STD)
    env_noisy = TradingEnv(test_all_noisy)

    drl_noisy_portfolio, drl_noisy_actions, drl_noisy_market = evaluate_drl(model_DRL, env_noisy)
    drl_noisy_net = ComputeNetReturns(drl_noisy_portfolio, drl_noisy_actions)

    # Add noise to the data and revaluate LSTM
    test_seqs_noisy = test_seqs.clone()
    test_seqs_noisy[:,:,:-1] = torch.tensor(
        add_noise(test_seqs_noisy[:,:,:-1].numpy(), std=NOISE_STD), dtype=torch.float32
    )

    with torch.no_grad():
        y_pred_noisy = model_LSTM(test_seqs_noisy.to(device)).squeeze().cpu()

    lstm_noisy_actions = torch.sign(y_pred_noisy).numpy()
    lstm_noisy_portfolio = lstm_noisy_actions * test_labels.numpy()
    lstm_noisy_net = ComputeNetReturns(lstm_noisy_portfolio, lstm_noisy_actions)

    # --- Stress Test Comparison --- #
    stress_strategies = {
        'DRL (clean)':  (drl_portfolio, drl_net, drl_actions),
        'DRL (noisy)':  (drl_noisy_portfolio, drl_noisy_net, drl_noisy_actions),
        'LSTM (clean)': (lstm_portfolio, lstm_net, lstm_actions),
        'LSTM (noisy)': (lstm_noisy_portfolio, lstm_noisy_net, lstm_noisy_actions),
    }
    GraphAndMetric_noisy(stress_strategies)

    # --- Volatility Shock Test (multi-window and adversarial) --- #
    drl_shock_results  = run_shock_test(drl_actions, drl_market, SHOCK_WINDOWS, magnitude=abs(SHOCK_VALUE))
    lstm_shock_results = run_shock_test(lstm_actions, test_labels.numpy(), SHOCK_WINDOWS, magnitude=abs(SHOCK_VALUE))

    shock_strategies = {
        'DRL (clean)':  (drl_portfolio, drl_net, drl_actions),
        'LSTM (clean)': (lstm_portfolio, lstm_net, lstm_actions),
    }

    for asset_name, start, duration, shocked_portfolio in drl_shock_results:
        net = ComputeNetReturns(shocked_portfolio, drl_actions)
        shock_strategies[f'DRL (shock@{asset_name})'] = (shocked_portfolio, net, drl_actions)

    for asset_name, start, duration, shocked_portfolio in lstm_shock_results:
        net = ComputeNetReturns(shocked_portfolio, lstm_actions)
        shock_strategies[f'LSTM (shock@{asset_name})'] = (shocked_portfolio, net, lstm_actions)

    GraphAndMetric_shock(shock_strategies, SHOCK_WINDOWS)