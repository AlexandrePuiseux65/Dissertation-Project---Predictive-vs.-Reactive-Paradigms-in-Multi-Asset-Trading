'''
    Benchmark and stress-testing suite comparing the DRL and LSTM trading
    models against baseline strategies (Buy & Hold, Random, Momentum),
    including robustness tests under feature noise (multiple magnitudes)
    and adversarial volatility shocks. All evaluation is done per asset
    (AAPL, TLT, BTC) and results are combined afterwards - raw price data
    is never concatenated across assets, only already-valid per-asset
    returns are.
'''

# --- Lib --- #
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import torch

# --- Path setup --- #
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # to src/
sys.path.append(SRC_DIR)

# --- Imports from models --- #
from models_reactive.DRL import PreparationData as PrepDRL, TradingEnv, FILE_SAVE_MODEL, FEATURE_COLS, WINDOW_SIZE
from models_reactive.DRL import FILE_PATH_STOCKS_PROCESSED, FILE_PATH_BONDS_PROCESSED, FILE_PATH_CRYPTO_PROCESSED

from models_predictive.evaluate import LoadModel as LoadLSTM, PrepareTestData as PrepLSTM

FILE_SAVE_IMG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img")
FILE_SAVE_TLB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tlb")

ASSETS = [
    ("AAPL", FILE_PATH_STOCKS_PROCESSED, "AAPL.parquet", 0),
    ("TLT",  FILE_PATH_BONDS_PROCESSED,  "TLT.parquet",  1),
    ("BTC",  FILE_PATH_CRYPTO_PROCESSED, "BTC-USD.parquet", 2),
]

NOISE_LEVELS = [0.05, 0.10, 0.25, 0.50]

SHOCK_WINDOWS = {
    "AAPL": (200, 50),
    "TLT":  (200, 50),
    "BTC":  (1000, 50),
}
SHOCK_MAGNITUDE = 0.05

# --- Metrics --- #
def Sharpe(returns):
    '''
        Compute the Sharpe ratio:
            - risk-adjusted return, calculated as the mean return divided
              by its standard deviation (assumes a risk-free rate of 0).
    '''
    return returns.mean() / (returns.std() + 1e-8)

def Sortino(returns, target=0.0):
    '''
        Compute the Sortino ratio:
            - risk-adjusted return metric that measures performance
              relative to downside volatility only (ignores upside variance).
    '''
    excess = returns - target
    downside = np.where(excess < 0, excess, 0)
    downside_std = np.sqrt(np.mean(downside**2)) + 1e-8
    return excess.mean() / downside_std

def MaxDrawdown(returns):
    '''
        Compute the Maximum Drawdown (MDD):
            - the largest drop in cumulative log-return from a running
              peak to a subsequent trough, before a new high is reached.
    '''
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    return (cumulative - running_max).min()

def CumulativeReturn(returns):
    '''
        Compute the total cumulative log-return over the full period.
    '''
    return np.sum(returns)

def ComputeNetReturns(returns, actions, cost_rate=0.001):
    '''
        Compute returns net of transaction costs:
            - a cost proportional to the change in position size is
              subtracted at each step (cost_rate expressed in decimal form,
              e.g. 0.001 = 10 basis points). Position always starts flat
              (prev=0) - call this once per asset, never on data already
              concatenated across assets.
    '''
    net = []
    prev = 0.0
    for i, act in enumerate(actions):
        cost = abs(act - prev) * cost_rate
        net.append(returns[i] - cost)
        prev = act
    return np.array(net)

def Turnover(actions):
    '''
        Compute the turnover of the trading strategy:
            - the average absolute change in position between consecutive
              steps, used as a proxy for trading frequency/stability.
    '''
    actions = np.array(actions)
    return np.abs(np.diff(actions)).mean()

# --- Graphs ---#
def cumulative_return(strategies):
    '''
        Plot and save the cumulative return over time for each strategy,
        showing both gross returns and returns net of transaction costs.
        Time steps are the concatenation of AAPL, then TLT, then BTC
        per-asset results (each internally valid, no cross-asset return).
    '''
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab10.colors

    for i, (name, (gross, net, actions)) in enumerate(strategies.items()):
        color = colors[i]
        plt.plot(np.cumsum(gross), label=f"{name} (gross)", color=color, linestyle='-')
        plt.plot(np.cumsum(net), label=f"{name} (10bps)", color=color, linestyle='--', alpha=0.6)

    plt.title("Cumulative Returns (Gross vs Net of Costs)")
    plt.xlabel("Time steps (AAPL, then TLT, then BTC)")
    plt.ylabel("Cumulative log-return")
    plt.legend(fontsize=8)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.savefig(os.path.join(FILE_SAVE_IMG, "cumulative_returns.png"))
    plt.close()
    print("Graph printed -> 'Cumulative Returns'")

def sharpe_graph(strategies, names):
    '''
        Plot and save a bar chart comparing the Sharpe ratio of each strategy.
    '''
    sharpe = [Sharpe(g) for g, _, _ in strategies.values()]
    plt.figure(figsize=(10, 6))
    plt.bar(names, sharpe, color='steelblue')
    plt.title("Sharpe Ratio")
    plt.axhline(0, color='black', linewidth=0.5)
    plt.savefig(os.path.join(FILE_SAVE_IMG, "sharpe.png"))
    plt.close()
    print("Graph printed -> 'Sharpe Ratio'")

def sortino_graph(strategies, names):
    '''
        Plot and save a bar chart comparing the Sortino ratio of each strategy.
    '''
    sortino = [Sortino(g) for g, _, _ in strategies.values()]
    plt.figure(figsize=(10, 6))
    plt.bar(names, sortino, color='seagreen')
    plt.title("Sortino Ratio")
    plt.axhline(0, color='black', linewidth=0.5)
    plt.savefig(os.path.join(FILE_SAVE_IMG, "sortino.png"))
    plt.close()
    print("Graph printed -> 'Sortino Ratio'")

def max_drawdown_graph(strategies, names):
    '''
        Plot and save a bar chart comparing the Max Drawdown of each strategy.
    '''
    mdd = [MaxDrawdown(g) for g, _, _ in strategies.values()]
    plt.figure(figsize=(10, 6))
    plt.bar(names, mdd, color='tomato')
    plt.title("Max Drawdown")
    plt.savefig(os.path.join(FILE_SAVE_IMG, "max_drawdown.png"))
    plt.close()
    print("Graph printed -> 'Max Drawdown'")

def GraphAndMetric(strategies):
    '''
        Print and save the benchmark metrics table (clean/baseline scenario,
        no noise or shocks), then generate all associated summary graphs.
    '''
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

# --- Per-asset evaluation helpers --- #
def evaluate_drl(model, test_df):
    '''
        Run one full evaluation episode of the DRL agent on a single
        asset's test set (its own TradingEnv - never a concatenation of
        several assets), using deterministic actions. Returns the
        portfolio returns, the sequence of actions taken, and the
        underlying market (buy-and-hold) returns for that asset.
    '''
    env = TradingEnv(test_df)
    obs, _ = env.reset()
    env.current_step = WINDOW_SIZE
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

def momentum_strategy(market, window=WINDOW_SIZE):
    '''
        Momentum baseline for a single asset: goes long if the sum of the
        past `window` returns was positive, short otherwise. Computed per
        asset so the lookback window never spans two different assets.
    '''
    actions = []
    for i in range(len(market)):
        if i < window:
            actions.append(0.0)
        else:
            past = market[i-window:i]
            actions.append(1.0 if past.sum() > 0 else -1.0)
    actions = np.array(actions)
    portfolio = actions * market
    return portfolio, actions

# --- Stress-test: noise injection --- #
def add_noise(data, level, seed=42):
    '''
        Inject Gaussian noise into the given data, scaled by each
        feature's own standard deviation (measured on the data passed
        in), so noise is proportionate regardless of the feature's scale.
        `level` is a fraction of that std (e.g. 0.1 = 10% of the feature's
        own std). Works for both 2D (rows, features) and 3D
        (sequences, timesteps, features) arrays.
    '''
    rng = np.random.default_rng(seed)
    reduce_axes = tuple(range(data.ndim - 1))
    col_std = data.std(axis=reduce_axes, keepdims=True)
    noise = rng.normal(0, 1, size=data.shape) * (col_std * level)
    return data + noise

def noise_sweep(model_DRL, model_LSTM, device, drl_test_dfs, lstm_test_data):
    '''
        Re-evaluate DRL and LSTM at several noise levels (NOISE_LEVELS),
        combining results across the three assets at each level. Returns
        {model_name: {level: (gross, net, actions)}}.
    '''
    results = {"DRL": {}, "LSTM": {}}

    for level in NOISE_LEVELS:
        # --- DRL --- #
        drl_gross, drl_actions_all = [], []
        for name, test_df in drl_test_dfs:
            noisy_df = test_df.copy()
            noisy_df[FEATURE_COLS] = add_noise(noisy_df[FEATURE_COLS].values, level)
            gross, actions, _ = evaluate_drl(model_DRL, noisy_df)
            drl_gross.append(gross)
            drl_actions_all.append(actions)
        drl_gross = np.concatenate(drl_gross)
        drl_actions_all = np.concatenate(drl_actions_all)
        drl_net = ComputeNetReturns(drl_gross, drl_actions_all)
        results["DRL"][level] = (drl_gross, drl_net, drl_actions_all)

        # --- LSTM --- #
        lstm_gross, lstm_actions_all = [], []
        for name, test_seqs, test_labels in lstm_test_data:
            seqs_noisy = test_seqs.clone()
            seqs_noisy[:, :, :-1] = torch.tensor(
                add_noise(seqs_noisy[:, :, :-1].numpy(), level), dtype=torch.float32
            )
            with torch.no_grad():
                y_pred_noisy = model_LSTM(seqs_noisy.to(device)).squeeze().cpu()
            actions = torch.sign(y_pred_noisy).numpy()
            gross = actions * test_labels.numpy()
            lstm_gross.append(gross)
            lstm_actions_all.append(actions)
        lstm_gross = np.concatenate(lstm_gross)
        lstm_actions_all = np.concatenate(lstm_actions_all)
        lstm_net = ComputeNetReturns(lstm_gross, lstm_actions_all)
        results["LSTM"][level] = (lstm_gross, lstm_net, lstm_actions_all)

    return results

def GraphAndMetric_noisy(results, clean_strategies):
    '''
        Print and save the metrics table for the noise stress-test across
        all NOISE_LEVELS (clean vs. each noise level, for DRL and LSTM),
        then plot Sharpe ratio as a function of noise level for both
        models.
    '''
    lines = []
    header = f"{'Model':<18} {'Return':>8} {'Return(10bps)':>14} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'Turnover':>10}"
    lines.append(header)
    lines.append("-" * 80)

    for model_name in ["DRL", "LSTM"]:
        gross, net, actions = clean_strategies[model_name]
        lines.append(f"{model_name + ' (clean)':<18} {CumulativeReturn(gross):>8.4f} {CumulativeReturn(net):>14.4f} "
                      f"{Sharpe(gross):>8.4f} {Sortino(gross):>8.4f} {MaxDrawdown(gross):>8.4f} "
                      f"{Turnover(actions):>10.4f}")
        for level, (gross, net, actions) in results[model_name].items():
            label = f"{model_name} (noise {level:.0%})"
            lines.append(f"{label:<18} {CumulativeReturn(gross):>8.4f} {CumulativeReturn(net):>14.4f} "
                          f"{Sharpe(gross):>8.4f} {Sortino(gross):>8.4f} {MaxDrawdown(gross):>8.4f} "
                          f"{Turnover(actions):>10.4f}")

    stress_output = "\n".join(lines)
    print(f"\n{stress_output}")

    with open(os.path.join(FILE_SAVE_TLB, "stress_test_results.txt"), "w") as f:
        f.write(stress_output)

    plt.figure(figsize=(10, 6))
    for model_name in ["DRL", "LSTM"]:
        levels = [0.0] + list(results[model_name].keys())
        sharpes = [Sharpe(clean_strategies[model_name][0])] + \
                  [Sharpe(results[model_name][l][0]) for l in results[model_name]]
        plt.plot([l * 100 for l in levels], sharpes, marker='o', label=model_name)
    plt.title("Sharpe Ratio vs Noise Level")
    plt.xlabel("Noise level (% of each feature's own std)")
    plt.ylabel("Sharpe ratio")
    plt.axhline(0, color='black', linewidth=0.5)
    plt.legend()
    plt.savefig(os.path.join(FILE_SAVE_IMG, "noise_sensitivity.png"))
    plt.close()
    print("Noise sweep complete.")

# --- Volatility Shock Test --- #
def apply_shock(returns, start, duration, shock_value):
    '''
        Overwrite a window of returns with a fixed shock value, simulating
        a sudden market move (e.g. a crash or a rally) over that window.
    '''
    shocked = np.array(returns).copy()
    shocked[start:start+duration] = shock_value
    return shocked

def adversarial_shock_value(actions, start, duration, magnitude=0.05):
    '''
        Determine the direction of the shock so that it always works
        against the position the model held during the window: a crash
        if the average position was long, or a rally if it was short.

        This ensures the shock is adversarial rather than accidentally
        benefiting whichever position the model happened to hold.
    '''
    avg_action = np.mean(actions[start:start+duration])
    return -magnitude if avg_action > 0 else magnitude

def run_shock_test(actions, market_or_labels, start, duration, magnitude=0.05):
    '''
        Apply an adversarial shock to a single (start, duration) window
        within one asset's own returns array, and recompute the resulting
        portfolio returns for that window.
    '''
    shock_val = adversarial_shock_value(actions, start, duration, magnitude)
    shocked_returns = apply_shock(market_or_labels, start, duration, shock_val)
    return actions * shocked_returns

def GraphAndMetric_shock(shock_strategies):
    '''
        Print and save the metrics table for the volatility shock test
        (clean vs. shocked performance for each model/asset window), then
        plot cumulative returns for DRL and LSTM in separate subplots.
    '''
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

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    for name, (gross, _, _) in shock_strategies.items():
        if name.startswith('DRL'):
            axes[0].plot(np.cumsum(gross), label=name)
        elif name.startswith('LSTM'):
            axes[1].plot(np.cumsum(gross), label=name)

    axes[0].set_title("DRL: Clean vs Adversarial Shocks (per asset)")
    axes[1].set_title("LSTM: Clean vs Adversarial Shocks (per asset)")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_ylabel("Cumulative log-return")
    axes[1].set_xlabel("Time steps (within each asset's own test set)")

    plt.tight_layout()
    plt.savefig(os.path.join(FILE_SAVE_IMG, "shock_test-separated.png"))
    plt.close()
    print("Shock test complete.")

# --- Best vs Final model comparison --- #
def evaluate_checkpoint(model, drl_test_dfs):
    '''
        Evaluate a single DRL checkpoint across all three assets (each in
        its own environment), returning the combined gross/net returns
        and actions, exactly as done for the main DRL evaluation above.
    '''
    gross_all, actions_all = [], []
    for name, test_df in drl_test_dfs:
        gross, actions, _ = evaluate_drl(model, test_df)
        gross_all.append(gross)
        actions_all.append(actions)
    gross_all = np.concatenate(gross_all)
    actions_all = np.concatenate(actions_all)
    net_all = ComputeNetReturns(gross_all, actions_all)
    return gross_all, net_all, actions_all

def GraphAndMetric_bestfinal(best_strategy, final_strategy):
    '''
        Print and save a metrics comparison table between the "best"
        (validation-selected) and "final" (end-of-training) DRL
        checkpoints, both evaluated on the same test data.
    '''
    lines = []
    header = f"{'Checkpoint':<10} {'Return':>8} {'Return(10bps)':>14} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'Turnover':>10}"
    lines.append(header)
    lines.append("-" * 70)
    for label, (gross, net, actions) in [("Best", best_strategy), ("Final", final_strategy)]:
        lines.append(f"{label:<10} {CumulativeReturn(gross):>8.4f} {CumulativeReturn(net):>14.4f} "
                      f"{Sharpe(gross):>8.4f} {Sortino(gross):>8.4f} {MaxDrawdown(gross):>8.4f} "
                      f"{Turnover(actions):>10.4f}")
    output = "\n".join(lines)
    print(f"\n{output}")
    with open(os.path.join(FILE_SAVE_TLB, "best_vs_final_results.txt"), "w") as f:
        f.write(output)

# --- Main --- #
if __name__ == "__main__":
    drl_test_dfs = [(name, PrepDRL(path, fname, asset_type)[2]) for name, path, fname, asset_type in ASSETS]

    model_DRL = PPO.load(os.path.join(FILE_SAVE_MODEL, "drl_v3_best.zip"),
                          env=TradingEnv(drl_test_dfs[0][1]), device='cpu')

    model_DRL_final = PPO.load(os.path.join(FILE_SAVE_MODEL, "drl_v3_final.zip"),
                                env=TradingEnv(drl_test_dfs[0][1]), device='cpu')

    best_strategy  = evaluate_checkpoint(model_DRL, drl_test_dfs)
    final_strategy = evaluate_checkpoint(model_DRL_final, drl_test_dfs)
    GraphAndMetric_bestfinal(best_strategy, final_strategy)

    model_LSTM, device = LoadLSTM()
    lstm_test_data = PrepLSTM(FILE_STOCKS="AAPL.parquet", FILE_BONDS="TLT.parquet", FILE_CRYPTO="BTC-USD.parquet")

    # --- Per-asset evaluation (DRL, LSTM, Random, Momentum, Buy&Hold) --- #
    drl_gross_all, drl_actions_all, drl_market_all = [], [], []
    lstm_gross_all, lstm_actions_all = [], []
    random_gross_all, random_actions_all = [], []
    momentum_gross_all, momentum_actions_all = [], []

    per_asset_drl = {}
    per_asset_lstm = {}

    for name, test_df in drl_test_dfs:
        gross, actions, market = evaluate_drl(model_DRL, test_df)
        per_asset_drl[name] = (gross, actions, market)
        drl_gross_all.append(gross)
        drl_actions_all.append(actions)
        drl_market_all.append(market)

        np.random.seed(42)
        random_actions = np.random.uniform(-1, 1, len(market))
        random_gross_all.append(random_actions * market)
        random_actions_all.append(random_actions)

        mom_gross, mom_actions = momentum_strategy(market)
        momentum_gross_all.append(mom_gross)
        momentum_actions_all.append(mom_actions)

    for name, test_seqs, test_labels in lstm_test_data:
        with torch.no_grad():
            y_pred = model_LSTM(test_seqs.to(device)).squeeze().cpu()
        actions = torch.sign(y_pred).numpy()
        gross = actions * test_labels.numpy()
        per_asset_lstm[name] = (gross, actions, test_labels.numpy())
        lstm_gross_all.append(gross)
        lstm_actions_all.append(actions)

    drl_portfolio = np.concatenate(drl_gross_all)
    drl_actions   = np.concatenate(drl_actions_all)
    drl_market    = np.concatenate(drl_market_all)
    drl_net       = ComputeNetReturns(drl_portfolio, drl_actions)

    lstm_portfolio = np.concatenate(lstm_gross_all)
    lstm_actions   = np.concatenate(lstm_actions_all)
    lstm_net       = ComputeNetReturns(lstm_portfolio, lstm_actions)

    random_portfolio = np.concatenate(random_gross_all)
    random_actions   = np.concatenate(random_actions_all)
    random_net       = ComputeNetReturns(random_portfolio, random_actions)

    momentum_portfolio = np.concatenate(momentum_gross_all)
    momentum_actions   = np.concatenate(momentum_actions_all)
    momentum_net       = ComputeNetReturns(momentum_portfolio, momentum_actions)

    # --- Graphs & Metrics --- #
    strategies = {
        'DRL':      (drl_portfolio, drl_net, drl_actions),
        'LSTM':     (lstm_portfolio, lstm_net, lstm_actions),
        'Momentum': (momentum_portfolio, momentum_net, momentum_actions),
        'Random':   (random_portfolio, random_net, random_actions),
        'Buy&Hold': (drl_market, drl_market, np.ones(len(drl_market))),
    }

    GraphAndMetric(strategies)

    # --- Stress-test: noise sweep --- #
    noise_results = noise_sweep(model_DRL, model_LSTM, device, drl_test_dfs, lstm_test_data)
    clean_for_noise = {
        "DRL":  (drl_portfolio, drl_net, drl_actions),
        "LSTM": (lstm_portfolio, lstm_net, lstm_actions),
    }
    GraphAndMetric_noisy(noise_results, clean_for_noise)

    # --- Volatility Shock Test (adversarial, one window per asset) --- #
    shock_strategies = {
        'DRL (clean)':  (drl_portfolio, drl_net, drl_actions),
        'LSTM (clean)': (lstm_portfolio, lstm_net, lstm_actions),
    }

    for asset_name, (gross, actions, market) in per_asset_drl.items():
        start, duration = SHOCK_WINDOWS[asset_name]
        shocked = run_shock_test(actions, market, start, duration, SHOCK_MAGNITUDE)
        net = ComputeNetReturns(shocked, actions)
        shock_strategies[f'DRL (shock@{asset_name})'] = (shocked, net, actions)

    for asset_name, (gross, actions, labels) in per_asset_lstm.items():
        start, duration = SHOCK_WINDOWS[asset_name]
        shocked = run_shock_test(actions, labels, start, duration, SHOCK_MAGNITUDE)
        net = ComputeNetReturns(shocked, actions)
        shock_strategies[f'LSTM (shock@{asset_name})'] = (shocked, net, actions)

    GraphAndMetric_shock(shock_strategies)