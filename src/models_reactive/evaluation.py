# --- Lib --- #
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# --- Global Variable --- #
from DRL import PreparationData, TradingEnv, FEATURE_COLS, FILE_SAVE_MODEL
from DRL import FILE_PATH_STOCKS_PROCESSED, FILE_PATH_BONDS_PROCESSED, FILE_PATH_CRYPTO_PROCESSED

# --- Function --- #
def evaluation_DRL_Action(portfolio_returns, market_returns, actions):
    actions = np.array(actions)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    axes[0].plot(np.cumsum(portfolio_returns), label="DRL", color="blue")
    axes[0].plot(np.cumsum(market_returns), label="Buy & Hold", color="orange")
    axes[0].legend()
    axes[0].set_title("Returns cumulatifs")

    axes[1].plot(actions, color="green", alpha=0.5)
    axes[1].axhline(0, color='black', linewidth=0.5)
    axes[1].set_title("Actions du modèle (-1=short, +1=long)")

    axes[2].hist(actions, bins=50, color="purple")
    axes[2].set_title("Distribution des actions")

    plt.tight_layout()
    plt.savefig(os.path.join(FILE_SAVE_MODEL, "evaluation_DRL_Action.png"))

    print(f"Action moyenne : {actions.mean():.4f}")
    print(f"% en short (action < 0) : {(actions < 0).mean():.2%}")
    print(f"% en long  (action > 0) : {(actions > 0).mean():.2%}")

def plot_training_evolution(model_dir):
    data = np.load(os.path.join(model_dir, "evaluations.npz"))
    timesteps = data["timesteps"]
    rewards = data["results"].mean(axis=1)

    plt.figure(figsize=(10, 4))
    plt.plot(timesteps, rewards)
    plt.axhline(0, color='red', linestyle='--', linewidth=0.8)
    plt.title("Évolution du reward moyen pendant l'entraînement")
    plt.xlabel("Timesteps")
    plt.ylabel("Mean reward (val set)")
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "training_evolution.png"))
    plt.show()

def max_drawdown(returns):
    cumulative = np.cumsum(returns)
    running_max= np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    return drawdown.min()

def sortino_ratio(returns, target=0.0):
    excess = returns - target
    downside = np.where(excess < 0, excess, 0)
    downside_std = np.sqrt(np.mean(downside**2)) + 1e-8
    return excess.mean()/downside_std

# --- Main --- # 
if __name__ == "__main__":
    # Prepartion of the data
    train_stocks, val_stocks, test_stocks = PreparationData(FILE_PATH_STOCKS_PROCESSED, "AAPL.parquet", asset_type=0)
    train_bonds, val_bonds, test_bonds  = PreparationData(FILE_PATH_BONDS_PROCESSED,  "TLT.parquet",  asset_type=1)
    train_crypto, val_crypto, test_crypto = PreparationData(FILE_PATH_CRYPTO_PROCESSED, "BTC-USD.parquet", asset_type=2)

    test_all = pd.concat([test_stocks, test_bonds, test_crypto]).reset_index(drop=True)
    env = TradingEnv(test_all)

    # Load the DRL models
    model = PPO.load(os.path.join(FILE_SAVE_MODEL, "drl_v1_1M.zip"), env=env, device='cpu')

    # add!
    obs, _ = env.reset()
    env.current_step = 24
    rewards = []
    actions = []
    portfolio_returns = []
    market_returns = []
    net_returns = []
    prev_action = 0.0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        
        current_price = env.close_prices[env.current_step]
        next_price    = env.close_prices[env.current_step + 1]
        log_return    = np.log(next_price / current_price)
        portfolio_return = action[0] * log_return
        
        # Transaction cost
        position_change   = abs(action[0] - prev_action)
        transaction_cost  = position_change * 0.001
        net_return        = portfolio_return - transaction_cost
        prev_action       = action[0]
        
        obs, reward, done, trunc, info = env.step(action)
        rewards.append(reward)
        actions.append(action[0])
        portfolio_returns.append(portfolio_return)
        net_returns.append(net_return)
        market_returns.append(log_return)

    evaluation_DRL_Action(portfolio_returns, market_returns, actions)
    plot_training_evolution(FILE_SAVE_MODEL)

    portfolio_returns = np.array(portfolio_returns)
    net_returns       = np.array(net_returns)
    market_returns    = np.array(market_returns)

    print(f"\n--- Without costs ---")
    print(f"Cumulative return  : {portfolio_returns.sum():.4f}")
    print(f"Sharpe ratio       : {portfolio_returns.mean() / (portfolio_returns.std() + 1e-8):.4f}")
    print(f"Sortino ratio      : {sortino_ratio(portfolio_returns):.4f}")
    print(f"Max Drawdown       : {max_drawdown(portfolio_returns):.4f}")

    print(f"\n--- With 10 bps ---")
    print(f"Cumulative return  : {net_returns.sum():.4f}")
    print(f"Sharpe ratio       : {net_returns.mean() / (net_returns.std() + 1e-8):.4f}")
    print(f"Sortino ratio      : {sortino_ratio(net_returns):.4f}")
    print(f"Max Drawdown       : {max_drawdown(net_returns):.4f}")

    print(f"\n--- Buy & Hold ---")
    print(f"Cumulative return  : {market_returns.sum():.4f}")
    print(f"Sharpe ratio       : {market_returns.mean() / (market_returns.std() + 1e-8):.4f}")
    print(f"Sortino ratio      : {sortino_ratio(market_returns):.4f}")
    print(f"Max Drawdown       : {max_drawdown(market_returns):.4f}")