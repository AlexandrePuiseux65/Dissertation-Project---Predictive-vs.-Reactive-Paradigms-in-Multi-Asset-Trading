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
    rewards   = data["results"].mean(axis=1)

    plt.figure(figsize=(10, 4))
    plt.plot(timesteps, rewards)
    plt.axhline(0, color='red', linestyle='--', linewidth=0.8)
    plt.title("Évolution du reward moyen pendant l'entraînement")
    plt.xlabel("Timesteps")
    plt.ylabel("Mean reward (val set)")
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "training_evolution.png"))
    plt.show()

# --- Main --- # 
if __name__ == "__main__":
    train_stocks, val_stocks, test_stocks = PreparationData(FILE_PATH_STOCKS_PROCESSED, "AAPL.parquet", asset_type=0)
    train_bonds,  val_bonds,  test_bonds  = PreparationData(FILE_PATH_BONDS_PROCESSED,  "TLT.parquet",  asset_type=1)
    train_crypto, val_crypto, test_crypto = PreparationData(FILE_PATH_CRYPTO_PROCESSED, "BTC-USD.parquet", asset_type=2)

    test_all = pd.concat([test_stocks, test_bonds, test_crypto]).reset_index(drop=True)
    env = TradingEnv(test_all)

    model = PPO.load(os.path.join(FILE_SAVE_MODEL, "best_model"), env=env, device='cpu')

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

    rewards = np.array(rewards)
    net_returns = np.array(net_returns)
    print(f"Cumulative reward : {rewards.sum():.4f}")
    print(f"Cumulative return (with 10bps) : {net_returns.sum():.4f}")
    print(f"Mean reward : {rewards.mean():.6f}")
    print(f"Sharpe ratio : {rewards.mean() / (rewards.std() + 1e-8):.4f}")