# Predictive vs. Reactive Paradigms in Multi-Asset Trading
*MSc Artificial Intelligence Dissertation — CS5099, University of St Andrews*

## Description

This repository contains the implementation for my MSc dissertation
*"Predictive vs. Reactive Paradigms in Multi-Asset Trading"*. It compares two trading paradigms across
three asset classes (stocks, bonds, crypto): a predictive LSTM model
trained to forecast log-returns from technical indicators, and a
reactive Deep Reinforcement Learning agent (PPO) trained directly on
risk-adjusted trading performance.

Both approaches are benchmarked against classical baselines (Buy & Hold, Random, Momentum) and stress-tested under feature noise and adversarial volatility shocks.

## Set up the environment

This project uses a Conda environment to manage the required dependencies.

```
# Activate the env.
conda activate trading

# Install the requirement.txt.
pip install -r config/requirements.txt

# Save the requirements after adding a new library.
pip freeze > config/requirements.txt
```

As this project was developed on the university's SSH cluster, the commands below show how to
request a GPU node and activate the environment after logging in.(*with the school GPU cluster*)

After SSH login, request a GPU node and activate the environment
```
srun -p gpu-l4-n2 -q gpu-l4-n2 --gpus 1 --pty bash
conda activate trading
python path/to/file.py
```

## Dependencies

- Python 3.11

See `config/requirement.txt`.

## Project Pipeline

1. `src/data/fetch.py` — fetch raw OHLCV data from the Alpaca API
2. `src/data/features.py` — resample to hourly and compute technical indicators
3. `src/models_predictive/LSTM_V1.py` — train the LSTM model
4. `src/models_reactive/DRL.py` — train the PPO trading agent
5. (a) `src/models_predictive/evaluate.py`
5. (b) `src/models_reactive/evaluate.py` — evaluate each model individually
6. `src/benchmark/benchmark.py` — compare all models/baselines and run stress tests

## Project directory Tree-map

```
.
├── config
│   └── requirements.txt
├── CS5099_paper
│   ├── DOER.pdf
│   └── Plan_and_context_survey_V1.pdf
├── data
│   ├── external
│   ├── processed
│   │   ├── bonds
│   │   │   └── TLT.parquet
│   │   ├── crypto
│   │   │   └── BTC-USD.parquet
│   │   └── stocks
│   │       └── AAPL.parquet
│   └── raw
│       ├── bonds
│       │   └── TLT.parquet
│       ├── crypto
│       │   └── BTC-USD.parquet
│       └── stocks
│           └── AAPL.parquet
├── LICENSE.md
├── Miniconda3-latest-Linux-x86_64.sh
├── model
│   ├── DRL_old
│   │   ├── drl_v1_100K.zip
│   │   ├── drl_v1-1_1M_fix.zip
│   │   ├── drl_v1_1M.zip
│   │   └── drl_v2_1M.zip
│   ├── drl_v3_1000000_steps.zip
│   ├── drl_v3_100000_steps.zip
│   ├── drl_v3_10000_steps.zip
│   ├── drl_v3_110000_steps.zip
│   ├── drl_v3_120000_steps.zip
│   ├── drl_v3_130000_steps.zip
│   ├── drl_v3_140000_steps.zip
│   ├── drl_v3_150000_steps.zip
│   ├── drl_v3_160000_steps.zip
│   ├── drl_v3_170000_steps.zip
│   ├── drl_v3_180000_steps.zip
│   ├── drl_v3_190000_steps.zip
│   ├── drl_v3_200000_steps.zip
│   ├── drl_v3_20000_steps.zip
│   ├── drl_v3_210000_steps.zip
│   ├── drl_v3_220000_steps.zip
│   ├── drl_v3_230000_steps.zip
│   ├── drl_v3_240000_steps.zip
│   ├── drl_v3_250000_steps.zip
│   ├── drl_v3_260000_steps.zip
│   ├── drl_v3_270000_steps.zip
│   ├── drl_v3_280000_steps.zip
│   ├── drl_v3_290000_steps.zip
│   ├── drl_v3_300000_steps.zip
│   ├── drl_v3_30000_steps.zip
│   ├── drl_v3_310000_steps.zip
│   ├── drl_v3_320000_steps.zip
│   ├── drl_v3_330000_steps.zip
│   ├── drl_v3_340000_steps.zip
│   ├── drl_v3_350000_steps.zip
│   ├── drl_v3_360000_steps.zip
│   ├── drl_v3_370000_steps.zip
│   ├── drl_v3_380000_steps.zip
│   ├── drl_v3_390000_steps.zip
│   ├── drl_v3_400000_steps.zip
│   ├── drl_v3_40000_steps.zip
│   ├── drl_v3_410000_steps.zip
│   ├── drl_v3_420000_steps.zip
│   ├── drl_v3_430000_steps.zip
│   ├── drl_v3_440000_steps.zip
│   ├── drl_v3_450000_steps.zip
│   ├── drl_v3_460000_steps.zip
│   ├── drl_v3_470000_steps.zip
│   ├── drl_v3_480000_steps.zip
│   ├── drl_v3_490000_steps.zip
│   ├── drl_v3_500000_steps.zip
│   ├── drl_v3_50000_steps.zip
│   ├── drl_v3_510000_steps.zip
│   ├── drl_v3_520000_steps.zip
│   ├── drl_v3_530000_steps.zip
│   ├── drl_v3_540000_steps.zip
│   ├── drl_v3_550000_steps.zip
│   ├── drl_v3_560000_steps.zip
│   ├── drl_v3_570000_steps.zip
│   ├── drl_v3_580000_steps.zip
│   ├── drl_v3_590000_steps.zip
│   ├── drl_v3_600000_steps.zip
│   ├── drl_v3_60000_steps.zip
│   ├── drl_v3_610000_steps.zip
│   ├── drl_v3_620000_steps.zip
│   ├── drl_v3_630000_steps.zip
│   ├── drl_v3_640000_steps.zip
│   ├── drl_v3_650000_steps.zip
│   ├── drl_v3_660000_steps.zip
│   ├── drl_v3_670000_steps.zip
│   ├── drl_v3_680000_steps.zip
│   ├── drl_v3_690000_steps.zip
│   ├── drl_v3_700000_steps.zip
│   ├── drl_v3_70000_steps.zip
│   ├── drl_v3_710000_steps.zip
│   ├── drl_v3_720000_steps.zip
│   ├── drl_v3_730000_steps.zip
│   ├── drl_v3_740000_steps.zip
│   ├── drl_v3_750000_steps.zip
│   ├── drl_v3_760000_steps.zip
│   ├── drl_v3_770000_steps.zip
│   ├── drl_v3_780000_steps.zip
│   ├── drl_v3_790000_steps.zip
│   ├── drl_v3_800000_steps.zip
│   ├── drl_v3_80000_steps.zip
│   ├── drl_v3_810000_steps.zip
│   ├── drl_v3_820000_steps.zip
│   ├── drl_v3_830000_steps.zip
│   ├── drl_v3_840000_steps.zip
│   ├── drl_v3_850000_steps.zip
│   ├── drl_v3_860000_steps.zip
│   ├── drl_v3_870000_steps.zip
│   ├── drl_v3_880000_steps.zip
│   ├── drl_v3_890000_steps.zip
│   ├── drl_v3_900000_steps.zip
│   ├── drl_v3_90000_steps.zip
│   ├── drl_v3_910000_steps.zip
│   ├── drl_v3_920000_steps.zip
│   ├── drl_v3_930000_steps.zip
│   ├── drl_v3_940000_steps.zip
│   ├── drl_v3_950000_steps.zip
│   ├── drl_v3_960000_steps.zip
│   ├── drl_v3_970000_steps.zip
│   ├── drl_v3_980000_steps.zip
│   ├── drl_v3_990000_steps.zip
│   ├── drl_v3_best.zip
│   ├── drl_v3_final.zip
│   ├── evaluations.npz
│   ├── LSTM_old
│   │   └── lstm_v2.pth
│   └── lstm_v1.pth
├── README.md
└── src
    ├── benchmark
    │   ├── benchmark.py
    │   ├── img
    │   │   ├── cumulative_returns.png
    │   │   ├── DRL
    │   │   │   ├── evaluation_DRL_Action.png
    │   │   │   └── training_evolution.png
    │   │   ├── LSTM
    │   │   │   └── predictions.png
    │   │   ├── max_drawdown.png
    │   │   ├── noise_sensitivity.png
    │   │   ├── sharpe.png
    │   │   ├── shock_test-separated.png
    │   │   └── sortino.png
    │   └── tlb
    │       ├── benchmark_results.txt
    │       ├── best_vs_final_results.txt
    │       ├── DRL_resultat.txt
    │       ├── LSTM_resultat.txt
    │       ├── shock_test_results.txt
    │       └── stress_test_results.txt
    ├── data
    │   ├── features.py
    │   ├── fetch.py
    │   └── size.py
    ├── models_predictive
    │   ├── evaluate.py
    │   ├── LSTM_V1.py
    │   ├── LSTM_V2.py
    │   └── __pycache__
    │       ├── evaluate.cpython-311.pyc
    │       └── LSTM_V1.cpython-311.pyc
    └── models_reactive
        ├── DRL.py
        ├── evaluate.py
        └── __pycache__
            └── DRL.cpython-311.pyc

26 directories, 148 files
```
Generated using 
```bash
sudo apt-get install tree

tree
```

## Author

I'd like to thank my dissertation supervisor Ognjen Arandelovic for his advice and help during this dissertation period, which enabled me to complete this project.

* [Alexandre Puiseux](https://github.com/AlexandrePuiseux65) - Author
* [Ognjen Arandelovic](https://www.st-andrews.ac.uk/computer-science/people/oa7/) ([Google Scholar](https://scholar.google.com/citations?user=D7bpRJ8AAAAJ&hl=en)) - Supervisor


# Acknowledgments
## Resources
> CodeSignal. — *Optimizing LSTM Models for Time Series Forecasting with PyTorch*, Last access 2026-06-27

> Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M., & Dormann, N. (2021).
> *Stable-Baselines3: Reliable Reinforcement Learning Implementations*. Journal of Machine
> Learning Research, 22(268), 1-8.
> http://jmlr.org/papers/v22/20-1364.html

> Towers, M., et al. (2024). *Gymnasium: A Standard Interface for Reinforcement Learning
> Environments*.
> https://github.com/Farama-Foundation/Gymnasium


## Dataset
> Alpaca API
> https://alpaca.markets/

## License

In the context of the master dissertation, this tool was developed for research purpuse. For additional information, please refer to the license provided.

[LICENSE](LICENSE.md)

## Citation

If you use this code in your own work, please cite it:

> Alexandre Puiseux (2026). *Predictive vs. Reactive Paradigms Across Asset Classes*. MSc Dissertation, University of St Andrews.