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
request a GPU node and activate the environment after logging in.

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
│   │   ├── crypto
│   │   └── stocks
│   └── raw
│       ├── bonds
│       ├── crypto
│       └── stocks
├── Miniconda3-latest-Linux-x86_64.sh
├── model
│   ├── drl_v1_100K.zip
│   ├── drl_v1_1M.zip
│   ├── drl_v2_1M.zip
│   ├── evaluations.npz
│   ├── lstm_v1.pth
│   └── lstm_v2.pth
├── README
└── src
    ├── benchmark
    │   ├── benchmark.py
    │   ├── img
    │   └── tlb
    ├── data
    │   ├── features.py
    │   └── fetch.py
    ├── models_predictive
    │   ├── evaluate.py
    │   ├── LSTM_V1.py
    │   └── LSTM_V2.py
    └── models_reactive
        ├── DRL.py
        └── evaluate.py

20 directories, 21 files
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