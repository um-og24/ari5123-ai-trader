# ARI5123 - Ensemble DQN+RF Trading System

## Overview
This repository contains the Python implementation of an ensemble DQN+RF trading system for AAPL stock, as described in the ARI5123 Assignment 2 paper.

## Virtual Environment Recommended (Python 3.12.3)
Create via: `python3 -m venv .venv`

Activate via (Linux): `source .venv\bin\activate`

Activate via (Windows): `source .venv\Source\activate`

Upgrade pip with: `pip install --upgrade pip`

## Dependencies
Install via: `pip install -r requirements.txt`

## Execution
Run the Streamlit app: `streamlit run src/main.py --server.port 8506`

Access the interface at `http://localhost:8506`.

## Structure
- `main.py`: Streamlit app entry point.
- `ensemble_agent.py`: Ensemble DQN+RF logic.
- `dqn_agent.py`: DQN implementation.
- `rf_agent.py`: RF implementation.
- `portfolio_tracker.py`: Portfolio management.
- `calculations.py`: Metrics and indicators.
- `utils.py`: Data preprocessing.
- `overview_ui.py`, `chart_builder.py`, `*_ui.py`: UI components.

## Data
Use AAPL training data from 2017-01-01 – 2022-12-31

Live Trading Simulation use AAPL form 2023-01-01 - 2025-05-25


## Settings for Reproducibility

- `models\AAPL\AAPL_ensemble_settings.json`:
```json
{
    "ticker": "AAPL",
    "training_start_date": "2017-01-01T00:00:00",
    "training_end_date": "2022-12-31T00:00:00",
    "lookback": 14,
    "batch_size": 32,
    "initial_cash": 1000,
    "trade_fee": 5e-05,
    "risk_per_trade": 0.02,
    "max_trades_per_epoch": 0,
    "max_fee_per_epoch": 0,
    "atr_multiplier": 1.5,
    "atr_period": 14,
    "atr_smoothing": true,
    "use_smote": false,
    "dqn_weight_scale": 0.6,
    "capital_type": "Explicit Value",
    "reference_capital": null,
    "capital_percentage": null,
    "epochs": 50,
    "confirmation_steps": 1
}
```


## Docker Image


If `docker-compose.yml` file is available, Then:-

Build the image locally: `docker-compose up --build -d`

Else:-

Login (make sure Docker Desktop is running): `docker login`

Build and push onto Docker hub: `docker build -t owengauci24/ari5123-ai-trader-app:latest . --push`

, or to make it platform independent: `docker buildx build --platform linux/amd64,linux/arm64 -t owengauci24/ari5123-ai-trader-app:latest . --push`

Download a copy: `docker pull owengauci24/ari5123-ai-trader-app:latest`

Run the image: `docker run -it -p 8506:8506 owengauci24/ari5123-ai-trader-app:latest`

