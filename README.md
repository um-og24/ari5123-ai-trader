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
Run the Streamlit app: `streamlit run src/main.py`

Access the interface at `http://localhost:8501`.

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
