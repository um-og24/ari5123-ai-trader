import os
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import sys
sys.dont_write_bytecode = True

import streamlit as st
st.set_page_config(page_title="Ensemble AI Auto Trader", layout="wide")

import pandas as pd

from ensemble_agent import EnsembleAgent
from portfolio_tracker import PortfolioTracker
from utils import Utils
from overview_ui import render_overview
from training_ui import render_training
from trading_ui import render_live_trading
from portfolio_ui import render_portfolio
from trading_logs_ui import render_trade_log
from system_logs_ui import render_system_logs

DEFAULT_TICKER="AAPL"

def _initialize_session_state():
    defaults = {
        'simulation_running': False,
        'simulation_paused': False,
        'training_running': False,
        'training_paused': False,
        'current_index': 0,
        'actions': None,
        'trade_log': [],
        'portfolio_tracker': PortfolioTracker(initial_cash=1000),
        'last_trade_id': None,
        'recent_actions': [],
        'action_counter': 0,
        'user_settings': None,
        'agent': None,
        'progress_bar': None,
        'system_logs_container': None,
        'system_logs': [],
        'status_text': None,
        'rf_metrics': None,
        'rf_training_active': False,
        'trading_start_date': pd.Timestamp.now().date() - pd.Timedelta(days=365),
        'trading_end_date': pd.Timestamp.now().date() - pd.Timedelta(days=1),
        'trading_simulation_delay': 5
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _render_sidebar(saved_settings=None):
    with st.sidebar:
        st.title("Settings")

        main_tab, training_tab, trading_tab, misc_tab=st.tabs(["Main", "Training", "Trading", "Misc"])

        with main_tab:
            st.subheader("Instrument")
            default_ticker = saved_settings['ticker'] if saved_settings and saved_settings['ticker'] is not None else DEFAULT_TICKER
            ticker = st.text_input("Stock Ticker", value=default_ticker.upper())
            ticker=ticker.upper()

            st.divider()

            st.subheader("Wealth")
            default_capital_type = saved_settings['capital_type'] if saved_settings and saved_settings['capital_type'] is not None else 'Explicit Value'
            capital_type = st.radio("Initial Capital Type", ["Explicit Value", "Percentage"], index=0 if default_capital_type == 'Explicit Value' else 1, horizontal=True)

            if capital_type == "Explicit Value":
                default_initial_cash = saved_settings['initial_cash'] if saved_settings and saved_settings['initial_cash'] is not None else 1000.0
                initial_cash = st.number_input("Initial Capital (€)", 100.0, 100000.0, float(default_initial_cash))
                reference_capital = None
                capital_percentage = None
            else:
                default_reference_capital = saved_settings['reference_capital'] if saved_settings and saved_settings['reference_capital'] is not None else 1000.0
                reference_capital = st.number_input("Reference Capital (€)", 100.0, 1000000.0, float(default_reference_capital))
                default_capital_percentage = saved_settings['capital_percentage'] * 100 if saved_settings and saved_settings['capital_percentage'] is not None else 100
                capital_percentage = st.slider("Capital Percentage (%)", 0, 100, int(default_capital_percentage), step=5) / 100
                if capital_percentage is None:
                    st.error("Capital percentage cannot be None. Using default value of 100%.")
                    capital_percentage = 1
                initial_cash = reference_capital * capital_percentage
                if initial_cash < 100:
                    st.warning("Computed initial capital is too low (< €100). Please increase reference capital or percentage.")

            st.divider()

        with training_tab:
            st.subheader("Training Parameters")

            st.subheader("Date Range")
            default_training_start_date = saved_settings['training_start_date'] if saved_settings and saved_settings['training_start_date'] is not None else pd.Timestamp.now().date() - pd.Timedelta(days=365)
            training_start_date = st.date_input("Start Date", value=default_training_start_date, key="training_start_date")
            default_training_end_date = saved_settings['training_end_date'] if saved_settings and saved_settings['training_end_date'] is not None else pd.Timestamp.now().date() - pd.Timedelta(days=1)
            default_training_end_date = default_training_end_date if default_training_end_date > training_start_date else pd.Timestamp(training_start_date).date() + pd.Timedelta(days=365)
            training_end_date = st.date_input("End Date", value=default_training_end_date, key="training_end_date")
            if training_end_date <= training_start_date:
                training_end_date = pd.Timestamp(training_end_date).date() + pd.Timedelta(days=365)
                st.warning("**End Date shifted forward by 365 days.")

            st.divider()

            st.subheader("Epochs")
            default_epochs = saved_settings['epochs'] if saved_settings and saved_settings['epochs'] is not None else 20
            epochs = st.slider("Training Epochs", 1, 50, default_epochs)
            default_lookback = saved_settings['lookback'] if saved_settings and saved_settings['lookback'] is not None else 30
            lookback = st.slider("Lookback Window (days)", 10, 60, default_lookback)
            default_max_trades_per_epoch = saved_settings['max_trades_per_epoch'] if saved_settings and saved_settings['max_trades_per_epoch'] is not None else 10000
            max_trades_per_epoch = st.number_input("Max Trades per Epoch (0 = unlimited)", 0, 100000, default_max_trades_per_epoch)
            default_max_fee_per_epoch = saved_settings['max_fee_per_epoch'] if saved_settings and saved_settings['max_fee_per_epoch'] is not None else 10000
            max_fee_per_epoch = st.number_input("Max Fee per Epoch (0 = unlimited)", 0, 100000, default_max_fee_per_epoch)

            st.divider()

            st.subheader("ATR Parameters")
            default_atr_multiplier = saved_settings['atr_multiplier'] if saved_settings and saved_settings['atr_multiplier'] is not None else 1.5
            atr_multiplier = st.number_input("ATR Multiplier", 0.5, 5.0, default_atr_multiplier, step=0.1)
            default_atr_period = saved_settings['atr_period'] if saved_settings and saved_settings['atr_period'] is not None else 14
            atr_period = st.number_input("ATR Period (days)", 5, 50, default_atr_period)
            default_atr_smoothing = saved_settings['atr_smoothing'] if saved_settings and saved_settings['atr_smoothing'] is not None else True
            atr_smoothing = st.toggle("Enable ATR Smoothing (EMA)", value=default_atr_smoothing)

            st.divider()

            st.subheader("SMOTE")
            default_use_smote = saved_settings['use_smote'] if saved_settings and saved_settings['use_smote'] is not None else False
            use_smote = st.toggle("Perform SMOTE", value=default_use_smote, key="use_smote")
            
            st.divider()

        with trading_tab:
            st.subheader("Trading Parameters")

            default_trade_fee = saved_settings['trade_fee'] * 100 if saved_settings and saved_settings['trade_fee'] is not None else 0.1
            trade_fee = st.number_input("Transaction Fee (%)", 0.0, 1.0, default_trade_fee) / 100
            default_risk_per_trade = saved_settings['risk_per_trade'] * 100 if saved_settings and saved_settings['risk_per_trade'] is not None else 1.0
            risk_per_trade = st.number_input("Risk per Trade (%)", 0.1, 10.0, default_risk_per_trade) / 100
            default_confirmation_steps = saved_settings['confirmation_steps'] if saved_settings and saved_settings['confirmation_steps'] is not None else 1
            confirmation_steps = st.slider("Signal Confirmation Steps", 1, 5, default_confirmation_steps, help="Number of consecutive identical Buy/Sell signals required to execute a trade.")

            st.divider()

            st.subheader("Bias")
            default_dqn_weight_scale = saved_settings['dqn_weight_scale'] * 100 if saved_settings and saved_settings['dqn_weight_scale'] is not None else 60
            dqn_weight_scale = st.slider("DQN Agent Bias Level (%) *", 0, 100, value=int(default_dqn_weight_scale), help="Percentage level to favour DQN when ensemble fails to decide.") / 100
            if dqn_weight_scale > 0.5:
                bias_message="*DQN Agent preferred"
            elif dqn_weight_scale == 0.5:
                bias_message="*Equal preference"
            else:
                bias_message="*RF Agent preferred"
            st.warning(bias_message)

            st.divider()

            st.subheader("Simulation Delay")
            default_trading_simulation_delay = saved_settings['trading_simulation_delay'] if saved_settings and saved_settings['trading_simulation_delay'] is not None else 5
            trading_simulation_delay = st.slider("Trading Simulation Delay (seconds)", 1, 60, default_trading_simulation_delay)

            st.divider()
        
        with misc_tab:
            st.subheader("Model Storage")
            default_model_dir = saved_settings['model_dir'].replace(f"\\{ticker}", "") if saved_settings and saved_settings['model_dir'] is not None else f'models'
            model_dir = st.text_input("Model Directory", value=default_model_dir.replace(f"\\{ticker}", "")).replace(f"\\{ticker}", "")

            st.divider()

        if st.session_state.portfolio_tracker.initial_cash != initial_cash:
            st.session_state.portfolio_tracker = PortfolioTracker(initial_cash=initial_cash)

        settings = {
            'ticker': ticker.upper(),
            'training_start_date': training_start_date,
            'training_end_date': training_end_date,
            'model_dir': model_dir,
            'capital_type': capital_type,
            'initial_cash': st.session_state.portfolio_tracker.initial_cash,
            'reference_capital': reference_capital,
            'capital_percentage': capital_percentage,
            'trade_fee': trade_fee,
            'lookback': lookback,
            'risk_per_trade': risk_per_trade,
            'epochs': epochs,
            'max_trades_per_epoch': max_trades_per_epoch,
            'max_fee_per_epoch': max_fee_per_epoch,
            'confirmation_steps': confirmation_steps,
            'dqn_weight_scale': dqn_weight_scale,
            'atr_multiplier': atr_multiplier,
            'atr_period': atr_period,
            'atr_smoothing': atr_smoothing,
            'use_smote': use_smote,
            'trading_simulation_delay': trading_simulation_delay
        }

    return settings

def _render_main_ui(agent, settings):
    st.title(f"Ensemble AI Auto Trader - {agent.ticker if agent else DEFAULT_TICKER}")
    
    has_enough_data = agent and agent.has_enough_training_data()

    if has_enough_data:
        tabs = st.tabs(["Overview", "Training", "Live Trading", "Portfolio", "Trade Log", "System Logs"])
    else:
        tabs = st.tabs(["Overview", "System Logs"])

    with tabs[0]:
        render_overview(agent, settings)

    with tabs[-1]:
        render_system_logs()

    if has_enough_data:
        with tabs[1]:
            render_training(agent, settings)

        with tabs[3]:
            render_portfolio(agent, settings)
        with tabs[4]:
            render_trade_log(agent, settings)
        with tabs[2]:
            render_live_trading(agent, settings)



def _main():
    st.session_state.user_settings = _render_sidebar(st.session_state.user_settings)
        
    if st.session_state.agent is None or st.session_state.agent.ticker != st.session_state.user_settings['ticker']:
        try:
            st.session_state.agent = EnsembleAgent.init_a_new_agent(st.session_state.user_settings)
        except Exception as e:
            st.error(f"Failed to initialize agent: {e}")
            Utils.log_message(f"ERROR: Agent initialization failed: {e}")
            st.session_state.agent = None

    _render_main_ui(st.session_state.agent, st.session_state.user_settings)




if __name__ == "__main__":
    _initialize_session_state()
    
    _main()
