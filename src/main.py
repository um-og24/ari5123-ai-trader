import os
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import sys
sys.dont_write_bytecode = True

import streamlit as st
st.set_page_config(page_title="Ensemble AI Auto Trader", layout="wide")

import pandas as pd

from utils import Utils
from ensemble_agent import EnsembleAgent
from portfolio_tracker import PortfolioTracker
from overview_ui import render_overview
from training_ui import render_training
from trading_ui import render_live_trading
from portfolio_ui import render_portfolio
from trading_logs_ui import render_trade_log
from system_logs_ui import render_system_logs


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
        'trading_simulation_delay': 5
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _render_sidebar(saved_settings=None):
    with st.sidebar:
        st.title("Settings")

        main_tab, training_tab, trading_tab, misc_tab=st.tabs(["Main", "Trainng", "Trading", "Misc"])
        with main_tab:
            st.subheader("Instrument")
            default_ticker = saved_settings['ticker'] if saved_settings and saved_settings['ticker'] is not None else 'AAPL'
            ticker = st.text_input("Stock Ticker", value=default_ticker.upper())
            ticker=ticker.upper()

            st.divider()
            
            st.subheader("Date Range")
            default_start_date = saved_settings['start_date'] if saved_settings and saved_settings['start_date'] is not None else pd.Timestamp.now().date() - pd.Timedelta(days=365)
            default_end_date = saved_settings['end_date'] if saved_settings and saved_settings['end_date'] is not None else pd.Timestamp.now().date() - pd.Timedelta(days=1)
            start_date = st.date_input("Start Date", value=default_start_date)
            end_date = st.date_input("End Date", value=default_end_date)
    
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
                default_capital_percentage = saved_settings['capital_percentage'] if saved_settings and saved_settings['capital_percentage'] is not None else 100.0
                capital_percentage = st.number_input("Capital Percentage (%)", 0.1, 100.0, float(default_capital_percentage))
                if capital_percentage is None:
                    st.error("Capital percentage cannot be None. Using default value of 100%.")
                    capital_percentage = 100.0
                initial_cash = reference_capital * (capital_percentage / 100.0)
                if initial_cash < 100:
                    st.warning("Computed initial capital is too low (< €100). Please increase reference capital or percentage.")

            st.divider()
        
        with training_tab:
            st.subheader("Training Parameters")
            default_epochs = saved_settings['epochs'] if saved_settings and saved_settings['epochs'] is not None else 10
            epochs = st.slider("Training Epochs", 1, 50, default_epochs)
            default_lookback = saved_settings['lookback'] if saved_settings and saved_settings['lookback'] is not None else 14
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
            atr_smoothing = st.checkbox("Enable ATR Smoothing (EMA)", value=default_atr_smoothing)
        
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
            'start_date': start_date,
            'end_date': end_date,
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
            'trading_simulation_delay': trading_simulation_delay
        }
        
        if st.button('Apply changes', type="primary", use_container_width=True):
            st.session_state.user_settings = settings
            st.session_state.agent = None  # Will force creation of new instance with new settings

    return settings

def _render_main_ui(agent, settings):
    has_enough_data = agent.has_enough_data()    
    if has_enough_data:
        tabs = st.tabs(["Overview", "Training", "Live Trading", "Portfolio", "Trade Log", "System Logs"])
        # st.session_state.system_logs_container=tabs[5].empty()
        # if tabs[5].button("Clear Logs", use_container_width=True):
        #     st.session_state.system_logs=[]
    else:
        tabs = st.tabs(["Overview", "System Logs"])
        # st.session_state.system_logs_container=tabs[1].empty()
        # if tabs[1].button("Clear Logs"):
        #     st.session_state.system_logs=[]

    with tabs[-1]:
        render_system_logs()

    with tabs[0]:
        render_overview(agent, settings)

    if has_enough_data:
        with tabs[1]:
            render_training(agent, settings)
        with tabs[2]:
            render_live_trading(agent, settings)
        with tabs[3]:
            render_portfolio(agent, settings)
        with tabs[4]:
            render_trade_log(agent, settings)


def _get_EnsembleAgent(settings):
    def init_new_ensemble_agent(settings):
        Utils.log_message(f"INFO: Initializing new EnsembleAgent: {settings}")
        return EnsembleAgent(
            ticker=settings['ticker'],
            start_date=settings['start_date'],
            end_date=settings['end_date'],
            model_dir=settings['model_dir'],
            lookback=settings['lookback'],
            trade_fee=settings['trade_fee'],
            initial_cash=settings['initial_cash'],
            risk_per_trade=settings['risk_per_trade'],
            max_trades_per_epoch=settings['max_trades_per_epoch'],
            max_fee_per_epoch=settings['max_fee_per_epoch'],
            atr_multiplier=settings['atr_multiplier'],
            atr_period=settings['atr_period'],
            atr_smoothing=settings['atr_smoothing'],
            dqn_weight_scale=settings['dqn_weight_scale'],
            capital_type=settings['capital_type'],
            reference_capital=settings['reference_capital'],
            capital_percentage=settings['capital_percentage'],
            epochs=settings['epochs'],
            confirmation_steps=settings['confirmation_steps'],
        )

    agent = st.session_state.agent

    if (
        agent is None or not agent.has_enough_data() or
        agent.ticker != settings['ticker'] or
        agent.start_date != settings['start_date'] or
        agent.end_date != settings['end_date']
        ):
        agent = init_new_ensemble_agent(settings)

    agent.data = _assert_agent_data(agent, settings)
    
    return agent

def _assert_agent_data(agent, settings):
    if not agent.has_enough_data():
        st.error("Agent does not enough data for the selected lookback window and batch size. Please select a wider date range or reduce the lookback/batch size.")
        return None
    else:
        full_data = agent.data.copy()
    
    full_data = full_data[(full_data.index.date >= settings['start_date']) & (full_data.index.date <= settings['end_date'])]

    if full_data.empty:
        st.error("No data available for the selected date range.")
        return None

    if isinstance(full_data.columns, pd.MultiIndex):
        full_data.columns = [col[0] for col in full_data.columns]

    batch_size = agent.dqn_agent.batch_size if agent.dqn_agent is not None else 64
    if len(full_data) < settings['lookback'] + batch_size + 1:
        st.error("Not enough data for the selected lookback window and batch size. Please select a wider date range or reduce the lookback/batch size.")
        return None

    if len(full_data) > 0:
        current_price = full_data['Close'].iloc[-1]
        min_trade_cost = current_price * (1 + settings['trade_fee'])
        if settings['initial_cash'] < min_trade_cost:
            st.error(f"Initial capital (€{settings['initial_cash']:.2f}) is too low to buy one share at €{current_price:.2f} with fee {settings['trade_fee']*100:.2f}%. Please increase capital.")
            return None

    return full_data



def _main():
    st.session_state.user_settings = _render_sidebar(st.session_state.user_settings)
    
    st.title(f"Ensemble AI Auto Trader - {st.session_state.user_settings['ticker']}")

    st.session_state.agent = _get_EnsembleAgent(st.session_state.user_settings)
    
    _render_main_ui(st.session_state.agent, st.session_state.user_settings)



if __name__ == "__main__":
    _initialize_session_state()
    
    _main()
