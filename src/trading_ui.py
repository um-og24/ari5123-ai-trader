import os
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import sys
sys.dont_write_bytecode = True

import streamlit as st
import numpy as np
import pandas as pd
from calculations import Calculations
from chart_builder import ChartBuilder
from utils import Utils, SETTING_KEYS

def _execute_trade(agent, settings, portfolio, action, current_price, timestamp, q_values, dqn_action, rf_action):
    current_price = float(current_price.iloc[0]) if hasattr(current_price, 'iloc') else float(current_price)
    transaction_icons=["✅", "🚫", "❌", "🕔"]
    transaction_message = ""
    transaction_icon=""
    fee_amount = 0.0  # Initialize fee amount
    min_trade_cost = current_price * (1 + settings['trade_fee'])
    
    # Debug action types
    Utils.log_message(f"DEBUG: Ensemble Action: {action}, type: {type(action)} | DQN Action: {dqn_action}, type: {type(dqn_action)} | RF Action: {rf_action}, type: {type(rf_action)}")
    
    if action == 1 and portfolio.cash < min_trade_cost:
        return "BUY SKIPPED: Insufficient capital to execute trades", transaction_icons[1]
    
    confirmation_steps = settings['confirmation_steps']
    recent_actions = st.session_state.recent_actions
    if len(recent_actions) >= confirmation_steps and all(a == action for a in recent_actions[-confirmation_steps:]) and action in [1, 2]:
        if action == 1:
            # Use ATR for dynamic volatility-based position sizing
            atr = 0.02 * current_price  # Default ATR
            if 'ATR' in agent.trading_data.columns and st.session_state.current_index < len(agent.trading_data):
                atr_value = agent.trading_data['ATR'].iloc[st.session_state.current_index]
                atr = float(atr_value) if pd.notna(atr_value) else atr
            
            atr_multiplier = 1.5  # Adjustable multiplier
            stop_loss_distance = max(atr * atr_multiplier, 1.0)
            max_risk_amount = portfolio.cash * settings['risk_per_trade']
            shares_to_buy = max(int(max_risk_amount / stop_loss_distance), 1)
            max_shares_by_cash = int(portfolio.cash / (current_price * (1 + settings['trade_fee'])))
            shares_to_buy = min(shares_to_buy, max_shares_by_cash)
            if shares_to_buy > 0:
                success, result = portfolio.buy(settings['ticker'], shares_to_buy, current_price, fee_percentage=settings['trade_fee'], timestamp=timestamp)
                if success:
                    # Retrieve fee from the latest transaction
                    if portfolio.transactions:
                        fee_amount = portfolio.transactions[-1]['fee']
                    transaction_message = f"BUY FILLED: {shares_to_buy} shares of {settings['ticker']} at €{current_price:.2f} with €{fee_amount:.2f} fee (ID: {result})"
                    transaction_icon=transaction_icons[0]
                    st.session_state.last_trade_id = result
                else:
                    transaction_message = f"BUY FAILED: {result}"
                    transaction_icon=transaction_icons[2]
            else:
                transaction_message = "BUY SKIPPED: Insufficient funds or position size too small"
                transaction_icon=transaction_icons[1]
        elif action == 2:
            if settings['ticker'] in portfolio.holdings and portfolio.holdings[settings['ticker']]['quantity'] > 0:
                shares_to_sell = portfolio.holdings[settings['ticker']]['quantity']
                success, result = portfolio.sell(settings['ticker'], shares_to_sell, current_price, fee_percentage=settings['trade_fee'], timestamp=timestamp)
                if success:
                    # Retrieve fee from the latest transaction
                    if portfolio.transactions:
                        fee_amount = portfolio.transactions[-1]['fee']
                    transaction_message = f"SELL FILLED: {shares_to_sell} shares of {settings['ticker']} at €{current_price:.2f} with €{fee_amount:.2f} fee (ID: {result})"
                    transaction_icon=transaction_icons[0]
                    st.session_state.last_trade_id = result
                else:
                    transaction_message = f"SELL FAILED: {result}"
                    transaction_icon=transaction_icons[2]
            else:
                transaction_message = "SELL SKIPPED: No shares to sell"
                transaction_icon=transaction_icons[1]
        if transaction_message and "SKIPPED" not in transaction_message and "FAILED" not in transaction_message:
            st.session_state.recent_actions = []
    elif action in [1, 2]:
        transaction_message = f"ACTION SKIPPED: Waiting for {confirmation_steps} consecutive {'Buy' if action == 1 else 'Sell'} signals (Current: {len(recent_actions) + 1})"
        transaction_icon=transaction_icons[3]

    if dqn_action != rf_action:
        Utils.log_message(f"INFO: Ensemble chose action {action} from DQN={dqn_action} and RF={rf_action}")

    if transaction_message:
        st.session_state.trade_log.append({
            "Time": timestamp,
            "Action": "Buy" if action == 1 else "Sell" if action == 2 else "Hold",
            "DQN Action": int(dqn_action),  # Ensure int
            "RF Action": int(rf_action),    # Ensure int
            "Price": round(current_price, 2),
            "Quantity": locals().get('shares_to_buy', 0) if action == 1 else locals().get('shares_to_sell', 0) if action == 2 else 0,
            "Value": round(locals().get('shares_to_buy', 0) * current_price, 2) if action == 1 else round(locals().get('shares_to_sell', 0) * current_price, 2) if action == 2 else 0,
            "Transaction": transaction_message,
            "Portfolio Value": round(portfolio.cash + portfolio.current_position_value, 2),
            "Fee": round(fee_amount, 2),  # Add fee to trade log
            "Q-Hold": round(q_values[0], 2),
            "Q-Buy": round(q_values[1], 2),
            "Q-Sell": round(q_values[2], 2)
        })
    Utils.log_message(f"INFO: Action={action}, DQN={dqn_action}, RF={rf_action}, Recent Actions={st.session_state.recent_actions}, Cash={portfolio.cash}, Min Trade Cost={min_trade_cost}, Fee={fee_amount}")
    return transaction_message, transaction_icon


def render_live_trading(agent, settings):
    cols = st.columns([2, 3])
    st.divider()
    date_cols=st.columns(2)

    with cols[0]:
        st.header("Live Trading")
    with cols[1]:
        if agent.has_pretrained_model():
            st.success(f"Loaded pre-trained ensemble model for {agent.ticker} with best Sharpe Ratio of {agent.best_sharpe:.4f}.")
            # Ensure RF scaler is fitted
            if not hasattr(agent.rf_agent.scaler, 'mean_'):
                st.error("RF Agent scaler is not fitted. Please train the model before live trading.")
                return
        else:
            st.warning("No pre-trained ensemble model found. Please train the model before live trading.")
            return  # Prevent simulation until model is trained

    st.session_state.trading_start_date = date_cols[0].date_input("Start Date", value=st.session_state.trading_start_date, key="trading_start_date_pick")
    st.session_state.trading_end_date = date_cols[1].date_input("End Date", value=st.session_state.trading_end_date, key="trading_end_date_pick")

    agent.trading_data = agent.get_data(st.session_state.trading_start_date, st.session_state.trading_end_date)

    if agent and agent.has_pretrained_model():
        with st.expander(f"Agent's Trading Dataset - {agent.ticker} ({st.session_state.trading_start_date} to {st.session_state.trading_end_date})"):
            st.dataframe(agent.trading_data)

    with cols[1]:
        checkpoint_info = agent.load_checkpoint(settings)
        trading_settings = None
        if checkpoint_info:
            _, _, _, _, trading_settings = checkpoint_info
        else:
            trading_settings = settings

        # Check for settings mismatches
        keys_to_compare = SETTING_KEYS.copy()
        Utils.check_and_restore_settings(agent, trading_settings, comparison_keys=keys_to_compare, context="Live Trading")

    st.divider()

    col1, col2, col3 = st.columns(3)
    start_button = col1.button("Start Trading", disabled=st.session_state.simulation_running, use_container_width=True)
    pause_button = col2.button("Resume Trading" if st.session_state.simulation_paused else "Pause Trading", disabled=not st.session_state.simulation_running, use_container_width=True)
    stop_button = col3.button("Stop Trading", disabled=not st.session_state.simulation_running, use_container_width=True)

    placeholder_info = st.empty()

    st.divider()

    st.subheader("Current Portfolio")
    portfolio = st.session_state.portfolio_tracker
    total_value = Calculations.to_scalar(portfolio.cash) + Calculations.to_scalar(portfolio.current_position_value)
    total_value_diff=total_value-Calculations.to_scalar(agent.initial_cash)
    metric_cols = st.columns(4)
    metric_cols[0].metric("Portfolio Value", f"€{float(total_value):.2f}", delta=f"{'-€' if total_value_diff < 0.0 else '€'}{abs(float(total_value_diff)):.2f}")
    metric_cols[1].metric("Cash", f"€{Calculations.to_scalar(portfolio.cash):.2f}")
    metric_cols[2].metric("Positions", f"€{Calculations.to_scalar(portfolio.current_position_value):.2f}")
    if agent and agent.trading_data is not None and not agent.trading_data.empty and len(agent.trading_data) > 0 and st.session_state.current_index > 0:
        i = min(st.session_state.current_index, len(agent.trading_data) - 1)
        current_price = float(agent.trading_data['Close'].iloc[i])
        prev_price = float(agent.trading_data['Close'].iloc[i-1]) if i > 0 else current_price
        price_change = ((current_price - prev_price) / prev_price * 100) if prev_price != 0 else 0
        metric_cols[3].metric(f"{settings['ticker']} Price", f"€{current_price:.2f}", delta=f"{price_change:.2f}%", delta_color="inverse")

    st.divider()

    trading_signals_header_cols=st.columns([1,2,2])
    trading_signals_header_cols[0].subheader("Trading Signals")    
    placeholder_timer = trading_signals_header_cols[1].empty()

    action_cols = st.columns([1, 2, 2])
    placeholder_timestamp = action_cols[0].empty()
    placeholder_action = action_cols[1].empty()
    placeholder_transaction = action_cols[2].empty()
    placeholder_chart = st.empty()

    st.divider()

    if start_button:
        st.session_state.simulation_running = True
        st.session_state.simulation_paused = False
        st.session_state.current_index = settings['lookback']
        st.session_state.trade_log = []
        st.session_state.actions = np.zeros(len(agent.trading_data), dtype=int)
        st.session_state.recent_actions = []
        portfolio.reset()
        # Initialize B&H portfolio
        if len(agent.trading_data) > settings['lookback']:
            initial_price = float(agent.trading_data['Close'].iloc[settings['lookback']])
            timestamp = agent.trading_data.index[settings['lookback']]
            portfolio.init_buy_and_hold(settings['ticker'], initial_price, fee_percentage=settings['trade_fee'], timestamp=timestamp)
            Utils.log_message(f"INFO: Initialized B&H portfolio for {settings['ticker']} at price €{initial_price:.2f}")
        else:
            Utils.log_message(f"ERROR: Insufficient data to initialize B&H portfolio")
            st.error("Not enough data to start trading. Please select a wider date range.")
            st.session_state.simulation_running = False
            return
        st.rerun()
    if pause_button:
        st.session_state.simulation_paused = not st.session_state.simulation_paused
        st.rerun()
    if stop_button:
        st.session_state.simulation_running = False
        st.session_state.simulation_paused = False
        st.session_state.recent_actions = []
        st.session_state.current_index = 0
        st.session_state.trade_log = []
        st.session_state.actions = None
        portfolio.reset()
        Utils.log_message(f"INFO: Live trading stopped and agent cleared")
        st.rerun()

    if agent and not agent.trading_data.empty:
        with placeholder_timestamp.container():
            st.write("**Current Date**")
            st.warning(agent.trading_data.index[st.session_state.current_index if st.session_state.current_index < len(agent.trading_data) else len(agent.trading_data) - 1].strftime("%A, %d %b %Y"))  # .strftime("%A, %d %b %Y %X") Includes timestamp

    if st.session_state.simulation_running and not st.session_state.simulation_paused:
        with placeholder_timer.container():
            prediciton_wait_text="Making a prediction, please wait..."
            st.info(f"⌛ {prediciton_wait_text}")
            st.toast(prediciton_wait_text, icon="⌛")
        norm_data = agent.dqn_agent.preprocess_data(agent.trading_data)
        if st.session_state.current_index < len(agent.trading_data) - 1 and st.session_state.current_index - settings['lookback'] < len(norm_data):
            i = st.session_state.current_index
            timestamp = agent.trading_data.index[i]
            with placeholder_timestamp.container():
                st.write("**Current Date**")
                st.info(timestamp.strftime("%A, %d %b %Y"))  # .strftime("%A, %d %b %Y %X") Includes timestamp
            state_index = i - settings['lookback']
            if state_index < 0 or state_index >= len(norm_data):
                Utils.log_message(f"ERROR: Invalid state_index: {state_index}, norm_data shape: {norm_data.shape}")
                st.error("Invalid state index. Please check data length and lookback settings.")
                return
            state = norm_data[state_index]
            action, q_values = agent.predict_live_action(state)
            dqn_action, _ = agent.dqn_agent.predict(state)
            rf_action, _ = agent.rf_agent.predict(state)
            Utils.log_message(f"DEBUG: Ensembler Action: {action}, type: {type(action)} | DQN Action: {dqn_action}, type: {type(dqn_action)} | RF Action: {rf_action}, type: {type(rf_action)}")
            if not st.session_state.recent_actions or st.session_state.recent_actions[-1] != action:
                st.session_state.recent_actions = [int(action)]
            else:
                st.session_state.recent_actions.append(int(action))
            st.session_state.actions[i] = int(action)
            sliced_data = agent.trading_data.iloc[:i + 1]
            ChartBuilder.plot_transaction_history(sliced_data, portfolio, "trading_1", placeholder_chart.container())
            # ChartBuilder.plot_trading_signals(sliced_data, st.session_state.actions[:i + 1], placeholder_chart.container())
            Utils.display_action(action, q_values, placeholder_action)
            current_price = agent.trading_data['Close'].iloc[i]
            portfolio = st.session_state.portfolio_tracker
            portfolio.update_bh_position_value(settings['ticker'], current_price, timestamp)
            transaction_message, transaction_icon = _execute_trade(agent, settings, portfolio, action, current_price, timestamp, q_values, dqn_action, rf_action)
            with placeholder_transaction.container():
                if transaction_message:
                    st.write("**Transaction Outcome**")
                    if "FAILED" in transaction_message:
                        st.error(transaction_message)
                    elif "SKIPPED" in transaction_message:
                        st.warning(transaction_message)
                    else:
                        st.success(transaction_message)
                    st.toast(transaction_message, icon=transaction_icon)
            st.session_state.current_index += 1
            Utils.display_timer_until_rerun(settings['trading_simulation_delay'], "Next trade signal in ", placeholder_timer)
        else:
            i=st.session_state.current_index
            ticker_prices = {settings['ticker']: agent.trading_data['Close'].iloc[i]}
            portfolio.update_position_values(ticker_prices, agent.trading_data.index[i-1])
            st.session_state.simulation_running = False
            st.session_state.recent_actions = []
            ChartBuilder.plot_transaction_history(agent.trading_data.iloc[:i + 1], portfolio, "trading_2", placeholder_chart.container())
            # ChartBuilder.plot_trading_signals(agent.trading_data.iloc[:i + 1], st.session_state.actions[:i + 1], placeholder_chart.container())
            placeholder_info.success("Trading simulation completed!")
            placeholder_timer.write("")
    elif st.session_state.simulation_running and st.session_state.simulation_paused:
        if st.session_state.actions is not None:
            i = st.session_state.current_index - 1
            ChartBuilder.plot_transaction_history(agent.trading_data.iloc[:i + 1], portfolio, "trading_3", placeholder_chart.container())
            # ChartBuilder.plot_trading_signals(agent.trading_data.iloc[:i + 1], st.session_state.actions[:i + 1], placeholder_chart.container())
            placeholder_info.info("Trading paused. Click 'Resume Trading' to continue.")
            placeholder_timer.write("")
    elif not st.session_state.simulation_running and st.session_state.actions is not None:
        placeholder_timer.write("")
        i = st.session_state.current_index - 1 if st.session_state.current_index > 0 else 0
        if i >= settings['lookback']:
            ChartBuilder.plot_transaction_history(agent.trading_data.iloc[:i + 1], portfolio, "trading_4", placeholder_chart.container())
            # ChartBuilder.plot_trading_signals(agent.trading_data.iloc[:i + 1], st.session_state.actions[:i + 1], placeholder_chart.container())
        else:
            with placeholder_chart.container():
                placeholder_info.info("Start trading to see signals")
    else:
        placeholder_timer.write("")
        with placeholder_chart.container():
            st.info("Start trading to see signals")