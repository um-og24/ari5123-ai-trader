import os
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import sys
sys.dont_write_bytecode = True

import streamlit as st
from calculations import Calculations
from utils import Utils, SETTING_KEYS


def render_overview(agent, settings):
    cols=st.columns([1, 3])
    cols[0].header("Overview")
    cols[1].write("")

    if agent is None:
        cols[1].error("Please configure Settings before proceeding...")
        return

    with cols[1]:
        keys_to_compare = SETTING_KEYS.copy()
        Utils.check_and_restore_settings(agent, settings, comparison_keys=keys_to_compare, context="Overview")

    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("Portfolio Summary")
        portfolio = st.session_state.portfolio_tracker
        total_value = Calculations.to_scalar(portfolio.cash) + Calculations.to_scalar(portfolio.current_position_value)
        total_value_diff=total_value-Calculations.to_scalar(agent.initial_cash)
        # pnl = Calculations.to_scalar(portfolio.realized_pnl) + Calculations.to_scalar(portfolio.unrealized_pnl)
        # pnl_pct = (pnl / Calculations.to_scalar(portfolio.initial_cash)) * 100 if portfolio.initial_cash > 0 else 0
        # st.metric("Total Portfolio Value", f"€{float(total_value):.2f}", delta=f"{'-€' if float(pnl) < 0.0 else '€'}{abs(float(pnl)):.2f} ({float(pnl_pct):.2f}%)")
        st.metric("Total Portfolio Value", f"€{float(total_value):.2f}", delta=f"{'-€' if total_value_diff < 0.0 else '€'}{abs(float(total_value_diff)):.2f}")
        col1_1, col1_2 = st.columns(2)
        col1_1.metric("Cash", f"€{Calculations.to_scalar(portfolio.cash):.2f}")
        col1_2.metric("Positions Value", f"€{Calculations.to_scalar(portfolio.current_position_value):.2f}")
        col1_3, col1_4 = st.columns(2)
        realized_pnl=f"{'-€' if Calculations.to_scalar(portfolio.realized_pnl) < 0.0 else '€'}{abs(Calculations.to_scalar(portfolio.realized_pnl)):.2f}"
        col1_3.metric("Realized P&L", realized_pnl, realized_pnl)
        unrealized_pnl=f"{'-€' if Calculations.to_scalar(portfolio.unrealized_pnl) < 0.0 else '€'}{abs(Calculations.to_scalar(portfolio.unrealized_pnl)):.2f}"
        col1_4.metric("Unrealized P&L", unrealized_pnl, unrealized_pnl)
        col1_5, col1_6 = st.columns(2)
        col1_5.metric("Total Fees Paid", f"€{Calculations.to_scalar(portfolio.total_fees):.2f}")
        fee_pct = (Calculations.to_scalar(portfolio.total_fees) / Calculations.to_scalar(portfolio.initial_cash)) * 100 if portfolio.initial_cash > 0 else 0
        col1_6.metric("Fees (% of Initial Capital)", f"{fee_pct:.2f}%")

    with col2:
        st.subheader("Model Training Summary")
        if agent:
            st.write(f"Training Date range: {agent.training_start_date} to {agent.training_end_date}")
            st.write(f"Training Data points: {len(agent.training_data) if agent and agent.training_data is not None and not agent.training_data.empty else '-'}")
            st.write(f"Training Lookback: {agent.lookback} days")
            
            batch_size = agent.dqn_agent.batch_size if agent and agent.dqn_agent is not None else 64
            st.write(f"Batch Size: {batch_size}")
            st.write(f"Trade Fee: {agent.trade_fee*100:.2f}%")
            st.write(f"Risk Per Trade: {agent.risk_per_trade*100:.2f}%")
            st.write(f"ATR Multiplier: {agent.atr_multiplier}")
            st.write(f"ATR Period: {agent.atr_period} days")
            st.write(f"ATR Smoothing: {'Enabled' if agent.atr_smoothing else 'Disabled'}")
            st.write(f"SMOTE: {'Enabled' if agent.use_smote else 'Disabled'}")
            st.write(f"Signal Confirmation Steps: {agent.confirmation_steps}")
            dqn_weight_scale=agent.dqn_weight_scale
            if dqn_weight_scale > 0.5:
                preferred_agent="DQN Agent"
            elif dqn_weight_scale == 0.5:
                preferred_agent="No preference"
            else:
                preferred_agent="RF Agent"
            st.write(f"Fallback Agent: {preferred_agent}")

    st.divider()