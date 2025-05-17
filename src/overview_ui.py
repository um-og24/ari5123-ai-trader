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
    with cols[1]:
        keys_to_compare = SETTING_KEYS.copy()
        Utils.check_and_restore_settings(agent, settings, comparison_keys=keys_to_compare, context="Overview")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Portfolio Summary")
        portfolio = st.session_state.portfolio_tracker
        total_value = Calculations.to_scalar(portfolio.cash) + Calculations.to_scalar(portfolio.current_position_value)
        pnl = Calculations.to_scalar(portfolio.realized_pnl) + Calculations.to_scalar(portfolio.unrealized_pnl)
        pnl_pct = (pnl / Calculations.to_scalar(portfolio.initial_cash)) * 100 if portfolio.initial_cash > 0 else 0
        st.metric(
            "Total Portfolio Value",
            f"€{float(total_value):.2f}",
            delta=f"€{float(pnl):.2f} ({float(pnl_pct):.2f}%)"
        )
        col1_1, col1_2 = st.columns(2)
        col1_1.metric("Cash", f"€{Calculations.to_scalar(portfolio.cash):.2f}")
        col1_2.metric("Positions Value", f"€{Calculations.to_scalar(portfolio.current_position_value):.2f}")
        col1_3, col1_4 = st.columns(2)
        col1_3.metric("Realized P&L", f"€{Calculations.to_scalar(portfolio.realized_pnl):.2f}")
        col1_4.metric("Unrealized P&L", f"€{Calculations.to_scalar(portfolio.unrealized_pnl):.2f}")
        col1_5, col1_6 = st.columns(2)
        col1_5.metric("Total Fees Paid", f"€{Calculations.to_scalar(portfolio.total_fees):.2f}")
        fee_pct = (Calculations.to_scalar(portfolio.total_fees) / Calculations.to_scalar(portfolio.initial_cash)) * 100 if portfolio.initial_cash > 0 else 0
        col1_6.metric("Fees (% of Initial Capital)", f"{fee_pct:.2f}%")
    with col2:
        st.subheader("Model & Data Settings")
        st.write(f"Date range: {agent.start_date} to {agent.end_date}")
        st.write(f"Data points: {len(agent.data) if agent and agent.data is not None and not agent.data.empty else "-"}")
        st.write(f"Lookback: {agent.lookback} days")
        
        batch_size = agent.dqn_agent.batch_size if agent.dqn_agent is not None else 64
        st.write(f"Batch size: {batch_size}")
        st.write(f"Trade fee: {agent.trade_fee*100:.2f}%")
        st.write(f"Risk per trade: {agent.risk_per_trade*100:.2f}%")
        st.write(f"ATR Multiplier: {agent.atr_multiplier}")
        st.write(f"ATR Period: {agent.atr_period} days")
        st.write(f"ATR Smoothing: {'Enabled' if agent.atr_smoothing else 'Disabled'}")
        st.write(f"Signal confirmation steps: {agent.confirmation_steps}")
        dqn_weight_scale=agent.dqn_weight_scale
        if dqn_weight_scale > 50:
            preferred_agent="DQN Agent"
        elif dqn_weight_scale == 50:
            preferred_agent="No preference"
        else:
            preferred_agent="RF Agent"
        st.write(f"Fallback agent: {preferred_agent}")
    st.divider()
    
    st.subheader(f"Agent's Dataset - {agent.ticker} ({agent.start_date} to {agent.end_date})")
    st.dataframe(agent.data)
    
    st.divider()