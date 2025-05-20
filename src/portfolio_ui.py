import os
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import sys
sys.dont_write_bytecode = True

import streamlit as st
import numpy as np
import pandas as pd
from calculations import Calculations
from chart_builder import ChartBuilder
from utils import Utils

def render_portfolio(agent, settings):
    st.header("Portfolio Performance & Summary")
    portfolio = st.session_state.portfolio_tracker

    st.subheader("Current Holdings")
    if portfolio.holdings:
        holdings_data = []
        for ticker, details in portfolio.holdings.items():
            latest_price = agent.training_data['Close'].iloc[-1] if len(agent.training_data) > 0 else details['avg_price']
            position_value = details['quantity'] * latest_price
            cost_basis = details['quantity'] * details['avg_price']
            unrealized_pnl = position_value - cost_basis
            unrealized_pnl_pct = (unrealized_pnl / cost_basis) * 100 if cost_basis > 0 else 0
            holdings_data.append({
                "Ticker": ticker,
                "Quantity": details['quantity'],
                "Avg Price": f"€{Calculations.to_scalar(details['avg_price']):.2f}",
                "Current Price": f"€{Calculations.to_scalar(latest_price):.2f}",
                "Position Value": f"€{Calculations.to_scalar(position_value):.2f}",
                "Unrealized P&L": f"€{Calculations.to_scalar(unrealized_pnl):.2f}",
                "Unrealized P&L %": f"{Calculations.to_scalar(unrealized_pnl_pct):.2f}%"
            })
        st.dataframe(pd.DataFrame(holdings_data), use_container_width=True)
    else:
        st.info("No current holdings")
        
    st.divider()
    if len(portfolio.history) > 1:
        # Skip the initial portfolio.history and portfolio.bb_history entry to align with simulation start
        portfolio_history = portfolio.history[1:]
        portfolio_bh_history = portfolio.bh_history[1:] if len(portfolio.bh_history) > 1 else portfolio.bh_history

        # Portfolio Comparison Chart
        st.subheader("Agent vs. Buy-and-Hold Comparison")

        ChartBuilder.plot_portfolio_comparison(portfolio_history, portfolio_bh_history, settings['ticker'])

        cols=st.columns(2)
        cols[0].write("**Agent History**")
        cols[0].dataframe(pd.DataFrame(portfolio_history))
        cols[1].write("**Buy-and-Hold Daily History**")
        cols[1].dataframe(pd.DataFrame(portfolio_bh_history))

        st.divider()

        # Portfolio Performance Chart
        st.subheader("Agent Portfolio")
        # Agent Portfolio Metrics
        metrics = Calculations.compute_metrics(portfolio_history, value_key='total_value')

        # Display Metrics    
        st.subheader("Metrics")
        cols= st.columns(6)
        cols[0].metric("Total Return", f"{metrics['total_return']:.2f}%")
        cols[1].metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%")
        cols[2].metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        cols[3].metric("Sortino Ratio", f"{metrics['sortino_ratio']:.2f}")
        cols[4].metric("VaR", f"€{metrics['var']:.2f}")
        cols[5].metric("CVaR", f"€{metrics['cvar']:.2f}")

        performance_fig = ChartBuilder.plot_portfolio_performance(portfolio_history, metrics)
        st.plotly_chart(performance_fig, key="pfp_agent", use_container_width=True)

        st.divider()

        # Buy & Hold Portfolio
        st.subheader("Buy & Hold Portfolio")
        # Buy-and-Hold Portfolio Metrics
        bh_metrics = Calculations.compute_metrics([portfolio_bh_history[0], portfolio_bh_history[-1]], value_key='total_value')

        # Display Metrics    
        st.subheader("Metrics")
        cols= st.columns(6)
        cols[0].metric("Total Return", f"{bh_metrics['total_return']:.2f}%")
        cols[1].metric("Max Drawdown", f"{bh_metrics['max_drawdown']:.2f}%")
        cols[2].metric("Sharpe Ratio", f"{bh_metrics['sharpe_ratio']:.2f}")
        cols[3].metric("Sortino Ratio", f"{bh_metrics['sortino_ratio']:.2f}")
        cols[4].metric("VaR", f"€{bh_metrics['var']:.2f}")
        cols[5].metric("CVaR", f"€{bh_metrics['cvar']:.2f}")

        performance_fig = ChartBuilder.plot_portfolio_performance([portfolio_bh_history[0], portfolio_bh_history[-1]], bh_metrics)
        st.plotly_chart(performance_fig, key="pfp_bh", use_container_width=True)

        with st.expander("**Buy & Hold Daily Portfolio**"):
            # All Buy-and-Hold Portfolio Metrics
            daily_bh_metrics = Calculations.compute_metrics(portfolio_bh_history, value_key='total_value')

            # Display Metrics    
            st.subheader("Metrics")
            cols= st.columns(6)
            cols[0].metric("Total Return", f"{daily_bh_metrics['total_return']:.2f}%")
            cols[1].metric("Max Drawdown", f"{daily_bh_metrics['max_drawdown']:.2f}%")
            cols[2].metric("Sharpe Ratio", f"{daily_bh_metrics['sharpe_ratio']:.2f}")
            cols[3].metric("Sortino Ratio", f"{daily_bh_metrics['sortino_ratio']:.2f}")
            cols[4].metric("VaR", f"€{daily_bh_metrics['var']:.2f}")
            cols[5].metric("CVaR", f"€{daily_bh_metrics['cvar']:.2f}")

            performance_fig = ChartBuilder.plot_portfolio_performance(portfolio_bh_history, daily_bh_metrics)
            st.plotly_chart(performance_fig, key="pfp_dbh", use_container_width=True)
    else:
        st.info("No portfolio history available")

    st.divider()