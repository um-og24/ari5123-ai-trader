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
            latest_price = agent.data['Close'].iloc[-1] if len(agent.data) > 0 else details['avg_price']
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
    
    st.subheader("Portfolio Metrics")
    if len(portfolio.history) > 1:
        # Skip the initial portfolio.history and portfolio.bb_history entry to align with simulation start
        portfolio_history = portfolio.history[1:]
        portfolio_bh_history = portfolio.bh_history[1:] if len(portfolio.bh_history) > 1 else portfolio.bh_history
        
        # Agent Portfolio Metrics
        values = [Calculations.to_scalar(h['total_value']) for h in portfolio_history]
        initial = values[0]
        current = values[-1]
        max_value = max(values)
        total_return = (current - initial) / initial * 100
        drawdown = ((max_value - current) / max_value) * 100 if max_value > 0 else 0
        sharpe_ratio = 0.0
        if len(values) > 10:
            daily_returns = [(values[i] - values[i-1])/values[i-1] for i in range(1, len(values))]
            sharpe_ratio = np.sqrt(252) * np.mean(daily_returns) / (np.std(daily_returns) + 1e-9)
        

        # Display Metrics
        st.write("**Agent Portfolio**")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Return", f"{total_return:.2f}%")
        col2.metric("Current Drawdown", f"{drawdown:.2f}%")
        col3.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
       
        st.write("")
        
        # Buy-and-Hold Portfolio Metrics
        bh_sharpe, bh_drawdown, bh_total_return = portfolio.evaluate_bh_performance(False)
        
        st.write("**Buy-and-Hold Portfolio**")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Return", f"{bh_total_return*100:.2f}%")
        col2.metric("Max Drawdown", f"€{bh_drawdown:.2f}")
        col3.metric("Sharpe Ratio", f"{bh_sharpe:.2f}")

        
        # Buy-and-Hold Portfolio Metrics
        bh_sharpe, bh_drawdown, bh_total_return = portfolio.evaluate_bh_performance(True)
        
        st.write("**Buy-and-Hold Portfolio (ALL)**")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Return", f"{bh_total_return*100:.2f}%")
        col2.metric("Max Drawdown", f"€{bh_drawdown:.2f}")
        col3.metric("Sharpe Ratio", f"{bh_sharpe:.2f}")

        st.divider()

        # Portfolio Performance Chart
        st.subheader("Portfolio Value Over Time")
        performance_fig = ChartBuilder.plot_portfolio_performance(portfolio_history)
        st.plotly_chart(performance_fig, use_container_width=True)

        st.divider()

        # Portfolio Comparison Chart
        st.subheader("Agent vs. Buy-and-Hold Comparison")
        ChartBuilder.plot_portfolio_comparison(portfolio_history, portfolio_bh_history, settings['ticker'])

        cols=st.columns(2)
        cols[0].write("**Agent History**")
        cols[0].dataframe(pd.DataFrame(portfolio_history))
        cols[1].write("**Buy-and-Hold History**")
        cols[1].dataframe(pd.DataFrame(portfolio_bh_history))
    else:
        st.info("No portfolio history available")

    st.divider()