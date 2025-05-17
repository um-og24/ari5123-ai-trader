import os
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import sys
sys.dont_write_bytecode = True

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils import Utils

def render_trade_log(agent, settings):
    st.header("Trade Log")
    portfolio = st.session_state.portfolio_tracker
    st.subheader("Recent Transactions")
    if portfolio.transactions:
        trans_df = pd.DataFrame(portfolio.transactions)
        trans_df['timestamp'] = pd.to_datetime(trans_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        trans_df['P&L'] = trans_df.apply(
            lambda row: f"€{row['pnl']:.2f}" if 'pnl' in row else "-", axis=1
        )
        trans_df['Fee'] = trans_df['fee'].apply(lambda x: f"€{x:.2f}")  # Format fee as currency
        st.dataframe(trans_df.iloc[::-1], use_container_width=True)
    else:
        st.info("No transactions yet")

    st.divider()

    st.subheader("AI Trading Signals Log")
    if st.session_state.trade_log:
        trade_log_df = pd.DataFrame(st.session_state.trade_log)
        trade_log_df['Time'] = pd.to_datetime(trade_log_df['Time']).dt.strftime('%Y-%m-%d %H:%M:%S')
        trade_log_df['Fee'] = trade_log_df['Fee'].apply(lambda x: f"€{x:.2f}")  # Format fee as currency
        st.dataframe(trade_log_df, use_container_width=True)
    else:
        st.info("No trades yet. Run live trading to see trades.")

    st.divider()