import os
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import sys
sys.dont_write_bytecode = True

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.figure_factory import create_distplot
from scipy import stats
from calculations import Calculations
from utils import Utils, FEATURE_COLUMNS

# Set seeds for reproducibility
np.random.seed(42)

class ChartBuilder:
    """Centralized class for creating Plotly charts for portfolio and training visualization."""
    
    @staticmethod
    @st.cache_data(ttl=1800)
    def plot_portfolio_performance(history, metrics, title='Performance Over Time', show_cash_positions=True, show_metrics=True):
        """
        Plot portfolio total value, cash, position value, and drawdown over time with trade P&L annotations
        and a metrics table (Sharpe, Sortino, Calmar, VaR, CVaR).

        Args:
            history (list[dict]): Portfolio history with timestamp, total_value, cash, position_value,
                                and optionally trades (list of {'timestamp': str, 'pnl': float}).
            title (str): Chart title.
            show_cash_positions (bool): If True, include cash and position value traces.
            show_metrics (bool): If True, display metrics table (Sharpe, Sortino, Calmar, VaR, CVaR).

        Returns:
            go.Figure: Plotly figure object.
        """
        try:
            timestamps = [pd.to_datetime(h['timestamp']) for h in history if h['total_value'] is not None]
            total_values = [Calculations.to_scalar(h['total_value']) for h in history if h['total_value'] is not None]
            
            if not timestamps:
                Utils.log_message(f"WARNING: No valid data for portfolio performance chart")
                return go.Figure()
            
            # Compute drawdowns over time
            total_values_np = np.array(total_values)
            peak_values = np.maximum.accumulate(total_values_np)
            drawdowns = (peak_values - total_values_np) / peak_values * 100
            
            # Create figure with secondary y-axis for drawdown
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Total portfolio value
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=total_values,
                mode='lines',
                name='Total Portfolio Value',
                line=dict(color='blue')
            ), secondary_y=False)
            
            # Cash and position value traces
            if show_cash_positions:
                cash = [Calculations.to_scalar(h['cash']) for h in history if h['total_value'] is not None]
                position_values = [Calculations.to_scalar(h['position_value']) for h in history if h['total_value'] is not None]
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=cash,
                    mode='lines',
                    name='Cash',
                    line=dict(color='green', dash='dash')
                ), secondary_y=False)
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=position_values,
                    mode='lines',
                    name='Position Value',
                    line=dict(color='orange', dash='dot')
                ), secondary_y=False)
            
            # Drawdown trace
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=drawdowns,
                mode='lines',
                name='Drawdown (%)',
                line=dict(color='red', width=1)
            ), secondary_y=True)
            
            # VaR line
            if metrics['var'] > 0:
                fig.add_hline(
                    y=total_values[-1] - metrics['var'],
                    line_dash="dash",
                    line_color="purple",
                    annotation_text=f"95% VaR: €{metrics['var']:.2f}",
                    annotation_position="top left",
                    secondary_y=False
                )
            
            # Trade P&L annotations
            for h in history:
                if 'trades' in h and h['trades']:
                    for trade in h['trades']:
                        if 'timestamp' in trade and 'pnl' in trade and trade['pnl'] != 0:
                            trade_time = pd.to_datetime(trade['timestamp'])
                            if trade_time in timestamps:
                                idx = timestamps.index(trade_time)
                                fig.add_annotation(
                                    x=trade_time,
                                    y=total_values[idx],
                                    text=f"P&L: €{trade['pnl']:.2f}",
                                    showarrow=True,
                                    arrowhead=2,
                                    ax=20,
                                    ay=-30,
                                    bgcolor="white",
                                    bordercolor="black",
                                    secondary_y=False
                                )
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title='Date',
                yaxis_title='Value (€)',
                yaxis2_title='Drawdown (%)',
                showlegend=True,
                template='plotly_white',
                height=600,
                xaxis=dict(tickformat='%Y-%m-%d')
            )
            
            # Add metrics table as annotation
            if show_metrics:
                metrics_table = (
                    f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}<br>"
                    f"Sortino Ratio: {metrics['sortino_ratio']:.2f}<br>"
                    f"Calmar Ratio: {metrics['calmar_ratio']:.2f}<br>"
                    f"Max Drawdown: {metrics['max_drawdown']:.2f}%<br>"
                    f"Current Drawdown: {metrics['current_drawdown']:.2f}%<br>"
                    f"Total Return: {metrics['total_return']:.2f}%<br>"
                    f"VaR (95%): €{metrics['var']:.2f}<br>"
                    f"CVaR (95%): €{metrics['cvar']:.2f}"
                )
                fig.add_annotation(
                    xref="paper", yref="paper",
                    x=0.01, y=0.53,
                    text=metrics_table,
                    showarrow=False,
                    font=dict(size=12),
                    align="left",
                    bgcolor="black",
                    bordercolor="black",
                    borderpad=4
                )
            
            return fig
        except Exception as e:
            Utils.log_message(f"ERROR: Error plotting portfolio performance: {e}")
            return go.Figure()

    @staticmethod
    def plot_transaction_history(data, portfolio, context, chart_placeholder):
        """
        Plot stock price with executed transactions, Volume, RSI, and MACD.

        Args:
            data (pd.DataFrame): DataFrame with stock price data (Open, High, Low, Close, Volume, SMA20,  SMA80, BB_Upper, BB_Lower, RSI, MACD).
            portfolio (PortfolioTracker): PortfolioTracker instance containing transactions list.
            chart_placeholder: Streamlit placeholder for rendering the chart.

        Returns:
            None: Renders the Plotly chart in the provided placeholder.
        """
        try:
            fig = make_subplots(
                rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                subplot_titles=['Stock Price with Transactions', 'Volume', 'RSI', 'MACD'],
                row_heights=[0.5, 0.2, 0.15, 0.15]
            )

            # Candlestick chart for stock price
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='Price',
                    increasing_line_color='green',
                    decreasing_line_color='red'
                ),
                row=1, col=1
            )

            # SMA20 and Bollinger Bands
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['SMA20'],
                    mode='lines',
                    name='SMA20',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )

            # SMA70 and Bollinger Bands
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['SMA70'],
                    mode='lines',
                    name='SMA70',
                    line=dict(color='darkgreen', width=2)
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['BB_Upper'],
                    mode='lines',
                    name='BB Upper',
                    line=dict(color='purple', width=1, dash='dash')
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['BB_Lower'],
                    mode='lines',
                    name='BB Lower',
                    line=dict(color='purple', width=1, dash='dash')
                ),
                row=1, col=1
            )

            # Process transactions
            if portfolio.transactions:
                trans_df = pd.DataFrame(portfolio.transactions)
                trans_df['timestamp'] = pd.to_datetime(trans_df['timestamp'])

                # Filter transactions within the data's date range
                trans_df = trans_df[
                    (trans_df['timestamp'] >= data.index.min()) &
                    (trans_df['timestamp'] <= data.index.max())
                ]

                # Buy transactions
                buy_trans = trans_df[trans_df['action'] == 'Buy']
                if not buy_trans.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=buy_trans['timestamp'],
                            y=buy_trans['price'],
                            mode='markers',
                            name='Buy Transaction',
                            marker=dict(symbol='triangle-up', size=10, color='green'),
                            text=[f"ID: {id}<br>Qty: {qty}<br>Price: €{price:.2f}<br>Fee: €{fee:.2f}"
                                for id, qty, price, fee in zip(buy_trans['id'], buy_trans['quantity'], buy_trans['price'], buy_trans['fee'])],
                            hoverinfo='text'
                        ),
                        row=1, col=1
                    )

                # Sell transactions
                sell_trans = trans_df[trans_df['action'] == 'Sell']
                if not sell_trans.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=sell_trans['timestamp'],
                            y=sell_trans['price'],
                            mode='markers',
                            name='Sell Transaction',
                            marker=dict(symbol='triangle-down', size=10, color='red'),
                            text=[f"ID: {id}<br>Qty: {qty}<br>Price: €{price:.2f}<br>Fee: €{fee:.2f}<br>P&L: €{pnl:.2f}"
                                for id, qty, price, fee, pnl in zip(sell_trans['id'], sell_trans['quantity'], sell_trans['price'], sell_trans['fee'], sell_trans['pnl'])],
                            hoverinfo='text'
                        ),
                        row=1, col=1
                    )

                # Connect transactions with a dotted line
                trade_trans = trans_df[trans_df['action'].isin(['Buy', 'Sell'])]
                if len(trade_trans) > 1:
                    fig.add_trace(
                        go.Scatter(
                            x=trade_trans['timestamp'],
                            y=trade_trans['price'],
                            mode='lines',
                            name='Transaction Line',
                            line=dict(color='orange', width=2, dash='dot')
                        ),
                        row=1, col=1
                    )

            # Volume subplot
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['Volume'],
                    name='Volume',
                    marker_color='grey',
                    opacity=0.5
                ),
                row=2, col=1
            )

            # RSI subplot
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['RSI'],
                    mode='lines',
                    name='RSI',
                    line=dict(color='orange', width=2)
                ),
                row=3, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

            # MACD subplot
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['MACD'],
                    mode='lines',
                    name='MACD',
                    line=dict(color='cyan', width=2)
                ),
                row=4, col=1
            )
            fig.add_hline(y=0, line_dash="dash", line_color="black", row=4, col=1)

            # Update layout
            fig.update_layout(
                title='Transaction History for Stock Price',
                xaxis4_title='Date',
                yaxis_title='Price (€)',
                yaxis2_title='Volume',
                yaxis3_title='RSI',
                yaxis4_title='MACD',
                showlegend=True,
                template='plotly_white',
                height=800
            )
            fig.update_xaxes(
                tickformat='%Y-%m-%d',
                rangeslider_visible=False,
                row=4, col=1
            )

            chart_placeholder.plotly_chart(fig, key=f"trans_history_plot_{context}", use_container_width=True)

        except Exception as e:
            Utils.log_message(f"ERROR: Error plotting transaction history: {e}")
            chart_placeholder.plotly_chart(go.Figure(), key=f"error_trans_history_plot_{context}")

    @staticmethod
    def plot_trading_signals(data, actions, chart_placeholder):
        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
            subplot_titles=['Stock Price with Trading Signals', 'Volume', 'RSI', 'MACD'],
            row_heights=[0.5, 0.2, 0.15, 0.15]
        )
        if data is not None and not data.empty:
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='Price',
                    increasing_line_color='green',
                    decreasing_line_color='red'
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['SMA20'],
                    mode='lines',
                    name='SMA20',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['BB_Upper'],
                    mode='lines',
                    name='BB Upper',
                    line=dict(color='purple', width=1, dash='dash')
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['BB_Lower'],
                    mode='lines',
                    name='BB Lower',
                    line=dict(color='purple', width=1, dash='dash')
                ),
                row=1, col=1
            )
            if len(actions) == len(data):
                buy_signals = data.index[actions == 1]
                buy_prices = data['Close'][actions == 1]
                fig.add_trace(
                    go.Scatter(
                        x=buy_signals,
                        y=buy_prices,
                        mode='markers',
                        name='Buy Signal',
                        marker=dict(symbol='triangle-up', size=10, color='green')
                    ),
                    row=1, col=1
                )
                sell_signals = data.index[actions == 2]
                sell_prices = data['Close'][actions == 2]
                fig.add_trace(
                    go.Scatter(
                        x=sell_signals,
                        y=sell_prices,
                        mode='markers',
                        name='Sell Signal',
                        marker=dict(symbol='triangle-down', size=10, color='red')
                    ),
                    row=1, col=1
                )
                trade_indices = data.index[np.isin(actions, [1, 2])]
                trade_prices = data['Close'][np.isin(actions, [1, 2])]
                if len(trade_indices) > 1:
                    fig.add_trace(
                        go.Scatter(
                            x=trade_indices,
                            y=trade_prices,
                            mode='lines',
                            name='Trade Signal Line',
                            line=dict(color='orange', width=2, dash='dot')
                        ),
                        row=1, col=1
                    )
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['Volume'],
                    name='Volume',
                    marker_color='grey',
                    opacity=0.5
                ),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['RSI'],
                    mode='lines',
                    name='RSI',
                    line=dict(color='orange', width=2)
                ),
                row=3, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['MACD'],
                    mode='lines',
                    name='MACD',
                    line=dict(color='cyan', width=2)
                ),
                row=4, col=1
            )
        fig.add_hline(y=0, line_dash="dash", line_color="black", row=4, col=1)
        fig.update_layout(
            title='Trading Signals for Stock Price',
            xaxis4_title='Date',
            yaxis_title='Price (€)',
            yaxis2_title='Volume',
            yaxis3_title='RSI',
            yaxis4_title='MACD',
            showlegend=True,
            template='plotly_white',
            height=800
        )
        fig.update_xaxes(
            tickformat='%Y-%m-%d',
            rangeslider_visible=False,
            row=4, col=1
        )
        chart_placeholder.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def plot_portfolio_comparison(agent_history, bh_history, rnd_history, ticker, title=None):
        try:
            agent_timestamps = [h['timestamp'] for h in agent_history if h['total_value'] is not None]
            agent_values = [Calculations.to_scalar(h['total_value']) for h in agent_history if h['total_value'] is not None]
            bh_timestamps = [h['timestamp'] for h in bh_history if h['total_value'] is not None]
            bh_values = [Calculations.to_scalar(h['total_value']) for h in bh_history if h['total_value'] is not None]
            rnd_timestamps = [h['timestamp'] for h in rnd_history if h['total_value'] is not None]
            rnd_values = [Calculations.to_scalar(h['total_value']) for h in rnd_history if h['total_value'] is not None]
            
            if len(agent_timestamps) < 2:
                Utils.log_message(f"WARNING: Insufficient agent data for comparison chart")
                st.warning("Buy-and-Hold portfolio data is insufficient. Ensure the portfolio is initialized and updated.")
                return

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=agent_timestamps,
                y=agent_values,
                mode='lines',
                name='Agent Portfolio',
                line=dict(color='blue')
            ))
            if bh_values and len(bh_values) > 1:
                fig.add_trace(go.Scatter(
                    x=bh_timestamps,
                    y=bh_values,
                    mode='lines',
                    name='Buy-and-Hold Daily',
                    line=dict(color='green')
                ))
                fig.add_trace(go.Scatter(
                    x=[bh_timestamps[0], bh_timestamps[-1]],
                    y=[bh_values[0], bh_values[-1]],
                    mode='lines',
                    name='Buy-and-Hold Portfolio',
                    line=dict(color='lightgreen', width=1, dash='dot')
                ))
            else:
                Utils.log_message(f"WARNING: Insufficient B&H data for comparison chart")
            
            if rnd_values and len(rnd_values) > 1:
                fig.add_trace(go.Scatter(
                    x=rnd_timestamps,
                    y=rnd_values,
                    mode='lines',
                    name='Random Strategy',
                    line=dict(color='orange', dash='dash')
                ))
            else:
                Utils.log_message(f"WARNING: Insufficient Random Values data for comparison chart")
            
            fig.update_layout(
                title=title or f'Portfolio Value Comparison for {ticker}',
                xaxis_title='Date',
                yaxis_title='Portfolio Value (€)',
                showlegend=True,
                template='plotly_white',
                height=500,
                xaxis=dict(tickformat='%Y-%m-%d')
            )
        except Exception as e:
            Utils.log_message(f"ERROR: Error plotting portfolio comparison: {e}")
            fig = go.Figure()
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def plot_portfolio_over_time(portfolio_values, chart_placeholder):
        if portfolio_values:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(portfolio_values))),
                y=portfolio_values,
                mode='lines',
                name='Portfolio Value',
                line=dict(color='blue')
            ))
            fig.update_layout(
                title='Portfolio Value Over Time',
                xaxis_title='Time Step',
                yaxis_title='Portfolio Value (€)',
                showlegend=True,
                template='plotly_white'
            )
            chart_placeholder.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def plot_training_metrics(epoch_metrics, metrics_table_placeholder, chart_placeholder):
        if epoch_metrics:
            with metrics_table_placeholder.container():
                st.write("**Training History**")
                st.dataframe(pd.DataFrame(epoch_metrics), use_container_width=True)
            try:
                if not epoch_metrics:
                    Utils.log_message(f"WARNING: No epoch metrics for training chart")
                    return go.Figure()
                
                epoch_df = pd.DataFrame(epoch_metrics)
                epoch_df['Epoch'] = epoch_df['Epoch'].astype(int)
                
                metrics_to_plot = {
                    'Total Reward': 'Total Reward',
                    'Portfolio Value (€)': 'Portfolio Value (€)',
                    'Avg Return': 'Avg Return'
                }
                
                fig = go.Figure()
                for display_name, column in metrics_to_plot.items():
                    if column in epoch_df.columns:
                        fig.add_trace(go.Scatter(
                            x=epoch_df['Epoch'],
                            y=epoch_df[column].astype(float),
                            mode='lines+markers',
                            name=display_name,
                            line=dict(width=2)
                        ))
                
                fig.update_layout(
                    title='DQN Training Metrics Over Epochs',
                    xaxis_title='Epoch',
                    yaxis_title='Metric Value',
                    showlegend=True,
                    template='plotly_white',
                    height=500
                )
            except Exception as e:
                Utils.log_message(f"ERROR: Error plotting training metrics: {e}")
                fig = go.Figure()
            chart_placeholder.plotly_chart(fig, use_container_width=True)

    @staticmethod
    @st.cache_data(ttl=1800)
    def plot_correlation_heatmap(data, title="Feature Correlation Matrix", context="correlation"):
        """
        Plot a correlation heatmap for the preprocessed DataFrame's features.

        Args:
            data (pd.DataFrame): Preprocessed DataFrame with FEATURE_COLUMNS.
            title (str): Plot title.
            context (str): Unique identifier for Streamlit key.

        Returns:
            None: Renders the Plotly chart in Streamlit.
        """
        try:
            if data.empty or not all(col in data.columns for col in FEATURE_COLUMNS):
                Utils.log_message(f"WARNING: Invalid data for correlation heatmap")
                st.plotly_chart(go.Figure())
                return

            # Check for constant features
            for col in ['Skewness', 'Kurtosis']:
                if data[col].std() == 0:
                    st.warning(f"Feature {col} has zero variance, which may skew correlations.")

            corr_matrix = data[FEATURE_COLUMNS].corr()
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='Viridis',
                zmin=-1, zmax=1,
                text=corr_matrix.values.round(2),
                texttemplate="%{text}",
                hoverinfo="z"
            ))
            fig.update_layout(
                title=title,
                width=800, height=800,
                xaxis_title="Features",
                yaxis_title="Features",
                xaxis_tickangle=45
            )
            st.plotly_chart(fig, key=f"corr_heatmap_{context}", use_container_width=True)
        except Exception as e:
            Utils.log_message(f"ERROR: Error plotting correlation heatmap: {e}")
            st.plotly_chart(go.Figure(), key=f"error_corr_heatmap_{context}")

    @staticmethod
    @st.cache_data(ttl=1800)
    def plot_indicator_signals_heatmap(data, title="Normalized Indicator Signals Over Time", context="indicator_signals"):
        """
        Plot a time-series heatmap of normalized indicator signals.

        Args:
            data (pd.DataFrame): Preprocessed DataFrame with FEATURE_COLUMNS.
            title (str): Plot title.
            context (str): Unique identifier for Streamlit key.

        Returns:
            None: Renders the Plotly chart in Streamlit.
        """
        try:
            if data.empty or not all(col in data.columns for col in FEATURE_COLUMNS):
                Utils.log_message(f"WARNING: Invalid data for indicator signals heatmap")
                st.plotly_chart(go.Figure())
                return

            indicators = ['RSI', 'MACD', 'ATR', 'BB_Penetration', 'Volatility', 'Returns']
            data_subset = data[indicators].copy()
            # Normalize to [0, 1]
            data_subset = (data_subset - data_subset.min()) / (data_subset.max() - data_subset.min() + 1e-8)
            fig = go.Figure(data=go.Heatmap(
                z=data_subset.T.values,
                x=data_subset.index,
                y=indicators,
                colorscale='Hot',
                zmin=0, zmax=1,
                hovertemplate="Date: %{x}<br>Indicator: %{y}<br>Value: %{z:.2f}<extra></extra>"
            ))
            fig.update_layout(
                title=title,
                xaxis_title="Date",
                yaxis_title="Indicators",
                height=400,
                xaxis_tickformat='%Y-%m-%d'
            )
            st.plotly_chart(fig, key=f"signals_heatmap_{context}", use_container_width=True)
        except Exception as e:
            Utils.log_message(f"ERROR: Error plotting indicator signals heatmap: {e}")
            st.plotly_chart(go.Figure(), key=f"error_signals_heatmap_{context}")

    @staticmethod
    @st.cache_data(ttl=1800)
    def plot_qq_plot(data, indicators=None, title="Q-Q Plot for Tail Analysis", context="qq_plot"):
        """
        Plot Q-Q plots to compare feature distributions to normal, emphasizing tails.

        Args:
            data (pd.DataFrame): Preprocessed DataFrame (not used with caching).
            ticker (str): Stock ticker symbol.
            start_date (str): Start date in YYYY-MM-DD format.
            end_date (str): End date in YYYY-MM-DD format.
            settings (dict): User settings for preprocessing.
            indicators (list): Features to plot (default: Returns, RSI, Volatility).
            title (str): Plot title.
            context (str): Unique identifier for Streamlit key.
        """
        try:
            if data.empty or not all(col in data.columns for col in FEATURE_COLUMNS):
                Utils.log_message(f"WARNING: Invalid cached data for Q-Q plot")
                st.plotly_chart(go.Figure())
                return
            indicators = indicators or ['RSI', 'Volatility', 'Returns']
            n_cols = 3
            n_rows = (len(indicators) + n_cols - 1) // n_cols
            fig = make_subplots(
                rows=n_rows,
                cols=n_cols,
                subplot_titles=[f"{ind}" for ind in indicators],
            )
            for idx, indicator in enumerate(indicators):
                if indicator in data.columns:
                    row = (idx // n_cols) + 1
                    col = (idx % n_cols) + 1
                    values = data[indicator].dropna()
                    # Generate Q-Q plot data
                    (osm, osr), (slope, intercept, r) = stats.probplot(values, dist="norm")
                    # Plot Q-Q points
                    fig.add_trace(
                        go.Scatter(
                            x=osm,
                            y=osr,
                            mode="markers",
                            name=indicator,
                            marker=dict(color="blue", size=5),
                            showlegend=False,
                        ),
                        row=row, col=col
                    )
                    # Plot reference line (normal distribution)
                    x = np.array([osm.min(), osm.max()])
                    y = slope * x + intercept
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=y,
                            mode="lines",
                            name="Normal Reference",
                            line=dict(color="red", width=2),
                            showlegend=False,
                        ),
                        row=row, col=col
                    )
            fig.update_layout(
                title=title,
                xaxis_title="Theoretical Quantiles",
                yaxis_title="Sample Quantiles",
                showlegend=True,
                template="plotly_white",
                height=400,
                width=1000,
            )
            for i in range(1, n_cols + 1):
                fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=i)
                fig.update_yaxes(title_text="Sample Quantiles", row=1, col=i)
            st.plotly_chart(fig, key=f"qq_plot_{context}", use_container_width=True)
        except Exception as e:
            Utils.log_message(f"ERROR: Error plotting Q-Q plot: {e}")
            st.plotly_chart(go.Figure(), key=f"error_qq_plot_{context}")

    @staticmethod
    @st.cache_data(ttl=1800)
    def plot_feature_distribution_with_kde(data, indicators=None, title="Feature Distributions with KDE", context="dist_kde"):
        """
        Plot histograms with KDE for selected features, annotating skewness and kurtosis.

        Args:
            data (pd.DataFrame): Preprocessed DataFrame (not used with caching).
            ticker (str): Stock ticker symbol.
            start_date (str): Start date in YYYY-MM-DD format.
            end_date (str): End date in YYYY-MM-DD format.
            settings (dict): User settings for preprocessing.
            indicators (list): Features to plot (default: Returns, RSI, Volatility).
            title (str): Plot title.
            context (str): Unique identifier for Streamlit key.
        """
        try:
            if data.empty or not all(col in data.columns for col in FEATURE_COLUMNS):
                Utils.log_message(f"WARNING: Invalid cached data for distribution with KDE")
                st.plotly_chart(go.Figure())
                return
            indicators = indicators or ['RSI', 'MACD', 'ATR', 'BB_Penetration', 'Volatility', 'Returns']
            n_cols = 3
            n_rows = (len(indicators) + n_cols - 1) // n_cols
            fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=[f"{ind} (Skew: {data[ind].skew():.2f}, Kurt: {data[ind].kurt():.2f})" for ind in indicators])
            for idx, indicator in enumerate(indicators):
                if indicator in data.columns:
                    row = (idx // n_cols) + 1
                    col = (idx % n_cols) + 1
                    # Histogram
                    fig.add_trace(
                        go.Histogram(
                            x=data[indicator],
                            name=indicator,
                            histnorm='probability density',
                            nbinsx=50,
                            marker_color='blue',
                            opacity=0.5,
                            showlegend=False
                        ),
                        row=row, col=col
                    )
                    # KDE
                    dist = create_distplot([data[indicator].dropna()], group_labels=[indicator], show_hist=False, show_rug=False)
                    for trace in dist['data']:
                        if trace['type'] == 'scatter':
                            fig.add_trace(
                                go.Scatter(
                                    x=trace['x'],
                                    y=trace['y'],
                                    mode='lines',
                                    name=f"{indicator} KDE",
                                    line=dict(color='red', width=2),
                                    showlegend=False
                                ),
                                row=row, col=col
                            )
            fig.update_layout(
                title=title,
                showlegend=False,
                template='plotly_white',
                height=300 * n_rows,
                width=800
            )
            st.plotly_chart(fig, key=f"dist_kde_{context}", use_container_width=True)
        except Exception as e:
            Utils.log_message(f"ERROR: Error plotting distribution with KDE: {e}")
            #st.plotly_chart(go.Figure, key=f"error_dist_kde_{context}")

    @staticmethod
    @st.cache_data(ttl=1800)
    def plot_indicator_timeseries(data, indicators=None, title="Indicator Time Series", context="timeseries"):
        try:
            if data.empty or not all(col in data.columns for col in FEATURE_COLUMNS):
                Utils.log_message(f"WARNING: Invalid data for indicator time series")
                st.plotly_chart(go.Figure())
                return
            indicators = indicators or ['RSI', 'MACD', 'ATR', 'BB_Penetration']
            fig = go.Figure()
            for indicator in indicators:
                if indicator in data.columns:
                    # Normalize to [0, 1]
                    values = (data[indicator] - data[indicator].min()) / (data[indicator].max() - data[indicator].min() + 1e-8)
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=values,
                            mode='lines',
                            name=indicator,
                            line=dict(width=2)
                        )
                    )
            fig.update_layout(
                title=title,
                xaxis_title="Date",
                yaxis_title="Normalized Value",
                showlegend=True,
                template='plotly_white',
                height=400,
                xaxis_tickformat='%Y-%m-%d'
            )
            st.plotly_chart(fig, key=f"timeseries_{context}", use_container_width=True)
        except Exception as e:
            Utils.log_message(f"ERROR: Error plotting indicator time series: {e}")
            st.plotly_chart(go.Figure(), key=f"error_timeseries_{context}")

    @staticmethod
    @st.cache_data(ttl=1800)
    def plot_feature_boxplots(data, indicators=None, title="Feature Variability", context="boxplots"):
        try:
            if data.empty or not all(col in data.columns for col in FEATURE_COLUMNS):
                Utils.log_message(f"WARNING: Invalid data for feature boxplots")
                st.plotly_chart(go.Figure())
                return
            indicators = indicators or ['RSI', 'MACD', 'ATR', 'Volatility', 'Returns']
            fig = go.Figure()
            for indicator in indicators:
                if indicator in data.columns:
                    fig.add_trace(
                        go.Box(
                            y=data[indicator],
                            name=indicator,
                            boxpoints='outliers',
                            jitter=0.3,
                            marker_color='purple'
                        )
                    )
            fig.update_layout(
                title=title,
                yaxis_title="Value",
                showlegend=False,
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig, key=f"boxplots_{context}", use_container_width=True)
        except Exception as e:
            Utils.log_message(f"ERROR: Error plotting feature boxplots: {e}")
            st.plotly_chart(go.Figure(), key=f"error_boxplots_{context}")

    @staticmethod
    @st.cache_data(ttl=1800)
    def plot_rf_feature_importance(_agent, title="RF Feature Importance", context="feature_importance"):
        try:
            if not hasattr(_agent.rf_agent, 'rf_model') or _agent.rf_agent.rf_model is None:
                Utils.log_message(f"WARNING: RF model not trained for feature importance")
                st.plotly_chart(go.Figure())
                return
            importances = _agent.rf_agent.rf_model.feature_importances_
            # Map PCA components back to original features
            feature_names = [f"PC{i+1}" for i in range(len(importances))]
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=feature_names,
                    y=importances,
                    marker_color='teal',
                    text=[f"{imp:.3f}" for imp in importances],
                    textposition='auto'
                )
            )
            fig.update_layout(
                title=title,
                xaxis_title="PCA Components",
                yaxis_title="Importance",
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig, key=f"feature_importance_{context}", use_container_width=True)
        except Exception as e:
            Utils.log_message(f"ERROR: Error plotting RF feature importance: {e}")
            st.plotly_chart(go.Figure(), key=f"error_feature_importance_{context}")

    @staticmethod
    @st.cache_data(ttl=1800)
    def plot_rolling_volatility(data, window=20, title="Rolling Volatility", context="volatility"):
        try:
            if data.empty or 'Returns' not in data.columns:
                Utils.log_message(f"WARNING: Invalid data for rolling volatility")
                st.plotly_chart(go.Figure())
                return
            volatility = data['Returns'].rolling(window=window, min_periods=1).std() * np.sqrt(252) * 100  # Annualized
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=volatility,
                    mode='lines',
                    name='Volatility (%)',
                    line=dict(color='red', width=2)
                )
            )
            fig.update_layout(
                title=title,
                xaxis_title="Date",
                yaxis_title="Annualized Volatility (%)",
                template='plotly_white',
                height=400,
                xaxis_tickformat='%Y-%m-%d'
            )
            st.plotly_chart(fig, key=f"volatility_{context}", use_container_width=True)
        except Exception as e:
            Utils.log_message(f"ERROR: Error plotting rolling volatility: {e}")
            st.plotly_chart(go.Figure(), key=f"error_volatility_{context}")

    @staticmethod
    @st.cache_data(ttl=1800)
    def plot_pair_scatter(data, indicators=None, title="Feature Pair Scatter", context="pair_scatter"):
        try:
            if data.empty or not all(col in data.columns for col in FEATURE_COLUMNS):
                Utils.log_message(f"WARNING: Invalid data for pair scatter")
                st.plotly_chart(go.Figure())
                return
            indicators = indicators or ['RSI', 'MACD', 'ATR']
            fig = make_subplots(
                rows=len(indicators), cols=len(indicators),
                subplot_titles=[f"{i1} vs {i2}" for i1 in indicators for i2 in indicators]
            )
            for i, ind1 in enumerate(indicators):
                for j, ind2 in enumerate(indicators):
                    fig.add_trace(
                        go.Scatter(
                            x=data[ind1],
                            y=data[ind2],
                            mode='markers',
                            marker=dict(size=5, opacity=0.5),
                            showlegend=False
                        ),
                        row=i+1, col=j+1
                    )
            fig.update_layout(
                title=title,
                showlegend=False,
                template='plotly_white',
                height=200 * len(indicators),
                width=200 * len(indicators)
            )
            st.plotly_chart(fig, key=f"pair_scatter_{context}", use_container_width=True)
        except Exception as e:
            Utils.log_message(f"ERROR: Error plotting pair scatter: {e}")
            st.plotly_chart(go.Figure(), key=f"error_pair_scatter_{context}")
