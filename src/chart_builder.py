import os
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import sys
sys.dont_write_bytecode = True

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from calculations import Calculations
from utils import Utils

class ChartBuilder:
    """Centralized class for creating Plotly charts for portfolio and training visualization."""
    
    @staticmethod
    def plot_portfolio_performance(history, title='Portfolio Performance Over Time', show_cash_positions=True, show_metrics=True):
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
            
            # Compute metrics
            metrics = Calculations.compute_metrics(history, value_key='total_value')
            
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
                    x=0.01, y=0.99,
                    text=metrics_table,
                    showarrow=False,
                    font=dict(size=12),
                    align="left",
                    bgcolor="white",
                    bordercolor="black",
                    borderpad=4
                )
            
            return fig
        except Exception as e:
            Utils.log_message(f"ERROR: Error plotting portfolio performance: {e}")
            return go.Figure()

    @staticmethod
    def plot_trading_signals(data, actions, chart_placeholder):
        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
            subplot_titles=['Stock Price with Trading Signals', 'Volume', 'RSI', 'MACD'],
            row_heights=[0.5, 0.2, 0.15, 0.15]
        )
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
    def plot_portfolio_comparison(agent_history, bh_history, ticker, title=None):
        try:
            agent_timestamps = [h['timestamp'] for h in agent_history if h['total_value'] is not None]
            agent_values = [Calculations.to_scalar(h['total_value']) for h in agent_history if h['total_value'] is not None]
            bh_timestamps = [h['timestamp'] for h in bh_history if h['total_value'] is not None]
            bh_values = [Calculations.to_scalar(h['total_value']) for h in bh_history if h['total_value'] is not None]
            
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
            if len(bh_values) > 1:
                fig.add_trace(go.Scatter(
                    x=bh_timestamps,
                    y=bh_values,
                    mode='lines',
                    name='Buy-and-Hold Daily',
                    line=dict(color='lightgreen', width=1, dash='dot')
                ))
                fig.add_trace(go.Scatter(
                    x=[bh_timestamps[0], bh_timestamps[-1]],
                    y=[bh_values[0], bh_values[-1]],
                    mode='lines',
                    name='Buy-and-Hold Portfolio',
                    line=dict(color='green')
                ))
            else:
                Utils.log_message(f"WARNING: Insufficient B&H data for comparison chart")
            
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