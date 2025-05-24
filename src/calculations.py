import os
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import sys
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from utils import Utils
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

class Calculations:
    """Centralized class for computing divers calculations."""
    
    @staticmethod
    def compute_metrics(history, value_key='total_value', is_bh=False, confidence_level=0.95):
        """
        Compute portfolio metrics: Sharpe ratio, Sortino ratio, Calmar ratio, max drawdown,
        current drawdown, total return, VaR, and CVaR (tail risk).

        Args:
            history (list[dict]): Portfolio history with 'total_value' or other value_key.
            value_key (str): Key for portfolio values (default: 'total_value').
            is_bh (bool): If True, treat as buy-and-hold portfolio (affects drawdown).
            confidence_level (float): Confidence level for VaR and CVaR (default: 0.95).

        Returns:
            dict: Metrics containing sharpe_ratio, sortino_ratio, calmar_ratio, max_drawdown,
                current_drawdown, total_return, var, cvar.
                Returns zeros if insufficient data.
        """
        try:
            values = np.array([Calculations.to_scalar(h[value_key]) for h in history if h[value_key] is not None])
            if len(values) < 2:
                Utils.log_message(f"WARNING: Insufficient history for metrics: {len(values)} entries")
                return {
                    'sharpe_ratio': 0.0, 'sortino_ratio': 0.0, 'calmar_ratio': 0.0,
                    'max_drawdown': 0.0, 'current_drawdown': 0.0, 'total_return': 0.0,
                    'var': 0.0, 'cvar': 0.0
                }

            # Total return (%)
            total_return = (values[-1] - values[0]) / values[0] * 100

            # Drawdowns (€ and %)
            max_drawdown = 0.0
            current_drawdown = 0.0
            if is_bh:
                # Max historical drawdown for B&H (€)
                max_drawdown = np.max(np.maximum.accumulate(values) - values)
                max_drawdown = (max_drawdown / np.maximum.accumulate(values).max() * 100) if values.max() > 0 else 0
            else:
                # Max and current drawdown for agent (%)
                max_value = np.max(values)
                max_drawdown = np.max(np.maximum.accumulate(values) - values)
                max_drawdown = (max_drawdown / max_value * 100) if max_value > 0 else 0
                current_drawdown = (max_value - values[-1]) / max_value * 100 if max_value > 0 else 0

            # Daily returns for ratios and VaR
            daily_returns = np.diff(values) / values[:-1]

            # Sharpe ratio
            sharpe_ratio = 0.0
            if len(values) > 10:
                sharpe_ratio = np.sqrt(252) * np.mean(daily_returns) / (np.std(daily_returns) + 1e-9)

            # Sortino ratio (uses downside deviation)
            sortino_ratio = 0.0
            if len(values) > 10:
                downside_returns = daily_returns[daily_returns < 0]
                downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-9
                sortino_ratio = np.sqrt(252) * np.mean(daily_returns) / (downside_std + 1e-9)

            # Calmar ratio (annualized return / max drawdown)
            calmar_ratio = 0.0
            if len(values) > 10 and max_drawdown > 0:
                annualized_return = ((values[-1] / values[0]) ** (252 / len(values)) - 1) * 100
                calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0.0

            # VaR and CVaR (Value at Risk and Conditional VaR)
            var = 0.0
            cvar = 0.0
            if len(daily_returns) > 10:
                var = -np.percentile(daily_returns, (1 - confidence_level) * 100) * values[-1]
                tail_losses = daily_returns[daily_returns <= -var / values[-1]]
                cvar = -np.mean(tail_losses) * values[-1] if len(tail_losses) > 0 else var

            return {
                'sharpe_ratio': float(sharpe_ratio),
                'sortino_ratio': float(sortino_ratio),
                'calmar_ratio': float(calmar_ratio),
                'max_drawdown': float(max_drawdown),
                'current_drawdown': float(current_drawdown),
                'total_return': float(total_return),
                'var': float(var),
                'cvar': float(cvar)
            }
        except Exception as e:
            Utils.log_message(f"ERROR: Error computing metrics: {e}")
            return {
                'sharpe_ratio': 0.0, 'sortino_ratio': 0.0, 'calmar_ratio': 0.0,
                'max_drawdown': 0.0, 'current_drawdown': 0.0, 'total_return': 0.0,
                'var': 0.0, 'cvar': 0.0
            }

    @staticmethod
    def compute_rsi(series, period=14):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    @staticmethod
    def compute_macd(series, params=(12, 26, 9)):
        ema12 = series.ewm(span=params[0], adjust=False).mean()
        ema26 = series.ewm(span=params[1], adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=params[2], adjust=False).mean()
        return macd - signal

    @staticmethod
    def compute_bollinger_band(series, window=20):
        sma = series.rolling(window=window, min_periods=1).mean()
        std = series.rolling(window=window, min_periods=1).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        normalized = (series - sma) / (std + 1e-9)
        return upper, lower, normalized

    @staticmethod
    def compute_atr(data, period=14, smoothing=False):
        high = data['High']
        low = data['Low']
        close = data['Close']
        tr = pd.DataFrame(index=data.index)
        tr['h_l'] = high - low
        tr['h_pc'] = abs(high - close.shift(1))
        tr['l_pc'] = abs(low - close.shift(1))
        tr['tr'] = tr[['h_l', 'h_pc', 'l_pc']].max(axis=1)
        if smoothing:
            atr = tr['tr'].ewm(span=period, adjust=False).mean()
        else:
            atr = tr['tr'].rolling(window=period, min_periods=1).mean()
        return atr

    @staticmethod
    def compute_softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    @staticmethod
    def perform_smote(X_train, y_train, k_neighbors=3, random_state=42):
        smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        unique, counts = np.unique(y_train, return_counts=True)
        label_distribution = dict(zip(unique, counts))
        return X_train, y_train, label_distribution

    @staticmethod
    def apply_pca(X_train, X_val, n_components=0.95, random_state=42):
        pca = PCA(n_components=n_components, random_state=random_state)
        pca.fit(X_train)
        X_train = pca.transform(X_train)
        X_val = pca.transform(X_val)
        n_pca_components = X_train.shape[1]
        return {"pca": pca, "X_train": X_train, "X_val": X_val, "n_pca_components": n_pca_components}

    @staticmethod
    def calculate_reward(prev_value, curr_value, action, fee_history, portfolio_value_history, data, index, initial_cash, epoch_returns):
        """
        Calculate the reward for a trading action based on returns, fees, drawdown, and technical indicators.

        Args:
            prev_value (float): Portfolio value before the action.
            curr_value (float): Portfolio value after the action.
            action (int): Action taken (0: Hold, 1: Buy, 2: Sell).
            fee_history (list): List of recent transaction fees.
            portfolio_value_history (list): History of portfolio values.
            data (pd.DataFrame): DataFrame containing price and indicator data.
            index (int): Current index in the data.
            initial_cash (float): Initial portfolio cash for scaling penalties.
            epoch_returns (list): List of returns for the current epoch.

        Returns:
            tuple: (reward, return) as floats.
        """
        # Calculate return
        ret = (curr_value - prev_value) / (prev_value + 1e-9)
        epoch_returns.append(ret)

        # Volatility (use ATR if available, else rolling std, capped for stability)
        volatility = data.get('ATR', data['Close'].pct_change().rolling(14).std()).iloc[index]
        if pd.isna(volatility) or volatility == 0:
            volatility = 0.02
        volatility = min(volatility, 0.05)  # Cap volatility to prevent excessive penalties
        avg_volatility = data.get('ATR', data['Close'].pct_change().rolling(252).std()).mean() or 0.02

        # Drawdown penalty (reduced severity)
        drawdown_penalty = 0.0
        if portfolio_value_history:
            max_value = np.max(portfolio_value_history)
            drawdown = (max_value - curr_value) / (max_value + 1e-9)
            trend = data['Close'].iloc[index] > data['Close'].rolling(50).mean().iloc[index]
            # Penalize drawdown, with higher penalty for Buy in uptrend or non-Sell actions
            drawdown_penalty = drawdown * (0.2 if action == 1 and trend else 0.3 if action != 2 else 0.1)

        # Fee penalty (reduced volatility impact)
        fee_penalty = 0.0
        if action != 0 and fee_history:
            recent_fees = fee_history[-1]
            portfolio_scale = curr_value / (initial_cash + 1e-9)
            volatility_factor = 0.3 if action == 2 else 0.5  # Lowered factors
            fee_penalty = recent_fees * (1.0 + volatility / (avg_volatility + 1e-9) * volatility_factor) / portfolio_scale
            fee_penalty = min(fee_penalty, 0.3)  # Lower cap

        # Slippage penalty (reduced multiplier)
        slippage = 0.0
        if action in [1, 2]:
            trade_size = abs(curr_value - prev_value) / (curr_value + 1e-9)
            slippage = trade_size * volatility * 0.05  # Reduced from 0.1

        # Base reward (increased for Buy and Sell)
        reward = 0.0
        indicator_bonus = 0.0
        if action == 2:  # Sell
            reward += 0.7  # Increased from 0.5
            current_rsi_div = data['RSI_Divergence'].iloc[index] if 'RSI_Divergence' in data.columns and not pd.isna(data['RSI_Divergence'].iloc[index]) else 0
            current_bb_pen = data.get('BB_Penetration', pd.Series(0)).iloc[index] if not pd.isna(data.get('BB_Penetration', pd.Series(0)).iloc[index]) else 0
            if current_rsi_div == 1 and ret > 0:
                indicator_bonus += 0.3
            if current_bb_pen == 1 and data['Close'].iloc[index] < data['Close'].rolling(20).mean().iloc[index]:
                indicator_bonus += 0.2
        elif action == 1:  # Buy
            reward += 0.4  # Increased from 0.2, removed volatility condition
            # Add Buy-specific indicator bonuses
            current_rsi = data.get('RSI', pd.Series(50)).iloc[index]
            current_macd = data.get('MACD', pd.Series(0)).iloc[index]
            if current_rsi < 30 and not pd.isna(current_rsi):
                indicator_bonus += 0.3
            if current_macd > 0 and not pd.isna(current_macd):
                indicator_bonus += 0.2
        elif action == 0:  # Hold
            if volatility < 0.01:
                reward += 0.3 * (initial_cash / curr_value)  # Increased from 0.2
            elif 0.01 <= volatility <= 0.03:
                reward += 0.1  # New reward for moderate volatility
            elif volatility > 0.03:
                reward += 0.05

        # Risk-adjusted reward (reduced weight, longer window)
        sharpe_reward = 0.0
        if len(epoch_returns) > 1:
            mean_ret = np.mean(epoch_returns[-200:])  # Increased from 100
            std_ret = np.std(epoch_returns[-200:]) + 1e-9
            sharpe_reward = np.clip(mean_ret / std_ret, -2, 2) * 0.5

        # Long-term growth bonus (shorter window, higher multiplier)
        growth_bonus = 0.0
        if len(portfolio_value_history) > 50:  # Reduced from 100
            long_term_growth = (curr_value - portfolio_value_history[-50]) / portfolio_value_history[-50]
            growth_bonus = 0.5 * np.clip(long_term_growth, 0, 1)  # Increased from 0.3

        # Combine components (adjusted weights)
        reward = 0.5 * sharpe_reward + 0.5 * (reward + indicator_bonus) - fee_penalty - drawdown_penalty - slippage + growth_bonus

        return float(reward), float(ret)

    @staticmethod
    def to_scalar(value):
        if isinstance(value, (np.ndarray, pd.Series)):
            return float(value.item() if value.size == 1 else value[-1])
        elif isinstance(value, (np.floating, np.integer)):
            return float(value)
        return float(value)