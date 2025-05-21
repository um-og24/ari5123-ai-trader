import os
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
sys.dont_write_bytecode = True

import random
import pickle
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import deque
from sklearn.preprocessing import StandardScaler
from scipy.special import softmax
from itertools import product
from utils import Utils, FEATURE_COLUMNS
from calculations import Calculations

class DQNAgent:
    def __init__(self, ticker, training_start_date, training_end_date, model_dir, lookback, initial_cash, trade_fee, risk_per_trade, atr_multiplier, atr_period, atr_smoothing, expected_feature_columns=FEATURE_COLUMNS):
        tf.keras.backend.clear_session()
        self.ticker = ticker
        self.training_start_date = pd.Timestamp(training_start_date).date()
        self.training_end_date = pd.Timestamp(training_end_date).date()
        self.lookback = lookback
        self.initial_cash = initial_cash
        self.trade_fee = trade_fee
        self.risk_per_trade = risk_per_trade
        self.atr_multiplier = atr_multiplier
        self.atr_period = atr_period
        self.atr_smoothing = atr_smoothing
        self.expected_feature_columns = expected_feature_columns
        self.feature_columns = [f"{col}_{i}" for i in range(lookback) for col in self.expected_feature_columns]
        
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

        self.state_scaler = StandardScaler()
        
        self.memory = deque(maxlen=15000)
        self.gamma = 0.95
        self.epsilon = 0.7
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.999
        self.temperature = 1.0
        self.temperature_min = 0.1
        self.temperature_decay = 0.999
        self.batch_size = 64

        # Portfolio management
        self.cash = initial_cash
        self.shares = 0
        self.fee_history = []
        self.cumulative_fees = 0
        self.portfolio_value_history = []
        self.returns = []
        self.trade_count = 0  # Explicit trade counter

        # Model storage paths
        if not model_dir or not isinstance(model_dir, str):
            raise ValueError("model_dir must be a non-empty string")
        if os.path.basename(model_dir) == ticker:
            self.model_dir = model_dir
        else:
            self.model_dir = os.path.join(model_dir, ticker)
        os.makedirs(self.model_dir, exist_ok=True)

        self.model_best_weights_save_path = os.path.join(self.model_dir, f"{ticker}_dqn_model_best.keras")
        self.checkpoint_model_path = os.path.join(self.model_dir, "checkpoints", f"{ticker}_checkpoint_dqn.keras")
        self.checkpoint_state_path = os.path.join(self.model_dir, "checkpoints", f"{ticker}_checkpoint_dqn_state.pkl")
        self.settings_save_path = os.path.join(self.model_dir, f"{ticker}_dqn_settings.json")

    def _build_model(self):
        def build():
            model = tf.keras.models.Sequential([
                tf.keras.layers.Input(shape=(self.lookback, len(self.expected_feature_columns))),
                tf.keras.layers.LSTM(64, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l1_l2(0.01)),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.LSTM(32),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(24, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
                tf.keras.layers.Dense(3, activation='linear')
            ])
            model.compile(loss='mse', optimizer='adam')
            return model
        return Utils.perform_using_retries(build)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def reset_portfolio(self):
        self.cash = self.initial_cash
        self.shares = 0
        self.fee_history = []
        self.cumulative_fees = 0
        self.portfolio_value_history = []
        self.returns = []
        self.trade_count = 0

    def simulate_trade(self, action, price, max_trades_per_epoch, max_fee_per_epoch, data=None, index=None, portfolio_tracker=None):
        if action is None:
            return self.get_portfolio_value(price)
        
        if max_trades_per_epoch > 0 and self.trade_count >= max_trades_per_epoch:
            Utils.log_message(f"INFO: Trade skipped: Maximum trades per epoch reached ({self.trade_count}/{max_trades_per_epoch})")
            return self.get_portfolio_value(price)
        if max_fee_per_epoch > 0 and self.cumulative_fees >= max_fee_per_epoch:
            Utils.log_message(f"INFO: Trade skipped: Maximum fee per epoch reached")
            return self.get_portfolio_value(price)
        
        current_value = self.get_portfolio_value(price)
        if current_value < self.initial_cash * 0.5:
            Utils.log_message(f"INFO: Trade skipped: Portfolio value below 50% of initial cash")
            return current_value
        
        atr = 0.02 * price
        if data is not None and index is not None and 'ATR' in data.columns:
            atr_value = data['ATR'].iloc[index]
            atr = float(atr_value) if pd.notna(atr_value) else atr
        
        # Ensure realistic stop-loss distance
        min_stop_loss = price * 0.01  # 1% of price
        stop_loss_distance = max(atr * self.atr_multiplier, min_stop_loss)
        max_risk_amount = self.cash * self.risk_per_trade
        buy_fee = self.trade_fee
        sell_fee = self.trade_fee
        fee_adjusted_price = price * (1 + buy_fee)
        future_selling_fee = price * sell_fee
        max_shares = np.floor(max_risk_amount / (stop_loss_distance + future_selling_fee))
        max_shares_by_cash = np.floor((self.cash * 0.5) / fee_adjusted_price)
        # Upper bound: max 25% of portfolio value
        max_shares_by_portfolio = np.floor((current_value * 0.25) / fee_adjusted_price)
        shares_to_buy = max(min(max_shares, max_shares_by_cash, max_shares_by_portfolio), 1)
        
        Utils.log_message(f"INFO: Trade calc: price={price:.2f}, atr={atr:.4f}, stop_loss_distance={stop_loss_distance:.2f}, max_risk_amount={max_risk_amount:.2f}, max_shares={max_shares}, max_shares_by_cash={max_shares_by_cash}, max_shares_by_portfolio={max_shares_by_portfolio}, shares_to_buy={shares_to_buy}")
        
        if portfolio_tracker is None:
            if action == 1:
                cost_before_fees = shares_to_buy * price
                fee_amount = cost_before_fees * buy_fee
                total_cost = cost_before_fees + fee_amount
                if self.cash >= total_cost and shares_to_buy > 0:
                    if fee_amount < 0:
                        Utils.log_message(f"ERROR: Negative fee amount: {fee_amount}")
                        return current_value
                    self.cumulative_fees += fee_amount
                    self.fee_history.append(fee_amount)
                    self.cash -= total_cost
                    self.shares += shares_to_buy
                    self.trade_count += 1
                    Utils.log_message(f"INFO: Buy executed: {shares_to_buy} shares at {price}, total cost: {total_cost}")
                else:
                    Utils.log_message(f"INFO: Trade skipped: Insufficient cash ({self.cash:.2f} < {total_cost:.2f}) or invalid shares_to_buy")
            elif action == 2 and self.shares > 0:
                revenue_before_fees = self.shares * price
                fee_amount = revenue_before_fees * sell_fee
                if fee_amount < 0:
                    Utils.log_message(f"ERROR: Negative fee amount: {fee_amount}")
                    return current_value
                net_revenue = revenue_before_fees - fee_amount
                self.cumulative_fees += fee_amount
                self.fee_history.append(fee_amount)
                self.cash += net_revenue
                self.shares = 0
                self.trade_count += 1
                Utils.log_message(f"INFO: Sell executed: {self.shares} shares at {price}, net revenue: {net_revenue}")
            else:
                Utils.log_message(f"INFO: Trade skipped: Action={action}, Cash={self.cash}, Shares={self.shares}, Shares_to_buy={shares_to_buy}")
        else:
            if action == 1:
                cost_before_fees = shares_to_buy * price
                fee_amount = cost_before_fees * buy_fee
                total_cost = cost_before_fees + fee_amount
                if portfolio_tracker.cash >= total_cost and shares_to_buy > 0:
                    success, transaction_id = portfolio_tracker.buy(self.ticker, shares_to_buy, price, fee_percentage=buy_fee)
                    if success:
                        if fee_amount < 0:
                            Utils.log_message(f"ERROR: Negative fee amount: {fee_amount}")
                            return current_value
                        self.cumulative_fees += fee_amount
                        self.fee_history.append(fee_amount)
                        self.cash = portfolio_tracker.cash
                        self.shares = portfolio_tracker.holdings.get(self.ticker, {'quantity': 0})['quantity']
                        self.trade_count += 1
                        Utils.log_message(f"INFO: Buy executed via PortfolioTracker: {shares_to_buy} shares at {price}, transaction_id: {transaction_id}")
                    else:
                        Utils.log_message(f"INFO: Buy failed: {transaction_id}")
                else:
                    Utils.log_message(f"INFO: Trade skipped: Insufficient cash ({portfolio_tracker.cash:.2f} < {total_cost:.2f}) or invalid shares_to_buy")
            elif action == 2 and self.shares > 0:
                success, transaction_id = portfolio_tracker.sell(self.ticker, self.shares, price, fee_percentage=sell_fee)
                if success:
                    revenue_before_fees = self.shares * price
                    fee_amount = revenue_before_fees * sell_fee
                    if fee_amount < 0:
                        Utils.log_message(f"ERROR: Negative fee amount: {fee_amount}")
                        return current_value
                    self.cumulative_fees += fee_amount
                    self.fee_history.append(fee_amount)
                    self.cash = portfolio_tracker.cash
                    self.shares = portfolio_tracker.holdings.get(self.ticker, {'quantity': 0})['quantity']
                    self.trade_count += 1
                    Utils.log_message(f"INFO: Sell executed via PortfolioTracker: {self.shares} shares at {price}, transaction_id: {transaction_id}")
                else:
                    Utils.log_message(f"INFO: Sell failed: {transaction_id}")
            else:
                Utils.log_message(f"INFO: Trade skipped: Action={action}, Cash={portfolio_tracker.cash}, Shares={self.shares}, Shares_to_buy={shares_to_buy}")

        current_value = self.get_portfolio_value(price)
        self.portfolio_value_history.append(current_value)
        return current_value

    def get_portfolio_value(self, price):
        return self.cash + self.shares * price

    def preprocess_data(self, data):
        processed_data = Utils.preprocess_data(data, volatility_period=self.atr_period, atr_smoothing=self.atr_smoothing) if not set(FEATURE_COLUMNS).issubset(data.columns) else data
        if not all(col in data.columns for col in self.expected_feature_columns):
            raise ValueError("Data is missing required columns")
        data = processed_data.copy()
        try:
            data_scaled = self.state_scaler.fit_transform(data) if not hasattr(self.state_scaler, 'mean_') else self.state_scaler.transform(data)
            Utils.log_message(f"DEBUG: State scaler mean: {self.state_scaler.mean_[:5]}, scale: {self.state_scaler.scale_[:5]}")
        except ValueError:
            Utils.log_message(f"INFO: Fitting state scaler")
            self.state_scaler.fit(data)
            data_scaled = self.state_scaler.transform(data)
        states = []
        for i in range(self.lookback, len(data)):
            state = data_scaled[i - self.lookback:i]
            if state.shape != (self.lookback, len(self.expected_feature_columns)):
                Utils.log_message(f"ERROR: Unexpected state shape at index {i}: {state.shape}, expected ({self.lookback}, {len(self.expected_feature_columns)})")
                return np.array([])
            states.append(state)
        states = np.array(states)
        Utils.log_message(f"INFO: Preprocessed data shape: {states.shape}, expected features: {self.expected_feature_columns}")
        return states

    def act(self, state):
        self.action_counts = getattr(self, 'action_counts', {'Hold': 0, 'Buy': 0, 'Sell': 0})
        
        expected_shape = (self.lookback, len(self.expected_feature_columns))
        if state.shape != expected_shape:
            Utils.log_message(f"ERROR: Invalid state shape: {state.shape}, expected {expected_shape}")
            return 0
        
        try:
            q_values = self.model.predict(state.reshape(1, self.lookback, len(self.expected_feature_columns)), verbose=0)[0]
            probabilities = softmax(q_values / self.temperature)
            action = np.random.choice([0, 1, 2], p=probabilities)
            action_name = ['Hold', 'Buy', 'Sell'][action]
            self.action_counts[action_name] += 1
            Utils.log_message(f"INFO: Softmax action: {action} ({action_name}), Probabilities: {probabilities}, Temperature: {self.temperature:.3f}, Counts: {self.action_counts}")
            if self.temperature > self.temperature_min:
                self.temperature *= self.temperature_decay
            return action
        except ValueError as e:
            Utils.log_message(f"ERROR: Error in DQN prediction: {e}, state shape: {state.shape}")
            return 0

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, float(np.squeeze(reward)), next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([m[0] for m in minibatch])
        actions = np.array([m[1] for m in minibatch])
        rewards = np.array([m[2] for m in minibatch])
        next_states = np.array([m[3] for m in minibatch])
        dones = np.array([m[4] for m in minibatch])
        target_q = self.model.predict(states, verbose=0)
        next_q = self.target_model.predict(next_states, verbose=0)
        for i in range(self.batch_size):
            if dones[i]:
                target_q[i][actions[i]] = rewards[i]
            else:
                target_q[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q[i])
        self.model.fit(states, target_q, epochs=1, verbose=0, batch_size=self.batch_size)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def predict(self, state):
        try:
            q_values = self.model.predict(state.reshape(1, self.lookback, len(self.expected_feature_columns)), verbose=0)[0]
            probabilities = softmax(q_values)
            action = int(np.argmax(probabilities))
            return action, probabilities
        except Exception as e:
            Utils.log_message(f"ERROR: DQN prediction failed: {e}. Defaulting to Hold.")
            return 0, [0.333, 0.333, 0.333]

    def evaluate_performance(self, data=None):
        if data is None:
            Utils.log_message(f"WARNING: No data provided for DQN performance evaluation. Using existing portfolio history.")
            values = np.array(self.portfolio_value_history)
            if len(values) < 2:
                return 0.0, 0.0, float(self.cumulative_fees), 0.0
            returns = np.diff(values) / values[:-1]
            sharpe_ratio = float(np.sqrt(252) * np.mean(returns) / (np.std(returns) + 1e-9))
            drawdown = float(np.max(np.maximum.accumulate(values) - values))
            return sharpe_ratio, drawdown, float(self.cumulative_fees), 0.0
        
        processed_data = self.preprocess_data(data)
        if processed_data.size == 0:
            Utils.log_message(f"ERROR: No valid preprocessed data for DQN performance evaluation")
            return 0.0, 0.0, 0.0, 0.0
        
        states = self.preprocess_data(processed_data)
        if len(states) == 0:
            Utils.log_message(f"ERROR: No valid states for DQN performance evaluation")
            return 0.0, 0.0, 0.0, 0.0
        
        self.reset_portfolio()
        portfolio_values = []
        num_trades = 0
        
        for t in range(self.lookback, len(processed_data) - 1):
            if t >= len(states) + self.lookback:
                Utils.log_message(f"ERROR: Index t={t} exceeds preprocessed states length {len(states)}")
                break
            state = states[t - self.lookback]
            action = self.act(state)
            price = processed_data['Close'].iloc[t]
            prev_value = self.simulate_trade(action, price, data=processed_data, index=t)
            portfolio_values.append(prev_value)
            if action in [1, 2] and self.fee_history and self.fee_history[-1] > 0:
                num_trades += 1
        
        values = np.array(portfolio_values)
        if len(values) < 2:
            Utils.log_message(f"WARNING: Insufficient portfolio values for performance metrics")
            return 0.0, 0.0, self.cumulative_fees, 0.0
        
        returns = np.diff(values) / values[:-1]
        sharpe_ratio = np.sqrt(252) * np.mean(returns) / (np.std(returns) + 1e-9)
        drawdown = np.max(np.maximum.accumulate(values) - values)
        Utils.log_message(f"INFO: DQN performance: Sharpe={sharpe_ratio:.4f}, Drawdown={drawdown:.2f}, Fees={self.cumulative_fees:.2f}, Trades={num_trades}")
        return sharpe_ratio, drawdown, self.cumulative_fees, 0.0

    def _save_settings(self):
        settings = {
            'ticker': self.ticker,
            'training_start_date': pd.Timestamp(self.training_start_date).isoformat(),
            'training_end_date': pd.Timestamp(self.training_end_date).isoformat(),
            'lookback': self.lookback,
            'trade_fee': self.trade_fee,
            'risk_per_trade': self.risk_per_trade,
            'atr_multiplier': self.atr_multiplier,
            'atr_period': self.atr_period,
            'atr_smoothing': self.atr_smoothing
        }
        try:
            os.makedirs(self.model_dir, exist_ok=True)
            with open(self.settings_save_path, 'w') as f:
                json.dump(settings, f, indent=4)
            Utils.log_message(f"INFO: Saved DQN settings to {self.settings_save_path}")
        except Exception as e:
            Utils.log_message(f"ERROR: Failed to save DQN settings: {e}")

    def _load_settings(self):
        if os.path.exists(self.settings_save_path):
            try:
                with open(self.settings_save_path, 'r') as f:
                    settings = json.load(f)
                self.ticker = settings.get('ticker', self.ticker)
                self.training_start_date = pd.Timestamp(settings['training_start_date']).date()
                self.training_end_date = pd.Timestamp(settings['training_end_date']).date()
                self.lookback = settings.get('lookback', self.lookback)
                self.trade_fee = settings.get('trade_fee', self.trade_fee)
                self.risk_per_trade = settings.get('risk_per_trade', self.risk_per_trade)
                self.atr_multiplier = settings.get('atr_multiplier', self.atr_multiplier)
                self.atr_period = settings.get('atr_period', self.atr_period)
                self.atr_smoothing = settings.get('atr_smoothing', self.atr_smoothing)
                Utils.log_message(f"INFO: Loaded DQN settings from {self.settings_save_path}: {settings}")
                return settings
            except Exception as e:
                Utils.log_message(f"ERROR: Failed to load DQN settings: {e}")
        else:
            Utils.log_message(f"INFO: No DQN settings file found at {self.settings_save_path}")
        return None

    def save_model(self):
        try:
            os.makedirs(self.model_dir, exist_ok=True)
            self.model.save(self.model_best_weights_save_path)
            self._save_settings()
            Utils.log_message(f"INFO: Saved DQN model to {self.model_best_weights_save_path}")
        except Exception as e:
            Utils.log_message(f"ERROR: Failed to save DQN model: {e}")
            raise

    def load_model(self):
        if not os.path.exists(self.model_best_weights_save_path):
            Utils.log_message(f"WARNING: No DQN model found at {self.model_best_weights_save_path}. Starting with fresh model.")
            return
        def load():
            try:
                self.model = tf.keras.models.load_model(self.model_best_weights_save_path)
                self.target_model = tf.keras.models.load_model(self.model_best_weights_save_path)
                self._load_settings()
                Utils.log_message(f"INFO: Loaded DQN model from {self.model_best_weights_save_path}")
            except (ValueError, OSError) as e:
                Utils.log_message(f"ERROR: Error loading DQN model: {e}. Starting with fresh model.")
                self.model = self._build_model()
                self.target_model = self._build_model()
                self.update_target_model()
        Utils.perform_using_retries(load)

    def save_checkpoint(self, epoch, epsilon):
        os.makedirs(os.path.join(self.model_dir, "checkpoints"), exist_ok=True)
        self.model.save(self.checkpoint_model_path)
        state = {'epoch': epoch, 'epsilon': epsilon, 'temperature': self.temperature}
        with open(self.checkpoint_state_path, "wb") as f:
            pickle.dump(state, f)
        Utils.log_message(f"INFO: Saved DQN checkpoint to {self.checkpoint_model_path}, {self.checkpoint_state_path}")

    def load_checkpoint(self):
        def load():
            if os.path.exists(self.checkpoint_model_path) and os.path.exists(self.checkpoint_state_path):
                try:
                    self.model = tf.keras.models.load_model(self.checkpoint_model_path)
                    self.target_model = tf.keras.models.load_model(self.checkpoint_model_path)
                    self.update_target_model()
                    with open(self.checkpoint_state_path, "rb") as f:
                        state = pickle.load(f)
                    epoch = state['epoch']
                    self.epsilon = state['epsilon']
                    self.temperature = state.get('temperature', 1.0)
                    Utils.log_message(f"INFO: Loaded DQN checkpoint from epoch {epoch}")
                    return epoch, self.epsilon
                except (ValueError, OSError, pickle.UnpicklingError) as e:
                    Utils.log_message(f"ERROR: Error loading DQN checkpoint: {e}. Starting with fresh model.")
                    self.model = self._build_model()
                    self.target_model = self._build_model()
                    self.update_target_model()
                    return None
            return None
        return Utils.perform_using_retries(load)

    def clear_checkpoints(self):
        checkpoint_files = [self.checkpoint_model_path, self.checkpoint_state_path]
        for file in checkpoint_files:
            if os.path.exists(file):
                os.remove(file)
                Utils.log_message(f"INFO: Removed DQN checkpoint file: {file}")
        checkpoint_dir = os.path.join(self.model_dir, "checkpoints")
        if os.path.exists(checkpoint_dir) and not os.listdir(checkpoint_dir):
            os.rmdir(checkpoint_dir)
            Utils.log_message(f"INFO: Removed empty checkpoints directory")

    def train(self, data, max_trades_per_epoch, max_fee_per_epoch):
        states = self.preprocess_data(data)
        if not self._validate_training_data(states, data):
            return [], {}

        self._apply_best_exploration_params(data)
        self.reset_portfolio()

        return self._run_training_loop(states, data, max_trades_per_epoch, max_fee_per_epoch)

    def _apply_best_exploration_params(self, data):
        params = self._tune_exploration_parameters(data)
        self.epsilon = params['epsilon']
        self.epsilon_decay = params['epsilon_decay']
        self.temperature = params['temperature']
        self.temperature_decay = params['temperature_decay']

    def _validate_training_data(self, states, data):
        if len(states) == 0:
            Utils.log_message("ERROR: No valid preprocessed data available for training")
            return False
        if len(data) < self.lookback + 1:
            Utils.log_message(f"ERROR: Insufficient data: {len(data)} rows, need at least {self.lookback + 1}")
            return False
        return True

    def _run_training_loop(self, states, data, max_trades, max_fee):
        portfolio_values = []
        total_reward = 0.0
        epoch_returns = []
        action_counts = {'Hold': 0, 'Buy': 0, 'Sell': 0}

        for t in range(self.lookback, len(data) - 1):
            if t >= len(states) + self.lookback:
                Utils.log_message(f"ERROR: Index t={t} exceeds preprocessed data length {len(states)}")
                break

            state = states[t - self.lookback]
            action = self.act(state)
            price = data['Close'].iloc[t]
            prev_value = self.simulate_trade(action, price, data, t, max_trades, max_fee)

            next_price = data['Close'].iloc[t + 1]
            curr_value = self.simulate_trade(None, next_price, data, t+1, max_trades, max_fee)

            reward, _ = Calculations.calculate_reward(prev_value, curr_value, action, self.fee_history, self.portfolio_value_history, data, t, self.initial_cash, epoch_returns)
            done = (t == len(data) - 2)

            next_state = states[t - self.lookback + 1] if t < len(states) + self.lookback - 1 else states[-1]
            self.remember(state, action, reward, next_state, done)
            self.replay()

            total_reward += float(np.squeeze(reward))
            portfolio_values.append(curr_value)
            action_counts[['Hold', 'Buy', 'Sell'][action]] += 1

            if done:
                self.update_target_model()
                Utils.log_message(f"INFO: DQN training completed - Total Reward: {total_reward:.4f}, Action Counts: {action_counts}")

        metrics = {
            "total_reward": total_reward,
            "num_trades": sum(action_counts.values()),
            "final_portfolio_value": portfolio_values[-1] if portfolio_values else 0.0,
            "avg_return": np.mean(epoch_returns) if epoch_returns else 0.0
        }
        return portfolio_values, metrics

    def _tune_exploration_parameters(self, data, validation_split=0.3):
        param_grid = {
            'epsilon': [0.5, 0.7, 1.0],
            'epsilon_decay': [0.995, 0.999],
            'temperature': [0.5, 1.0],
            'temperature_decay': [0.995, 0.999]
        }
        param_combinations = list(product(*param_grid.values()))

        states = self.preprocess_data(data)
        if len(states) == 0:
            Utils.log_message("ERROR: No valid preprocessed data for exploration tuning")
            return self._get_current_exploration_params()

        train_size = int(len(states) * (1 - validation_split))
        train_states, val_states = states[:train_size], states[train_size:]
        train_data, val_data = data.iloc[:train_size + self.lookback], data.iloc[train_size + self.lookback:]

        best_score, best_params = -float('inf'), None
        for params in param_combinations:
            params_dict = dict(zip(param_grid.keys(), params))
            self._apply_exploration_params(params_dict)
            self._simulate_training(train_states, train_data)
            sharpe = self._evaluate_validation(val_states, val_data, train_size)

            if sharpe > best_score:
                best_score, best_params = sharpe, params_dict
            Utils.log_message(f"INFO: Exploration tuning: Params={params_dict}, Sharpe={sharpe:.4f}")

        if best_params:
            Utils.log_message(f"INFO: Best exploration parameters: {best_params}, Sharpe={best_score:.4f}")
            return best_params
        else:
            Utils.log_message("WARNING: Exploration tuning failed, using default parameters")
            return self._get_current_exploration_params()

    def simulate_trade(self, action, price, max_trades_per_epoch, max_fee_per_epoch, data=None, index=None, portfolio_tracker=None):
        if action is None:
            return self.get_portfolio_value(price)
        
        if max_trades_per_epoch > 0 and self.trade_count >= max_trades_per_epoch:
            Utils.log_message(f"INFO: Trade skipped: Maximum trades per epoch reached ({self.trade_count}/{max_trades_per_epoch})")
            return self.get_portfolio_value(price)
        if max_fee_per_epoch > 0 and self.cumulative_fees >= max_fee_per_epoch:
            Utils.log_message(f"INFO: Trade skipped: Maximum fee per epoch reached ({self.cumulative_fees:.2f}/{max_fee_per_epoch})")
            return self.get_portfolio_value(price)
        
        current_value = self.get_portfolio_value(price)
        if current_value < self.initial_cash * 0.5:
            Utils.log_message(f"INFO: Trade skipped: Portfolio value below 50% of initial cash")
            return current_value
        
        atr = 0.02 * price
        if data is not None and index is not None and 'ATR' in data.columns:
            atr_value = data['ATR'].iloc[index]
            atr = float(atr_value) if pd.notna(atr_value) else atr
        
        min_stop_loss = price * 0.01
        stop_loss_distance = max(atr * self.atr_multiplier, min_stop_loss)
        max_risk_amount = self.cash * self.risk_per_trade
        buy_fee = self.trade_fee
        sell_fee = self.trade_fee
        fee_adjusted_price = price * (1 + buy_fee)
        future_selling_fee = price * sell_fee
        max_shares = np.floor(max_risk_amount / (stop_loss_distance + future_selling_fee))
        max_shares_by_cash = np.floor((self.cash * 0.5) / fee_adjusted_price)
        max_shares_by_portfolio = np.floor((current_value * 0.25) / fee_adjusted_price)
        shares_to_buy = max(min(max_shares, max_shares_by_cash, max_shares_by_portfolio), 1)
        
        Utils.log_message(f"INFO: Trade calc: price={price:.2f}, atr={atr:.4f}, stop_loss_distance={stop_loss_distance:.2f}, max_risk_amount={max_risk_amount:.2f}, max_shares={max_shares}, max_shares_by_cash={max_shares_by_cash}, max_shares_by_portfolio={max_shares_by_portfolio}, shares_to_buy={shares_to_buy}")
        
        if portfolio_tracker is None:
            if action == 1:
                cost_before_fees = shares_to_buy * price
                fee_amount = cost_before_fees * buy_fee
                total_cost = cost_before_fees + fee_amount
                if self.cash >= total_cost and shares_to_buy > 0:
                    if fee_amount < 0:
                        Utils.log_message(f"ERROR: Negative fee amount: {fee_amount}")
                        return current_value
                    self.cumulative_fees += fee_amount
                    self.fee_history.append(fee_amount)
                    self.cash -= total_cost
                    self.shares += shares_to_buy
                    self.trade_count += 1
                    Utils.log_message(f"INFO: Buy executed: {shares_to_buy} shares at {price}, fee: {fee_amount:.2f}, cumulative_fees: {self.cumulative_fees:.2f}")
                else:
                    Utils.log_message(f"INFO: Trade skipped: Insufficient cash ({self.cash:.2f} < {total_cost:.2f}) or invalid shares_to_buy")
            elif action == 2 and self.shares > 0:
                revenue_before_fees = self.shares * price
                fee_amount = revenue_before_fees * sell_fee
                if fee_amount < 0:
                    Utils.log_message(f"ERROR: Negative fee amount: {fee_amount}")
                    return current_value
                net_revenue = revenue_before_fees - fee_amount
                self.cumulative_fees += fee_amount
                self.fee_history.append(fee_amount)
                self.cash += net_revenue
                self.shares = 0
                self.trade_count += 1
                Utils.log_message(f"INFO: Sell executed: {self.shares} shares at {price}, fee: {fee_amount:.2f}, cumulative_fees: {self.cumulative_fees:.2f}")
            else:
                Utils.log_message(f"INFO: Trade skipped: Action={action}, Cash={self.cash}, Shares={self.shares}, Shares_to_buy={shares_to_buy}")
        else:
            if action == 1:
                cost_before_fees = shares_to_buy * price
                total_cost = cost_before_fees * (1 + buy_fee)
                if portfolio_tracker.cash >= total_cost and shares_to_buy > 0:
                    try:
                        success, transaction_id = portfolio_tracker.buy(self.ticker, shares_to_buy, price, fee_percentage=buy_fee)
                        if success:
                            fee_amount = portfolio_tracker.transactions[-1]['fee']
                            if fee_amount < 0:
                                Utils.log_message(f"ERROR: Negative fee amount: {fee_amount}")
                                return current_value
                            self.cumulative_fees += fee_amount
                            self.fee_history.append(fee_amount)
                            self.cash = portfolio_tracker.cash
                            self.shares = portfolio_tracker.holdings.get(self.ticker, {'quantity': 0})['quantity']
                            self.trade_count += 1
                            Utils.log_message(f"INFO: Buy executed via PortfolioTracker: {shares_to_buy} shares at {price}, fee: {fee_amount:.2f}, cumulative_fees: {self.cumulative_fees:.2f}, total_fees: {portfolio_tracker.total_fees:.2f}, transaction_id: {transaction_id}")
                        else:
                            Utils.log_message(f"INFO: Buy failed: {transaction_id}")
                    except Exception as e:
                        Utils.log_message(f"ERROR: PortfolioTracker buy failed: {e}")
                        return current_value
                else:
                    Utils.log_message(f"INFO: Trade skipped: Insufficient cash ({portfolio_tracker.cash:.2f} < {total_cost:.2f}) or invalid shares_to_buy")
            elif action == 2 and self.shares > 0:
                try:
                    success, transaction_id = portfolio_tracker.sell(self.ticker, self.shares, price, fee_percentage=sell_fee)
                    if success:
                        fee_amount = portfolio_tracker.transactions[-1]['fee']
                        if fee_amount < 0:
                            Utils.log_message(f"ERROR: Negative fee amount: {fee_amount}")
                            return current_value
                        self.cumulative_fees += fee_amount
                        self.fee_history.append(fee_amount)
                        self.cash = portfolio_tracker.cash
                        self.shares = portfolio_tracker.holdings.get(self.ticker, {'quantity': 0})['quantity']
                        self.trade_count += 1
                        Utils.log_message(f"INFO: Sell executed via PortfolioTracker: {self.shares} shares at {price}, fee: {fee_amount:.2f}, cumulative_fees: {self.cumulative_fees:.2f}, total_fees: {portfolio_tracker.total_fees:.2f}, transaction_id: {transaction_id}")
                    else:
                        Utils.log_message(f"INFO: Sell failed: {transaction_id}")
                except Exception as e:
                    Utils.log_message(f"ERROR: PortfolioTracker sell failed: {e}")
                    return current_value
            else:
                Utils.log_message(f"INFO: Trade skipped: Action={action}, Cash={portfolio_tracker.cash}, Shares={self.shares}, Shares_to_buy={shares_to_buy}")

        current_value = self.get_portfolio_value(price)
        self.portfolio_value_history.append(current_value)
        return current_value

    def _evaluate_validation(self, states, data, offset):
        self.reset_portfolio()
        values = []
        for t in range(self.lookback, len(data) - 1):
            idx = t - self.lookback - offset
            if idx >= len(states):
                break
            state = states[idx]
            action = self.act(state)
            price = data['Close'].iloc[t]
            value = self.simulate_trade(action, price, data, t, 0, 0)
            values.append(value)

        if len(values) < 2:
            return -np.inf
        returns = np.diff(values) / values[:-1]
        return float(np.sqrt(252) * np.mean(returns) / (np.std(returns) + 1e-9))

    def _get_current_exploration_params(self):
        return {
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'temperature': self.temperature,
            'temperature_decay': self.temperature_decay
        }

    def _apply_exploration_params(self, params):
        self.epsilon = params['epsilon']
        self.epsilon_decay = params['epsilon_decay']
        self.temperature = params['temperature']
        self.temperature_decay = params['temperature_decay']