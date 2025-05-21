import os
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import sys
sys.dont_write_bytecode = True

import pickle
import json
import numpy as np
import pandas as pd
import datetime
import random
from rf_agent import RFAgent
from dqn_agent import DQNAgent
from portfolio_tracker import PortfolioTracker
from utils import Utils, FEATURE_COLUMNS
from calculations import Calculations

class EnsembleAgent:
    def __init__(self, ticker, training_start_date, training_end_date, model_dir, lookback, initial_cash, trade_fee, risk_per_trade, 
                max_trades_per_epoch, max_fee_per_epoch, atr_multiplier, atr_period, atr_smoothing, use_smote, dqn_weight_scale,
                capital_type, reference_capital, capital_percentage, epochs, confirmation_steps, expected_feature_columns=FEATURE_COLUMNS):

        if max_fee_per_epoch > 0 and max_fee_per_epoch < trade_fee * initial_cash:
            Utils.log_message(f"WARNING: max_fee_per_epoch (€{max_fee_per_epoch:.2f}) may be too low for initial_cash (€{initial_cash:.2f}) and trade_fee ({trade_fee*100:.2f}%)")

        self.ticker = ticker
        self.training_start_date = pd.Timestamp(training_start_date).date()
        self.training_end_date = pd.Timestamp(training_end_date).date()
        self.lookback = lookback
        self.trade_fee = trade_fee
        self.initial_cash = initial_cash
        self.risk_per_trade = risk_per_trade
        self.max_trades_per_epoch = max_trades_per_epoch
        self.max_fee_per_epoch = max_fee_per_epoch
        self.atr_multiplier = atr_multiplier
        self.atr_period = atr_period
        self.atr_smoothing = atr_smoothing
        self.use_smote = use_smote
        self.dqn_weight_scale = dqn_weight_scale
        self.capital_type = capital_type
        self.reference_capital = reference_capital
        self.capital_percentage = capital_percentage
        self.epochs = epochs
        self.confirmation_steps = confirmation_steps
        self.expected_feature_columns = expected_feature_columns
        if not model_dir or not isinstance(model_dir, str):
            raise ValueError("model_dir must be a non-empty string")
        if os.path.basename(model_dir) == ticker:
            self.model_dir = model_dir
        else:
            self.model_dir = os.path.join(model_dir, ticker)
        os.makedirs(self.model_dir, exist_ok=True)
        self.settings_save_path = os.path.join(self.model_dir, f"{ticker}_ensemble_settings.json")
        self.metrics_save_path = os.path.join(self.model_dir, f"{ticker}_ensemble_metrics.json")
        self.checkpoint_portfolio_path = os.path.join(self.model_dir, "checkpoints", f"{ticker}_checkpoint_portfolio.pkl")

        self.training_data = self.get_data(self.training_start_date, self.training_end_date)
        if self.training_data.empty:
            raise ValueError(f"No training data could be obtained for ticker {self.ticker} from {self.training_start_date} to {self.training_end_date}")
        
        self.dqn_agent = DQNAgent(
            ticker=self.ticker, training_start_date=self.training_start_date, training_end_date=self.training_end_date, model_dir=self.model_dir,
            lookback=self.lookback, initial_cash=self.initial_cash, trade_fee=self.trade_fee, risk_per_trade=self.risk_per_trade,
            atr_multiplier=self.atr_multiplier, atr_period=self.atr_period, atr_smoothing=self.atr_smoothing,
            expected_feature_columns=self.expected_feature_columns
        )
        self.rf_agent = RFAgent(
            ticker=self.ticker, training_start_date=self.training_start_date, training_end_date=self.training_end_date, model_dir=self.model_dir,
            lookback=self.lookback, trade_fee=self.trade_fee, use_smote=self.use_smote, expected_feature_columns=self.expected_feature_columns
        )
        
        self.best_sharpe = -float("inf")
        self.cumulative_fees = 0
        self.fee_history = []
        self.reset_portfolio()
        self.sharpe_history = {'dqn': [], 'rf': []}
        self.cached_sharpe = {'dqn': 0.0, 'rf': 0.0, 'timestamp': None}
        self.cache_duration = 60.0
        self.load_model()
        self.trading_data=None

    def has_enough_training_data(self):
        return (self.training_data is not None and not self.training_data.empty and len(self.training_data) >= self.lookback + self.dqn_agent.batch_size + 1)

    def get_data(self, start_date, end_date):
        try:
            df = Utils.fetch_data(self.ticker, start_date, end_date)
            if df.empty:
                raise ValueError(f"No training_data fetched for {self.ticker} from {self.training_start_date} to {self.training_end_date}")

            processed_df = Utils.preprocess_data(df, volatility_period=self.atr_period, atr_smoothing=self.atr_smoothing) if not set(FEATURE_COLUMNS).issubset(df.columns) else df
            if processed_df.empty:
                raise ValueError(f"ERROR: Failed to preprocess training_data for {self.ticker}")

            return processed_df
        except Exception as e:
            Utils.log_message(f"ERROR: Error fetching training_data for {self.ticker}: {e}")
            return pd.DataFrame()

    def reset_portfolio(self):
        self.cash = self.initial_cash
        self.shares = 0
        self.portfolio_value_history = []
        self.returns = []
        # self.cumulative_fees = 0
        self.epoch_cumulative_fees = 0
        self.epoch_metrics = []
        self.portfolio_values = []
        self.epoch_trade_count = 0
        self.action_counts = {'Hold': 0, 'Buy': 0, 'Sell': 0}

    def _save_metrics(self):
        metrics = {
            "best_sharpe": self.best_sharpe,
            "last_drawdown": self.evaluate_performance()[1],
            "total_trades": sum(self.action_counts.values()),
            "dqn_temperature": self.dqn_agent.temperature,
            "rf_best_score": self.rf_agent.best_score,
            "rf_best_params": self.rf_agent.best_params,
            "rf_n_pca_components": self.rf_agent.n_pca_components,
            "last_updated": datetime.datetime.now().isoformat()
        }
        try:
            os.makedirs(self.model_dir, exist_ok=True)
            with open(self.metrics_save_path, "w") as f:
                json.dump(metrics, f, indent=4)
            Utils.log_message(f"INFO: Saved metrics to {self.metrics_save_path}")
        except Exception as e:
            Utils.log_message(f"ERROR: Failed to save metrics: {e}")

    def _save_settings(self, settings=None):
        settings = {
            'ticker': settings['ticker'] if settings else self.ticker,
            'training_start_date': pd.Timestamp(settings['training_start_date'] if settings else self.training_start_date).isoformat(),
            'training_end_date': pd.Timestamp(settings['training_end_date'] if settings else self.training_end_date).isoformat(),
            'lookback': settings['lookback'] if settings else self.lookback,
            'initial_cash': settings['initial_cash'] if settings else self.initial_cash,
            'trade_fee': settings['trade_fee'] if settings else self.trade_fee,
            'risk_per_trade': settings['risk_per_trade'] if settings else self.risk_per_trade,
            'max_trades_per_epoch': settings['max_trades_per_epoch'] if settings else self.max_trades_per_epoch,
            'max_fee_per_epoch': settings['max_fee_per_epoch'] if settings else self.max_fee_per_epoch,
            'atr_multiplier': settings['atr_multiplier'] if settings else self.atr_multiplier,
            'atr_period': settings['atr_period'] if settings else self.atr_period,
            'atr_smoothing': settings['atr_smoothing'] if settings else self.atr_smoothing,
            'use_smote': settings['use_smote'] if settings else self.use_smote,
            'dqn_weight_scale': settings['dqn_weight_scale'] if settings else self.dqn_weight_scale,
            'capital_type': settings['capital_type'] if settings else self.capital_type,
            'reference_capital': settings['reference_capital'] if settings else self.reference_capital,
            'capital_percentage': settings['capital_percentage'] if settings else self.capital_percentage,
            'epochs': settings['epochs'] if settings else self.epochs,
            'confirmation_steps': settings['confirmation_steps'] if settings else self.confirmation_steps
        }
        try:
            os.makedirs(self.model_dir, exist_ok=True)
            with open(self.settings_save_path, 'w') as f:
                json.dump(settings, f, indent=4)
            Utils.log_message(f"INFO: Saved ensemble settings to {self.settings_save_path}")
        except Exception as e:
            Utils.log_message(f"ERROR: Failed to save ensemble settings: {e}")

    def _load_settings(self):
        if os.path.exists(self.settings_save_path):
            with open(self.settings_save_path, 'r') as f:
                settings = json.load(f)
            self.ticker = settings.get('ticker', self.ticker)
            self.training_start_date = pd.Timestamp(settings['training_start_date']).date()
            self.training_end_date = pd.Timestamp(settings['training_end_date']).date()
            self.initial_cash = settings.get('initial_cash', self.initial_cash)
            self.trade_fee = settings.get('trade_fee', self.trade_fee)
            self.risk_per_trade = settings.get('risk_per_trade', self.risk_per_trade)
            self.max_trades_per_epoch = settings.get('max_trades_per_epoch', self.max_trades_per_epoch)
            self.max_fee_per_epoch = settings.get('max_fee_per_epoch', self.max_fee_per_epoch)
            self.risk_per_trade = settings.get('risk_per_trade', self.risk_per_trade)
            self.atr_multiplier = settings.get('atr_multiplier', self.atr_multiplier)
            self.atr_period = settings.get('atr_period', self.atr_period)
            self.atr_smoothing = settings.get('atr_smoothing', self.atr_smoothing)
            self.use_smote = settings.get('use_smote', self.use_smote)
            self.dqn_weight_scale = settings.get('dqn_weight_scale', self.dqn_weight_scale)
            self.capital_type = settings.get('capital_type', self.capital_type)
            self.reference_capital = settings.get('reference_capital', self.reference_capital)
            self.capital_percentage = settings.get('capital_percentage', self.capital_percentage)
            self.epochs = settings.get('epochs', self.epochs)
            self.confirmation_steps = settings.get('confirmation_steps', self.confirmation_steps)
            Utils.log_message(f"INFO: Loaded ensemble settings from {self.settings_save_path}: {settings}")
            return settings
        else:
            Utils.log_message(f"INFO: No ensemble settings file found at {self.settings_save_path}")
        return None

    def update_settings(self, new_settings):
        self._save_settings(new_settings)

    def load_model(self):
        saved_settings = self._load_settings()
        if saved_settings:
            self.dqn_agent = DQNAgent(
                model_dir=self.model_dir,
                initial_cash=saved_settings['initial_cash'],
                lookback=saved_settings['lookback'],
                ticker=saved_settings['ticker'],
                training_start_date=self.training_start_date,
                training_end_date=self.training_end_date,
                trade_fee=saved_settings['trade_fee'],
                risk_per_trade=saved_settings['risk_per_trade'],
                atr_multiplier=saved_settings['atr_multiplier'],
                atr_period=saved_settings['atr_period'],
                atr_smoothing=saved_settings['atr_smoothing']
            )
            self.rf_agent = RFAgent(
                model_dir=self.model_dir,
                lookback=saved_settings['lookback'],
                ticker=saved_settings['ticker'],
                training_start_date=self.training_start_date,
                training_end_date=self.training_end_date,
                trade_fee=saved_settings['trade_fee'],
                use_smote=saved_settings['use_smote']
            )
        self.dqn_agent.load_model()
        self.rf_agent.load_model()
        
        if os.path.exists(self.metrics_save_path):
            try:
                with open(self.metrics_save_path, "r") as f:
                    metrics = json.load(f)
                self.best_sharpe = float(metrics.get("best_sharpe", -float("inf")))
                Utils.log_message(f"INFO: Loaded best Sharpe score from JSON: {self.best_sharpe:.4f}")
            except (json.JSONDecodeError, ValueError) as e:
                Utils.log_message(f"ERROR: Failed to load metrics from {self.metrics_save_path}: {e}")
        
        old_text_path = os.path.join(self.model_dir, f"{self.ticker}_ensemble_best_sharpe_score.txt")
        if os.path.exists(old_text_path):
            try:
                with open(old_text_path, "r") as f:
                    self.best_sharpe = float(f.read())
                Utils.log_message(f"INFO: Loaded best Sharpe score from legacy text file: {self.best_sharpe:.4f}")
                self._save_metrics()
                os.remove(old_text_path)
                Utils.log_message(f"INFO: Migrated best_sharpe to JSON and removed {old_text_path}")
            except (ValueError, OSError) as e:
                Utils.log_message(f"ERROR: Failed to load or migrate from {old_text_path}: {e}")

    def get_configuration_settings(self):
        return {
            'ticker': self.ticker,
            'training_start_date': self.training_start_date,
            'training_end_date': self.training_end_date,
            'model_dir': self.model_dir,
            'initial_cash': self.initial_cash,
            'trade_fee': self.trade_fee,
            'lookback': self.lookback,
            'risk_per_trade': self.risk_per_trade,
            'max_trades_per_epoch': self.max_trades_per_epoch,
            'max_fee_per_epoch': self.max_fee_per_epoch,
            'dqn_weight_scale': self.dqn_weight_scale,
            'atr_multiplier': self.atr_multiplier,
            'atr_period': self.atr_period,
            'atr_smoothing': self.atr_smoothing,
            'use_smote': self.use_smote,
            'capital_type': self.capital_type,
            'reference_capital': self.reference_capital,
            'capital_percentage': self.capital_percentage,
            'epochs': self.epochs,
            'confirmation_steps': self.confirmation_steps,
        }

    def has_pretrained_model(self):
        return (os.path.exists(self.dqn_agent.model_best_weights_save_path) and 
                os.path.exists(self.rf_agent.rf_best_model_save_path) and 
                os.path.exists(self.rf_agent.kmeans_save_path))

    def save_checkpoint(self, epoch, epsilon, epoch_metrics=None, portfolio_values=None, settings=None):
        if settings is None:
            Utils.log_message(f"WARNING: Checkpoint saved with settings=None.")
            settings_copy = {}
        else:
            settings_copy = settings.copy()
            if settings_copy.get('training_start_date'):
                settings_copy['training_start_date'] = pd.Timestamp(settings_copy['training_start_date']).isoformat()
            if settings_copy.get('training_end_date'):
                settings_copy['training_end_date'] = pd.Timestamp(settings_copy['training_end_date']).isoformat()

        self.dqn_agent.save_checkpoint(epoch, epsilon)
        self.rf_agent.save_checkpoint()
        portfolio_state = {
            'cash': self.cash,
            'shares': self.shares,
            'portfolio_value_history': self.portfolio_value_history,
            'returns': self.returns,
            'cumulative_fees': self.cumulative_fees,
            'fee_history': self.fee_history,
            'epoch_metrics': epoch_metrics if epoch_metrics is not None else self.epoch_metrics,
            'portfolio_values': portfolio_values if portfolio_values is not None else self.portfolio_values,
            'epoch_trade_count': self.epoch_trade_count,
            'settings': settings_copy,
            'action_counts': self.action_counts,
            'sharpe_history': self.sharpe_history
        }
        os.makedirs(os.path.join(self.model_dir, "checkpoints"), exist_ok=True)
        with open(self.checkpoint_portfolio_path, "wb") as f:
            pickle.dump(portfolio_state, f)
        Utils.log_message(f"INFO: Saved portfolio checkpoint to {self.checkpoint_portfolio_path}")

    def load_checkpoint(self, settings=None):
        dqn_checkpoint = self.dqn_agent.load_checkpoint()
        rf_checkpoint = self.rf_agent.load_checkpoint()
        
        if (dqn_checkpoint and rf_checkpoint and os.path.exists(self.checkpoint_portfolio_path)):
            try:
                epoch, epsilon = dqn_checkpoint
                with open(self.checkpoint_portfolio_path, "rb") as f:
                    portfolio_state = pickle.load(f)
                self.cash = portfolio_state['cash']
                self.shares = portfolio_state['shares']
                self.portfolio_value_history = portfolio_state['portfolio_value_history']
                self.returns = portfolio_state['returns']
                self.cumulative_fees = portfolio_state['cumulative_fees']
                self.fee_history = portfolio_state['fee_history']
                self.epoch_metrics = portfolio_state.get('epoch_metrics', [])
                self.portfolio_values = portfolio_state.get('portfolio_values', [])
                self.epoch_trade_count = portfolio_state.get('epoch_trade_count', 0)
                self.action_counts = portfolio_state.get('action_counts', {'Hold': 0, 'Buy': 0, 'Sell': 0})
                self.sharpe_history = portfolio_state.get('sharpe_history', {'dqn': [], 'rf': []})
                loaded_settings = portfolio_state.get('settings', None)
                self.atr_multiplier = loaded_settings.get('atr_multiplier', self.atr_multiplier)
                self.atr_period = loaded_settings.get('atr_period', self.atr_period)
                self.atr_smoothing = loaded_settings.get('atr_smoothing', self.atr_smoothing)
                self.use_smote = loaded_settings.get('use_smote', self.use_smote)
                self.dqn_weight_scale = loaded_settings.get('dqn_weight_scale', self.dqn_weight_scale)
                if loaded_settings:
                    if loaded_settings.get('training_start_date'):
                        loaded_settings['training_start_date'] = pd.Timestamp(loaded_settings['training_start_date']).date()
                    if loaded_settings.get('training_end_date'):
                        loaded_settings['training_end_date'] = pd.Timestamp(loaded_settings['training_end_date']).date()
                    self.training_start_date = loaded_settings['training_start_date']
                    self.training_end_date = loaded_settings['training_end_date']
                    Utils.log_message(f"DEBUG: Loaded checkpoint settings: {loaded_settings}")
                else:
                    Utils.log_message(f"WARNING: No settings found in checkpoint, using input settings")
                    loaded_settings = settings
                Utils.log_message(f"INFO: Loaded ensemble checkpoint from epoch {epoch}, epoch_trade_count: {self.epoch_trade_count}")
                return epoch, epsilon, self.epoch_metrics, self.portfolio_values, loaded_settings
            except (ValueError, OSError, pickle.UnpicklingError) as e:
                Utils.log_message(f"ERROR: Error loading ensemble checkpoint: {e}. Starting with fresh portfolio.")
                self.reset_portfolio()
                return None
        self.epoch_trade_count = 0
        self.action_counts = {'Hold': 0, 'Buy': 0, 'Sell': 0}
        self.sharpe_history = {'dqn': [], 'rf': []}
        return None

    def clear_checkpoints(self):
        self.dqn_agent.clear_checkpoints()
        self.rf_agent.clear_checkpoints()
        if os.path.exists(self.checkpoint_portfolio_path):
            os.remove(self.checkpoint_portfolio_path)
            Utils.log_message(f"INFO: Removed portfolio checkpoint file: {self.checkpoint_portfolio_path}")
        checkpoint_dir = os.path.join(self.model_dir, "checkpoints")
        if os.path.exists(checkpoint_dir) and not os.listdir(checkpoint_dir):
            os.rmdir(checkpoint_dir)
            Utils.log_message(f"INFO: Removed empty checkpoints directory")


    def train(self, current_epoch, max_epochs, portfolio_values=None, progress_callback=None):
        state_training_data = self.training_data[self.expected_feature_columns].dropna()
        if state_training_data.empty:
            Utils.log_message(f"ERROR: No valid state training_data available for fitting state scaler.")
            return [], {}, {}
        
        self.dqn_agent.state_scaler.fit(state_training_data)
        Utils.log_message(f"DEBUG: State scaler fitted with {self.dqn_agent.state_scaler.n_features_in_} features, mean: {self.dqn_agent.state_scaler.mean_[:5]}")

        training_data = self.dqn_agent.preprocess_data(self.training_data)
        if len(training_data) == 0:
            Utils.log_message(f"ERROR: No valid preprocessed training_data available for training.")
            return [], {}, {}

        if len(self.training_data) < self.lookback + 1:
            Utils.log_message(f"ERROR: Insufficient training_data: {len(self.training_data)} rows, need at least {self.lookback + 1}")
            return [], {}, {}
        
        Utils.log_message(f"INFO: Training epoch {current_epoch + 1} with training_data length: {len(self.training_data)}, preprocessed states: {len(training_data)}")
        
        self.portfolio_values = portfolio_values if portfolio_values is not None else []

        if current_epoch == 0: # Reset's the portfolio only at the start of training
            self.reset_portfolio()
            initial_cash = self.initial_cash
            Utils.log_message(f"INFO: Portfolio reset at epoch {current_epoch + 1} with initial cash: €{initial_cash:.2f}")

        current_cash = self.dqn_agent.cash
        min_cash_threshold = self.initial_cash * self.trade_fee
        if current_cash < min_cash_threshold:
            Utils.log_message(f"WARNING: Cash (€{current_cash:.2f}) below threshold (€{min_cash_threshold:.2f}) at epoch {current_epoch + 1}. Resetting portfolio.")
            self.reset_portfolio()
            current_cash = self.initial_cash
            Utils.log_message(f"INFO: Portfolio reset with cash: €{current_cash:.2f}")
        
        self.epoch_trade_count = 0
        self.epoch_cumulative_fees = 0  # Track fees for this epoch
        self.returns = self.returns.copy() if self.returns else []
        
        if not self.rf_agent or not os.path.exists(self.rf_agent.rf_best_model_save_path) or current_epoch == 0:
            Utils.log_message(f"INFO: Training RF Agent independently")
            self.rf_agent.train(self.training_data, progress_callback=progress_callback)

        if not self.rf_agent:
            Utils.log_message(f"ERROR: RFAgent model not loaded after training. Attempting to train with diverse labels.")
            try:
                state_data_subset = state_training_data.iloc[-self.lookback:].values
                labels = self.rf_agent._create_labels(self.training_data.iloc[-self.lookback-1:])
                if len(np.unique(labels)) < 2:
                    Utils.log_message(f"ERROR: Default RF model training failed: Only one label in data.")
                    raise ValueError("Insufficient label diversity for default RF model")
                from sklearn.ensemble import RandomForestClassifier
                self.rf_agent = RandomForestClassifier(
                    n_estimators=100, max_depth=5, min_samples_split=5, min_samples_leaf=5, random_state=42
                )
                self.rf_agent.fit(state_data_subset, labels)
                Utils.log_message(f"INFO: Initialized default RF model with {len(state_data_subset)} samples.")
            except Exception as e:
                Utils.log_message(f"ERROR: Failed to initialize default RF model: {e}. Skipping epoch.")
                return [], {}, {}

        if progress_callback:
            progress_callback(
                1.0,
                params=self.rf_agent.best_params if hasattr(self.rf_agent, 'best_params') else {},
                mean_cv_score=self.rf_agent.best_score if hasattr(self.rf_agent, 'best_score') else 0.0,
                best_params=self.rf_agent.best_params if hasattr(self.rf_agent, 'best_params') else {},
                best_score=self.rf_agent.best_score if hasattr(self.rf_agent, 'best_score') else 0.0,
                training_phase="rf",
                pre_smote_label_distribution=self.rf_agent.pre_smote_label_distribution if hasattr(self.rf_agent, 'pre_smote_label_distribution') else {},
                post_smote_label_distribution=self.rf_agent.post_smote_label_distribution if hasattr(self.rf_agent, 'post_smote_label_distribution') else {}
            )
            progress_callback(
                (current_epoch) / max_epochs,
                training_phase="dqn",
                override_status_message=f"Processing epoch {current_epoch + 1} of {max_epochs}..."
            )

        portfolio_values = []
        total_reward = 0
        num_trades = 0
        epoch_returns = []
        action_counts = {'Hold': 0, 'Buy': 0, 'Sell': 0}
        
        # Initialize PortfolioTracker for this epoch
        portfolio_tracker = PortfolioTracker(self.initial_cash)
        
        for t in range(self.lookback, len(self.training_data) - 1):
            if t >= len(training_data) + self.lookback:
                Utils.log_message(f"ERROR: Index t={t} exceeds preprocessed training_data length {len(training_data)}")
                break
            state = training_data[t - self.lookback]
            action = self.act(state)
            price = self.training_data['Close'].iloc[t]
            prev_value = self.dqn_agent.simulate_trade(action, price, data=self.training_data, index=t, 
                                            max_trades_per_epoch=self.max_trades_per_epoch, 
                                            max_fee_per_epoch=self.max_fee_per_epoch,
                                            portfolio_tracker=portfolio_tracker)
            next_price = self.training_data['Close'].iloc[t + 1]
            curr_value = self.dqn_agent.simulate_trade(None, next_price, max_trades_per_epoch=self.max_trades_per_epoch,
                                                    max_fee_per_epoch=self.max_fee_per_epoch, data=self.training_data, index=t + 1,
                                                    portfolio_tracker=portfolio_tracker)
            reward, ret = Calculations.calculate_reward(prev_value, curr_value, action, self.dqn_agent.fee_history, self.portfolio_value_history, self.training_data, t, self.initial_cash, epoch_returns)
            done = (t == len(self.training_data) - 2)
            next_state = training_data[t - self.lookback + 1] if t < len(training_data) + self.lookback - 1 else training_data[-1]
            self.dqn_agent.remember(state, action, reward, next_state, done)
            self.dqn_agent.replay()
            total_reward += float(reward)
            portfolio_values.append(curr_value)
            action_name = ['Hold', 'Buy', 'Sell'][action]
            action_counts[action_name] += 1
            self.action_counts[action_name] += 1
            if action in [1, 2] and self.dqn_agent.fee_history and self.dqn_agent.fee_history[-1] > 0:
                num_trades += 1
                self.epoch_cumulative_fees += self.dqn_agent.fee_history[-1]
                self.cumulative_fees += self.dqn_agent.fee_history[-1]
            if done:
                self.dqn_agent.update_target_model()
                Utils.log_message(f"INFO: Ensemble training completed - Total Reward: {total_reward:.4f}, Trades: {num_trades}, Action Counts: {action_counts}")

        self.portfolio_value_history = portfolio_values if current_epoch == 0 else self.portfolio_value_history + portfolio_values
        self.returns.extend(epoch_returns)
        
        sharpe, drawdown, _, best_sharpe = self.evaluate_performance()

        dqn_sharpe = self.dqn_agent.evaluate_performance()[0]
        rf_sharpe = self.rf_agent.evaluate_performance()[0]
        self.cached_sharpe = {
            'dqn': dqn_sharpe,
            'rf': rf_sharpe,
            'timestamp': datetime.datetime.now()
        }
        self.sharpe_history['dqn'].append(dqn_sharpe)
        self.sharpe_history['rf'].append(rf_sharpe)
        if len(self.sharpe_history['dqn']) > 5:
            self.sharpe_history['dqn'].pop(0)
            self.sharpe_history['rf'].pop(0)
        
        current_cash = self.dqn_agent.cash
        if not portfolio_values:
            Utils.log_message(f"WARNING: No portfolio values generated in epoch {current_epoch + 1}. Returning default metrics.")
            return portfolio_values, {
                "total_reward": total_reward,
                "num_trades": num_trades,
                "final_portfolio_value": 0.0,
                "avg_return": np.mean(epoch_returns) if epoch_returns else 0.0,
                "current_cash": current_cash
            }, {
                "Epoch": current_epoch + 1,
                "Sharpe Ratio": f"{sharpe:.4f}",
                "Best Sharpe Ratio": f"{best_sharpe:.4f}",
                "Portfolio Value (€)": "0.00",
                "Current Cash (€)": f"{current_cash:.2f}",
                "Max Drawdown (€)": f"{drawdown:.2f}",
                "Cumulative Fees (€)": f"{portfolio_tracker.total_fees:.2f}",
                "Avg Return": f"{np.mean(epoch_returns) if epoch_returns else 0.0:.4f}",
                "Total Reward": f"{total_reward:.4f}",
                "Buy Actions": action_counts['Buy'],
                "Sell Actions": action_counts['Sell'],
                "Hold Actions": action_counts['Hold'],
                "Total Trades": num_trades
            }

        epoch_metrics_dict = {
            "Epoch": current_epoch + 1,
            "Sharpe Ratio": f"{sharpe:.4f}",
            "Best Sharpe Ratio": f"{best_sharpe:.4f}",
            "Portfolio Value (€)": f"{portfolio_values[-1]:.2f}",
            "Current Cash (€)": f"{current_cash:.2f}",
            "Max Drawdown (€)": f"{drawdown:.2f}",
            "Cumulative Fees (€)": f"{portfolio_tracker.total_fees:.2f}",
            "Epoch Cumulative Fees (€)": f"{self.epoch_cumulative_fees:.2f}",
            "Avg Return": f"{np.mean(epoch_returns) if epoch_returns else 0.0:.4f}",
            "Buy Actions": action_counts['Buy'],
            "Sell Actions": action_counts['Sell'],
            "Hold Actions": action_counts['Hold'],
            "Total Trades": num_trades,
            "Total Reward": f"{total_reward:.4f}"
        }
        
        Utils.log_message(f"INFO: Epoch {current_epoch + 1} - Sharpe Ratio: {sharpe:.4f} / {best_sharpe:.4f} - Drawdown: {drawdown:.2f} - Total Reward: {total_reward:.4f} - Trades: {num_trades} - Avg Return: {np.mean(epoch_returns) if epoch_returns else 0.0:.4f} - Epsilon: {self.dqn_agent.epsilon:.3f} - Fees: {portfolio_tracker.total_fees:.2f} - Epoch Fees: {self.epoch_cumulative_fees:.2f} - Portfolio Value: {portfolio_values[-1] if portfolio_values else 0.0:.2f} - Trade Count: {self.epoch_trade_count}/{self.max_trades_per_epoch} - Cash: {current_cash:.2f} - Buy Actions: {action_counts['Buy']} - Sell Actions: {action_counts['Sell']} - Hold Actions: {action_counts['Hold']}")
        
        if sharpe > self.best_sharpe:
            self.best_sharpe = sharpe
            self.dqn_agent.save_model()
            self.rf_agent.save_model()
            self._save_settings()
            self._save_metrics()
            Utils.log_message(f"INFO: Saved ensemble models and metrics with Sharpe Ratio: {sharpe:.4f}")
        
        return portfolio_values, {
            "total_reward": total_reward,
            "num_trades": num_trades,
            "final_portfolio_value": portfolio_values[-1] if portfolio_values else 0.0,
            "avg_return": np.mean(epoch_returns) if epoch_returns else 0.0,
            "current_cash": current_cash
        }, epoch_metrics_dict

    def evaluate_performance(self):
        values = np.array(self.portfolio_value_history).flatten()
        if len(values) < 2:
            return 0.0, 0.0, 0.0, self.best_sharpe
        returns = np.diff(values) / values[:-1]
        sharpe_ratio = np.sqrt(252) * np.mean(returns) / (np.std(returns) + 1e-9)
        drawdown = np.max(np.maximum.accumulate(values) - values)
        return sharpe_ratio, drawdown, self.cumulative_fees, self.best_sharpe

    def preprocess_training_data(self, training_data=None):
        training_data = training_data if training_data is not None else self.training_data
        return self.dqn_agent.preprocess_training_data(training_data)

    def act(self, state):
        dqn_action = self.dqn_agent.act(state)
        rf_action, _ = self.rf_agent.predict(state)
        action = self._ensemble_action(dqn_action, rf_action, state)
        action_name = ['Hold', 'Buy', 'Sell'][action]
        self.action_counts[action_name] += 1
        Utils.log_message(f"INFO: Ensemble action: {action} ({action_name}), DQN={dqn_action}, RF={rf_action}, Counts: {self.action_counts}")
        return action

    def _prepare_state(self, training_data, index):
        window = training_data.iloc[index-self.lookback:index][self.expected_feature_columns]
        dqn_state = window.values  # (lookback, n_features)
        rf_state = window.values.flatten().reshape(1, -1)  # (1, lookback * n_features)
        return dqn_state, rf_state

    def _ensemble_action(self, dqn_action, rf_action, state):
        try:
            if dqn_action == rf_action:
                return dqn_action
            
            now = datetime.datetime.now()
            if (self.cached_sharpe['timestamp'] and 
                (now - self.cached_sharpe['timestamp']).total_seconds() < self.cache_duration):
                dqn_sharpe = self.cached_sharpe['dqn']
                rf_sharpe = self.cached_sharpe['rf']
                Utils.log_message(f"DEBUG: Using cached Sharpe ratios")
            else:
                dqn_sharpe = float(self.dqn_agent.evaluate_performance()[0])
                rf_sharpe = float(self.rf_agent.evaluate_performance()[0])
                self.cached_sharpe = {
                    'dqn': dqn_sharpe,
                    'rf': rf_sharpe,
                    'timestamp': now
                }
                self.sharpe_history['dqn'].append(dqn_sharpe)
                self.sharpe_history['rf'].append(rf_sharpe)
                if len(self.sharpe_history['dqn']) > 5:
                    self.sharpe_history['dqn'].pop(0)
                    self.sharpe_history['rf'].pop(0)
            
            ma_dqn_sharpe = np.mean(self.sharpe_history['dqn']) if self.sharpe_history['dqn'] else dqn_sharpe
            ma_rf_sharpe = np.mean(self.sharpe_history['rf']) if self.sharpe_history['rf'] else rf_sharpe
            
            # Get prediction confidence using probabilities
            _, dqn_probabilities = self.dqn_agent.predict(state)
            dqn_confidence = np.max(dqn_probabilities)
            _, rf_proba = self.rf_agent.predict(state)
            rf_confidence = np.max(rf_proba)
            
            # Combine Sharpe and confidence
            dqn_score = ma_dqn_sharpe * dqn_confidence
            rf_score = ma_rf_sharpe * rf_confidence
            dqn_weight = dqn_score / (dqn_score + rf_score + 1e-9)
            dqn_weight = np.clip(dqn_weight, 0.3, 0.7)
            
            chosen_action = dqn_action if random.random() < dqn_weight else rf_action
            Utils.log_message(f"INFO: Disagreement: DQN={dqn_action}, RF={rf_action}, MA DQN Sharpe={ma_dqn_sharpe:.4f}, MA RF Sharpe={ma_rf_sharpe:.4f}, DQN Prob={dqn_confidence:.4f}, RF Prob={rf_confidence:.4f}, dqn_weight={dqn_weight:.4f}. Chose {chosen_action}.")
        except Exception as e:
            Utils.log_message(f"ERROR: Error in ensemble action: {e}")
            if dqn_action == rf_action:
                return dqn_action
            else:
                chosen_action = dqn_action if random.random() < self.dqn_weight_scale else rf_action
                Utils.log_message(f"INFO: Disagreement: DQN={dqn_action}, RF={rf_action}. Chose {chosen_action}.")
            
        return chosen_action

    def _validate_settings(self):
        saved_settings = self._load_settings() or {}
        current_settings = {
            'ticker': self.ticker,
            'training_start_date': self.training_start_date,
            'training_end_date': self.training_end_date,
            'lookback': self.lookback,
            'trade_fee': self.trade_fee,
            'risk_per_trade': self.risk_per_trade,
            'max_trades_per_epoch': self.max_trades_per_epoch,
            'max_fee_per_epoch': self.max_fee_per_epoch,
            'atr_multiplier': self.atr_multiplier,
            'atr_period': self.atr_period,
            'atr_smoothing': self.atr_smoothing,
            'dqn_weight_scale': self.dqn_weight_scale,
            'capital_type': self.capital_type,
            'reference_capital': self.reference_capital,
            'capital_percentage': self.capital_percentage,
            'epochs': self.epochs,
            'confirmation_steps': self.confirmation_steps
        }
        mismatches = []
        for key in current_settings:
            saved_value = saved_settings.get(key, None)
            current_value = current_settings[key]
            if saved_value is not None and saved_value != current_value:
                mismatches.append(f"{key}: Model={saved_value}, Current={current_value}")
        if mismatches:
            Utils.log_message(f"WARNING: Settings mismatch between model and current settings: {mismatches}")
            return False, mismatches
        return True, []

    def _detect_data_drift(self, state):
        try:
            if state.ndim == 1:
                state = state.reshape(1, -1)
            
            expected_rf_features = self.rf_agent.lookback * len(self.rf_agent.expected_feature_columns)
            state_flat_rf = state.reshape(-1)
            if state_flat_rf.shape[0] != expected_rf_features:
                Utils.log_message(f"ERROR: Flattened RF state shape {state_flat_rf.shape} does not match expected {expected_rf_features}")
                return False
            state_mean_rf = state_flat_rf
            state_std_rf = np.std(state_flat_rf) + 1e-9
            
            state_flat_dqn = state[-1] if state.shape[0] > 1 else state_flat_rf
            state_mean_dqn = state_flat_dqn
            state_std_dqn = np.std(state_flat_dqn) + 1e-9

            train_mean_rf = self.rf_agent.scaler.mean_
            train_std_rf = self.rf_agent.scaler.scale_
            mean_z_score_rf = np.abs((state_mean_rf - train_mean_rf) / train_std_rf)
            std_z_score_rf = np.abs(state_std_rf - train_std_rf.mean()) / (train_std_rf.std() + 1e-9)

            train_mean_dqn = self.dqn_agent.state_scaler.mean_
            train_std_dqn = self.dqn_agent.state_scaler.scale_
            mean_z_score_dqn = np.abs((state_mean_dqn - train_mean_dqn) / train_std_dqn)
            std_z_score_dqn = np.abs(state_std_dqn - train_std_dqn.mean()) / (train_std_dqn.std() + 1e-9)

            drift_threshold = 3.0
            if (np.any(mean_z_score_rf > drift_threshold) or std_z_score_rf > drift_threshold or
                np.any(mean_z_score_dqn > drift_threshold) or std_z_score_dqn > drift_threshold):
                Utils.log_message(f"WARNING: training_data drift detected: RF Mean z-score={mean_z_score_rf.max():.2f}, RF Std z-score={std_z_score_rf:.2f}, DQN Mean z-score={mean_z_score_dqn.max():.2f}, DQN Std z-score={std_z_score_dqn:.2f}")
                return True
            return False
        except Exception as e:
            Utils.log_message(f"ERROR: Error in drift detection: {e}")
            return False

    def predict_live_action(self, state, start_date=None, end_date=None):
        is_valid, mismatches = self._validate_settings()
        if not is_valid:
            Utils.log_message(f"WARNING: Proceeding with prediction despite settings mismatches: {mismatches}")
        
        if start_date is not None and end_date is not None and self._detect_data_drift(state):
            Utils.log_message(f"INFO: Triggering RFAgent retraining due to trading_data drift")
            recent_trading_data = self.get_data(start_date, end_date)
            if not recent_trading_data.empty:
                self.rf_agent.train(recent_trading_data, force_grid_search=False)
                Utils.log_message(f"INFO: RFAgent retrained with updated PCA model")
                state_training_data = recent_trading_data[self.expected_feature_columns].dropna()
                if not state_training_data.empty:
                    self.dqn_agent.state_scaler.fit(state_training_data)
                    Utils.log_message(f"INFO: DQN state scaler updated due to trading_data drift")
            else:
                Utils.log_message(f"WARNING: Skipping fetch recent trading data ({start_date} - {end_date}) for retraining due to data drift detection")
        
        dqn_action, dqn_probabilities = self.dqn_agent.predict(state)
        dqn_probabilities[1] -= dqn_probabilities[1] * self.trade_fee
        dqn_probabilities[2] -= dqn_probabilities[2] * self.trade_fee
        dqn_action = int(np.argmax(dqn_probabilities))
        rf_action, rf_proba = self.rf_agent.predict(state)
        action = int(self._ensemble_action(dqn_action, rf_action, state))
        Utils.log_message(f"INFO: Ensemble live action: {action}, DQN={dqn_action}, RF={rf_action}, DQN Prob={dqn_probabilities}, RF Prob={rf_proba}")
        return action, dqn_probabilities


    @staticmethod
    def init_a_new_agent(settings):
        def assert_agent_training_data(agent, settings):
            if not agent.has_enough_training_data():
                raise ValueError("Agent does not have enough training_data for the selected lookback window and batch size. Please select a wider date range or reduce the lookback/batch size.")
            else:
                full_training_data = agent.training_data.copy()

            full_training_data = full_training_data[(full_training_data.index.date >= settings['training_start_date']) & (full_training_data.index.date <= settings['training_end_date'])]

            if full_training_data.empty:
                raise ValueError("No training_data available for the selected date range.")

            if isinstance(full_training_data.columns, pd.MultiIndex):
                full_training_data.columns = [col[0] for col in full_training_data.columns]

            batch_size = agent.dqn_agent.batch_size if agent.dqn_agent is not None else 64
            if len(full_training_data) < settings['lookback'] + batch_size + 1:
                raise ValueError("Not enough training_data for the selected lookback window and batch size. Please select a wider date range or reduce the lookback/batch size.")

            if len(full_training_data) > 0:
                current_price = full_training_data['Close'].iloc[-1]
                min_trade_cost = current_price * (1 + settings['trade_fee'])
                if settings['initial_cash'] < min_trade_cost:
                    raise ValueError(f"Initial capital (€{settings['initial_cash']:.2f}) is too low to buy one share at €{current_price:.2f} with fee {settings['trade_fee']*100:.2f}%. Please increase capital.")

            return full_training_data

        agent = EnsembleAgent(
            ticker=settings['ticker'],
            training_start_date=settings['training_start_date'],
            training_end_date=settings['training_end_date'],
            model_dir=settings['model_dir'],
            lookback=settings['lookback'],
            trade_fee=settings['trade_fee'],
            initial_cash=settings['initial_cash'],
            risk_per_trade=settings['risk_per_trade'],
            max_trades_per_epoch=settings['max_trades_per_epoch'],
            max_fee_per_epoch=settings['max_fee_per_epoch'],
            atr_multiplier=settings['atr_multiplier'],
            atr_period=settings['atr_period'],
            atr_smoothing=settings['atr_smoothing'],
            use_smote=settings['use_smote'],
            dqn_weight_scale=settings['dqn_weight_scale'],
            capital_type=settings['capital_type'],
            reference_capital=settings['reference_capital'],
            capital_percentage=settings['capital_percentage'],
            epochs=settings['epochs'],
            confirmation_steps=settings['confirmation_steps']
        )

        Utils.log_message(f"INFO: Initializing new EnsembleAgent for {agent.ticker} between {agent.training_start_date} and {agent.training_end_date}: {settings}")

        agent.training_training_data = assert_agent_training_data(agent, settings)

        return agent
