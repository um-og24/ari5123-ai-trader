import os
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import sys
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import pickle
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score, train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from itertools import product
from calculations import Calculations
from utils import Utils, FEATURE_COLUMNS

# Set seeds for reproducibility
np.random.seed(42)

class RFAgent:
    def __init__(self, ticker, training_start_date, training_end_date, model_dir, lookback, trade_fee, use_smote, expected_feature_columns=FEATURE_COLUMNS):
        self.ticker = ticker
        self.training_start_date = pd.Timestamp(training_start_date).date()
        self.training_end_date = pd.Timestamp(training_end_date).date()
        self.lookback = lookback
        self.rf_model = None
        self.kmeans = None
        self.scaler = StandardScaler()
        self.pca = None
        self.n_pca_components = None
        self.best_params = None
        self.best_score = -float('inf')
        self.pre_smote_label_distribution = None
        self.post_smote_label_distribution = None
        self.trade_fee = trade_fee
        self.use_smote = use_smote
        self.portfolio_value_history = []
        self.expected_feature_columns = expected_feature_columns
        
        if not model_dir or not isinstance(model_dir, str):
            raise ValueError("'model_dir' param must be a non-empty string")
        if os.path.basename(model_dir) == ticker:
            self.model_dir = model_dir
        else:
            self.model_dir = os.path.join(model_dir, ticker)
        os.makedirs(self.model_dir, exist_ok=True)
        self.rf_best_model_save_path = os.path.join(self.model_dir, f"{ticker}_rf_model_best.pkl")
        self.kmeans_save_path = os.path.join(self.model_dir, f"{ticker}_rf_kmeans_model.pkl")
        self.scaler_save_path = os.path.join(self.model_dir, f"{ticker}_rf_scaler.pkl")
        self.pca_save_path = os.path.join(self.model_dir, f"{ticker}_rf_pca.pkl")
        self.rf_checkpoint_save_path = os.path.join(self.model_dir, "checkpoints", f"{ticker}_checkpoint_rf.pkl")
        self.kmeans_checkpoint_save_path = os.path.join(self.model_dir, "checkpoints", f"{ticker}_checkpoint_rf_kmeans.pkl")
        self.hyperparams_file = os.path.join(self.model_dir, f"{ticker}_rf_hyperparams.json")
        self.settings_save_path = os.path.join(self.model_dir, f"{ticker}_rf_settings.json")
        
        os.makedirs(self.model_dir, exist_ok=True)
        self._load_hyperparameters()

    def _load_hyperparameters(self):
        if os.path.exists(self.hyperparams_file):
            try:
                with open(self.hyperparams_file, 'r') as f:
                    hyperparams = json.load(f)
                self.best_params = hyperparams.get('best_params')
                self.best_score = hyperparams.get('best_score', -float('inf'))
                self.n_pca_components = hyperparams.get('n_pca_components')
                Utils.log_message(f"INFO: Loaded hyperparameters from {self.hyperparams_file}: {hyperparams}")
            except Exception as e:
                Utils.log_message(f"ERROR: Failed to load hyperparameters from {self.hyperparams_file}: {e}")
                self.best_params = None
                self.best_score = -float('inf')
                self.n_pca_components = None
        else:
            Utils.log_message(f"INFO: No hyperparameters file found at {self.hyperparams_file}")

    def _save_hyperparameters(self):
        if self.best_params is not None:
            hyperparams = {
                'best_params': self.best_params,
                'best_score': self.best_score,
                'n_pca_components': self.n_pca_components,
                'use_smote': self.use_smote,
            }
            try:
                os.makedirs(self.model_dir, exist_ok=True)
                with open(self.hyperparams_file, 'w') as f:
                    json.dump(hyperparams, f, indent=4)
                Utils.log_message(f"INFO: Saved hyperparameters to {self.hyperparams_file}")
            except Exception as e:
                Utils.log_message(f"ERROR: Failed to save hyperparameters to {self.hyperparams_file}: {e}")

    def _save_settings(self):
        settings = {
            'ticker': self.ticker,
            'training_start_date': pd.Timestamp(self.training_start_date).isoformat(),
            'training_end_date': pd.Timestamp(self.training_end_date).isoformat(),
            'lookback': self.lookback,
            'trade_fee': self.trade_fee,
            'n_pca_components': self.n_pca_components,
            'use_smote': self.use_smote,
        }
        try:
            os.makedirs(self.model_dir, exist_ok=True)
            with open(self.settings_save_path, 'w') as f:
                json.dump(settings, f, indent=4)
            Utils.log_message(f"INFO: Saved RF settings to {self.settings_save_path}")
        except Exception as e:
            Utils.log_message(f"ERROR: Failed to save RF settings: {e}")

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
                self.n_pca_components = settings.get('n_pca_components')
                self.use_smote = settings.get('use_smote', self.use_smote)
                Utils.log_message(f"INFO: Loaded RF settings from {self.settings_save_path}: {settings}")
                return settings
            except Exception as e:
                Utils.log_message(f"ERROR: Failed to load RF settings: {e}")
        else:
            Utils.log_message(f"INFO: No RF settings file found at {self.settings_save_path}")
        return None

    def preprocess_data(self, data):
        return Utils.preprocess_data(data, volatility_period=self.atr_period, atr_smoothing=self.atr_smoothing) if not set(FEATURE_COLUMNS).issubset(data.columns) else data

    def _create_labels(self, data):
        returns = data['Returns'].shift(-1)
        
        # Use ATR and Returns to create a threshold in the scale of Returns
        atr_scaled = data['ATR'] / data['ATR'].mean()  # Normalize ATR
        threshold = data['Returns'].std() * atr_scaled.mean()  # Scale by Returns volatility
        threshold *= 0.5  # Adjust to ensure reasonable number of Buy/Sell labels
        
        labels = np.zeros(len(returns))
        labels[returns > threshold] = 1  # Buy
        labels[returns < -threshold] = 2  # Sell
        
        # Handle NaNs and slice
        valid_mask = ~np.isnan(returns) & ~np.isnan(data['ATR'])
        labels = labels[valid_mask]
        labels = labels[self.lookback:len(data)]
        
        # Log statistics for debugging
        Utils.log_message(f"DEBUG: Returns mean={returns.mean():.6f}, std={returns.std():.6f}, min={returns.min():.6f}, max={returns.max():.6f}")
        Utils.log_message(f"DEBUG: ATR mean={data['ATR'].mean():.6f}, std={data['ATR'].std():.6f}")
        Utils.log_message(f"DEBUG: Threshold={threshold:.6f}")
        unique, counts = np.unique(labels, return_counts=True)
        Utils.log_message(f"INFO: Initial label distribution: {dict(zip(unique, counts))}")
        
        # Fallback: Use quantile-based labeling if insufficient diversity
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            Utils.log_message(f"WARNING: Insufficient label diversity with threshold {threshold:.6f}. Falling back to quantile-based labeling.")
            valid_returns = returns[valid_mask][self.lookback:len(data)]
            if len(valid_returns) > 0:
                buy_threshold = valid_returns.quantile(0.75)
                sell_threshold = valid_returns.quantile(0.25)
                labels = np.zeros(len(valid_returns))
                labels[valid_returns > buy_threshold] = 1  # Buy
                labels[valid_returns < sell_threshold] = 2  # Sell
                unique, counts = np.unique(labels, return_counts=True)
                Utils.log_message(f"DEBUG: Quantile thresholds: Buy={buy_threshold:.6f}, Sell={sell_threshold:.6f}")
                Utils.log_message(f"INFO: Fallback label distribution: {dict(zip(unique, counts))}")
            
            # Final diversity check
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                Utils.log_message(f"ERROR: Still insufficient label diversity after fallback: {unique_labels}")
                raise ValueError(f"Insufficient label diversity: {unique_labels}")
        
        return labels

    def _prepare_features(self, data, apply_pca=True):
        features = []
        feature_columns = self.expected_feature_columns
        valid_indices = []
        for i in range(self.lookback, len(data)):
            window = data.iloc[i - self.lookback:i][feature_columns]
            if window.isna().any().any():
                continue  # Skip windows with NaN values
            feature = window.values.flatten()
            features.append(feature)
            valid_indices.append(i)
        features = np.array(features)
        Utils.log_message(f"INFO: Prepared {len(features)} features from {len(data)} rows")
        if apply_pca and self.pca is not None:
            features = self.pca.transform(features)
        return features, valid_indices

    def predict(self, state):
        if self.rf_model is None or self.kmeans is None or self.pca is None:
            Utils.log_message(f"WARNING: RF model, KMeans, or PCA not trained. Defaulting to Hold.")
            return 0, [0.33, 0.33, 0.33]
        
        expected_features = self.lookback * len(self.expected_feature_columns)
        if state.shape == (self.lookback, len(self.expected_feature_columns)):
            state = state.reshape(1, -1)
        elif state.shape != (1, expected_features):
            Utils.log_message(f"ERROR: Unexpected state shape: {state.shape}, expected (1, {expected_features})")
            # return 0, [0.33, 0.33, 0.33]
            raise ValueError(f"ERROR: Unexpected state shape: {state.shape}, expected (1, {expected_features})")
        
        state_scaled = self.scaler.transform(state)
        state_pca = self.pca.transform(state_scaled)
        cluster = self.kmeans.predict(state_pca)[0]
        
        proba = self.rf_model.predict_proba(state_pca)[0]
        full_proba = np.array([0.33, 0.33, 0.33])
        n_classes = len(self.rf_model.classes_)
        for i, class_idx in enumerate(self.rf_model.classes_):
            full_proba[int(class_idx)] = proba[i]
        full_proba[1] *= (1 - self.trade_fee)
        full_proba[2] *= (1 - self.trade_fee)
        action = int(np.argmax(full_proba))

        return action, full_proba

    def evaluate_performance(self, data=None):
        if data is None:
            Utils.log_message(f"WARNING: No data provided for RF performance evaluation. Using empty history.")
            return 0.0, 0.0, 0.0, 0.0
        
        processed_data = self.preprocess_data(data)
        if processed_data.empty:
            Utils.log_message(f"ERROR: No valid preprocessed data for RF performance evaluation")
            return 0.0, 0.0, 0.0, 0.0
        
        features = self._prepare_features(processed_data, apply_pca=True)
        if len(features) == 0:
            Utils.log_message(f"ERROR: No valid features for RF performance evaluation")
            return 0.0, 0.0, 0.0, 0.0
        
        self.portfolio_value_history = []
        cash = 10000
        shares = 0
        cumulative_fees = 0
        num_trades = 0
        
        for t in range(self.lookback, len(processed_data) - 1):
            state = features[t - self.lookback]
            action, _ = self.predict(state.reshape(1, -1), data=processed_data)
            price = processed_data['Close'].iloc[t]
            
            if action == 1 and cash >= price * (1 + self.trade_fee):
                shares_to_buy = min(100, int(cash / (price * (1 + self.trade_fee))))
                cost = shares_to_buy * price
                fee = cost * self.trade_fee
                cash -= (cost + fee)
                shares += shares_to_buy
                cumulative_fees += fee
                num_trades += 1
            elif action == 2 and shares > 0:
                revenue = shares * price
                fee = revenue * self.trade_fee
                cash += revenue - fee
                shares = 0
                cumulative_fees += fee
                num_trades += 1
            
            portfolio_value = cash + shares * price
            self.portfolio_value_history.append(portfolio_value)
        
        values = np.array(self.portfolio_value_history)
        if len(values) < 2:
            return 0.0, 0.0, float(cumulative_fees), 0.0
        
        returns = np.diff(values) / values[:-1]
        sharpe_ratio = float(np.sqrt(252) * np.mean(returns) / (np.std(returns) + 1e-9))
        drawdown = float(np.max(np.maximum.accumulate(values) - values))
        Utils.log_message(f"INFO: RF performance: Sharpe={sharpe_ratio:.4f}, Drawdown={drawdown:.2f}, Fees={cumulative_fees:.2f}, Trades={num_trades}")
        return sharpe_ratio, drawdown, float(cumulative_fees), 0.0

    def save_model(self):
        os.makedirs(self.model_dir, exist_ok=True)
        try:
            with open(self.rf_best_model_save_path, "wb") as f:
                pickle.dump(self.rf_model, f)
            with open(self.kmeans_save_path, "wb") as f:
                pickle.dump(self.kmeans, f)
            with open(self.scaler_save_path, "wb") as f:
                pickle.dump(self.scaler, f)
            with open(self.pca_save_path, "wb") as f:
                pickle.dump(self.pca, f)
            self._save_settings()
            Utils.log_message(f"INFO: Saved RF model to {self.rf_best_model_save_path}, KMeans to {self.kmeans_save_path}, Scaler to {self.scaler_save_path}, PCA to {self.pca_save_path}")
        except Exception as e:
            Utils.log_message(f"ERROR: Failed to save RF model: {e}")
            raise

    def load_model(self):
        if not all(os.path.exists(path) for path in [self.rf_best_model_save_path, self.kmeans_save_path, self.scaler_save_path, self.pca_save_path]):
            Utils.log_message(f"WARNING: No RF model, KMeans, Scaler, or PCA found at {self.model_dir}. Starting with fresh model.")
            return
        try:
            with open(self.rf_best_model_save_path, "rb") as f:
                self.rf_model = pickle.load(f)
            with open(self.kmeans_save_path, "rb") as f:
                self.kmeans = pickle.load(f)
            with open(self.scaler_save_path, "rb") as f:
                self.scaler = pickle.load(f)
            with open(self.pca_save_path, "rb") as f:
                self.pca = pickle.load(f)
            self._load_settings()
            self._load_hyperparameters()
            Utils.log_message(f"INFO: Loaded RF model from {self.rf_best_model_save_path}, KMeans from {self.kmeans_save_path}, Scaler from {self.scaler_save_path}, PCA from {self.pca_save_path}")
        except (ValueError, OSError, pickle.UnpicklingError) as e:
            Utils.log_message(f"ERROR: Error loading RF model: {e}. Starting with fresh model.")
            self.rf_model = None
            self.kmeans = None
            self.scaler = StandardScaler()
            self.pca = None

    def save_checkpoint(self):
        os.makedirs(os.path.join(self.model_dir, "checkpoints"), exist_ok=True)
        try:
            if self.rf_model:
                with open(self.rf_checkpoint_save_path, "wb") as f:
                    pickle.dump(self.rf_model, f)
            if self.kmeans:
                with open(self.kmeans_checkpoint_save_path, "wb") as f:
                    pickle.dump(self.kmeans, f)
            Utils.log_message(f"INFO: Saved RF checkpoint to {self.rf_checkpoint_save_path} and KMeans checkpoint to {self.kmeans_checkpoint_save_path}")
            return True
        except Exception as e:
            Utils.log_message(f"ERROR: Error saving RF checkpoint: {e}")
            return False

    def load_checkpoint(self):
        if os.path.exists(self.rf_checkpoint_save_path) and os.path.exists(self.kmeans_checkpoint_save_path):
            try:
                with open(self.rf_checkpoint_save_path, "rb") as f:
                    self.rf_model = pickle.load(f)
                with open(self.kmeans_checkpoint_save_path, "rb") as f:
                    self.kmeans = pickle.load(f)
                Utils.log_message(f"INFO: Loaded RF checkpoint from {self.rf_checkpoint_save_path} and KMeans checkpoint from {self.kmeans_checkpoint_save_path}")
                self._load_hyperparameters()
                return True
            except (ValueError, OSError, pickle.UnpicklingError) as e:
                Utils.log_message(f"ERROR: Error loading RF checkpoint: {e}")
                self.rf_model = None
                self.kmeans = None
                return False
        return False

    def clear_checkpoints(self):
        checkpoint_files = [self.rf_checkpoint_save_path, self.kmeans_checkpoint_save_path]
        for file in checkpoint_files:
            if os.path.exists(file):
                os.remove(file)
                Utils.log_message(f"INFO: Removed RF checkpoint file: {file}")
        checkpoint_dir = os.path.join(self.model_dir, "checkpoints")
        if os.path.exists(checkpoint_dir) and not os.listdir(checkpoint_dir):
            os.rmdir(checkpoint_dir)
            Utils.log_message(f"INFO: Removed empty checkpoints directory")


    def train(self, data, progress_callback=None, force_grid_search=False):
        Utils.log_message(f"INFO: Training RF Agent for {self.ticker}")

        features, labels = self._prepare_training_data(data)
        if features is None:
            return

        features_scaled = self.scaler.fit_transform(features)
        features_scaled, labels = self._handle_smote(features_scaled, labels)

        features_pca = self._scale_and_pca(features_scaled)
        self._train_kmeans(features_pca)

        X_train, X_val, y_train, y_val = self._split_time_series(features_pca, labels)

        if not hasattr(self, 'best_score'):
            self.best_score = 0

        if self.best_params and not force_grid_search:
            val_score = self._train_and_evaluate_model(self.best_params, X_train, y_train, X_val, y_val)
            self._finalize_model(val_score, self.best_params, X_val)
            if progress_callback:
                self._emit_final_progress(progress_callback, val_score)
        else:
            self._grid_search_rf(features_pca, labels, X_train, y_train, X_val, y_val, progress_callback)

    def _prepare_training_data(self, data):
        processed = self.preprocess_data(data)
        if processed.empty:
            Utils.log_message("ERROR: No valid preprocessed data")
            return None, None

        features, valid_idx = self._prepare_features(processed, apply_pca=False)
        labels = self._create_labels(processed)

        if len(valid_idx) != len(labels):
            labels = labels[:len(valid_idx)]
            features = features[:len(labels)]
            Utils.log_message(f"INFO: Aligned to {len(features)} samples")

        if len(features) < 50:
            Utils.log_message("ERROR: Insufficient data")
            return None, None

        if len(np.unique(labels)) < 2:
            raise ValueError("Insufficient label diversity")

        self.pre_smote_label_distribution = dict(zip(*np.unique(labels, return_counts=True)))
        return features, labels

    def _handle_smote(self, features, labels):
        if self.use_smote:
            try:
                features, labels, dist = Calculations.perform_smote(features, labels)
                self.post_smote_label_distribution = dist
                Utils.log_message(f"INFO: Post-SMOTE label distribution: {dist}")
            except ValueError as e:
                Utils.log_message(f"WARNING: SMOTE failed: {e}")
                self.post_smote_label_distribution = self.pre_smote_label_distribution
        else:
            self.post_smote_label_distribution = self.pre_smote_label_distribution
        return features, labels

    def _scale_and_pca(self, features):
        pca_result = Calculations.apply_pca(features, features)
        self.pca = pca_result['pca']
        self.n_pca_components = pca_result['n_pca_components']
        Utils.log_message(f"INFO: Applied PCA: {self.n_pca_components} components, variance: {sum(self.pca.explained_variance_ratio_):.4f}")
        return pca_result['X_train']

    def _train_kmeans(self, features):
        best_score, best_n = -1, 3
        for n in range(2, 6):
            kmeans = KMeans(n_clusters=n, random_state=42)
            score = silhouette_score(features, kmeans.fit_predict(features))
            if score > best_score:
                best_score, best_n = score, n
        self.kmeans = KMeans(n_clusters=best_n, random_state=42)
        self.kmeans.fit(features)
        Utils.log_message(f"INFO: KMeans trained with {best_n} clusters, silhouette score: {best_score:.4f}")

    def _split_time_series(self, features, labels):
        val_size = len(features) // 3
        return features[:-val_size], features[-val_size:], labels[:-val_size], labels[-val_size:]

    def _train_and_evaluate_model(self, params, X_train, y_train, X_val, y_val):
        model = RandomForestClassifier(**params, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
        self.rf_model = model
        return model.score(X_val, y_val)

    def _finalize_model(self, val_score, params, X_val):
        if val_score > self.best_score:
            self.best_score = val_score
            self.best_params = params
            self.save_model()
            self._save_hyperparameters()
            importances = self.rf_model.feature_importances_
            Utils.log_message(f"INFO: New best RF model: Score={val_score:.4f}, Params={params}")
            Utils.log_message(f"INFO: PCA importances: " + ", ".join(f"PC{i+1}: {imp:.4f}" for i, imp in enumerate(importances)))

    def _emit_final_progress(self, callback, val_score):
        callback(
            1.0,
            params=self.best_params,
            mean_cv_score=val_score,
            best_params=self.best_params,
            best_score=self.best_score,
            training_phase="rf",
            override_status_message=f"RF Training completed: Validation Score={val_score:.4f}, Params={self.best_params}",
            pre_smote_label_distribution=self.pre_smote_label_distribution,
            post_smote_label_distribution=self.post_smote_label_distribution
        )

    def _grid_search_rf(self, features, labels, X_train, y_train, X_val, y_val, callback):
        from itertools import product
        tscv = TimeSeriesSplit(n_splits=2)
        param_grid = {
            'n_estimators': [50, 100, 150, 200, 250, 300],
            'max_depth': [None, 3, 5, 7, 10],
            'min_samples_split': [5, 10, 15],
            'min_samples_leaf': [5, 10, 15],
            'max_features': ['sqrt', 'log2', None]
        }
        param_combinations = list(product(*param_grid.values()))

        for i, combo in enumerate(param_combinations):
            params = dict(zip(param_grid.keys(), combo))
            model = RandomForestClassifier(**params, random_state=42, class_weight='balanced')
            cv_score = np.mean(cross_val_score(model, features, labels, cv=tscv, scoring='accuracy', n_jobs=-1))
            model.fit(X_train, y_train)
            val_score = model.score(X_val, y_val)

            score_gap = cv_score - val_score
            if val_score > self.best_score and score_gap < 0.15:
                self.rf_model = model
                self._finalize_model(val_score, params, X_val)

            progress = (i + 1) / len(param_combinations)
            Utils.log_message(f"INFO: Grid Search {progress*100:.1f}% - Params: {params}, CV: {cv_score:.4f}, Val: {val_score:.4f}, Best: {self.best_score:.4f}")
            if callback:
                callback(
                    progress,
                    params=params,
                    mean_cv_score=val_score,
                    best_params=self.best_params,
                    best_score=self.best_score,
                    training_phase="rf",
                    pre_smote_label_distribution=self.pre_smote_label_distribution,
                    post_smote_label_distribution=self.post_smote_label_distribution
                )
