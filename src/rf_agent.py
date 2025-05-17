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

class RFAgent:
    def __init__(self, ticker, start_date, end_date, model_dir, lookback, trade_fee,
                 expected_feature_columns=FEATURE_COLUMNS):
        self.ticker = ticker
        self.start_date = pd.Timestamp(start_date).date()
        self.end_date = pd.Timestamp(end_date).date()
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
                'n_pca_components': self.n_pca_components
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
            'lookback': self.lookback,
            'trade_fee': self.trade_fee,
            'n_pca_components': self.n_pca_components
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
                self.lookback = settings.get('lookback', self.lookback)
                self.trade_fee = settings.get('trade_fee', self.trade_fee)
                self.n_pca_components = settings.get('n_pca_components')
                Utils.log_message(f"INFO: Loaded RF settings from {self.settings_save_path}: {settings}")
                return settings
            except Exception as e:
                Utils.log_message(f"ERROR: Failed to load RF settings: {e}")
        else:
            Utils.log_message(f"INFO: No RF settings file found at {self.settings_save_path}")
        return None

    def preprocess_data(self, data):
        return Utils.preprocess_data(data)

    def _prepare_features(self, data, apply_pca=True):
        features = []
        feature_columns = self.expected_feature_columns
        for i in range(self.lookback, len(data)):
            window = data.iloc[i - self.lookback:i][feature_columns]
            feature = window.values.flatten()
            features.append(feature)
        features = np.array(features)
        
        if apply_pca and self.pca is not None:
            features = self.pca.transform(features)

        return features

    def _create_labels(self, data):
        returns = data['Close'].pct_change().shift(-1)
        atr = data['ATR'] / data['Close']  # Normalizing ATR
        threshold = atr.mean()
        labels = np.zeros(len(returns))
        labels[returns > threshold] = 1
        labels[returns < -threshold] = 2
        return labels[self.lookback:len(data)]

    def train(self, data, use_smote=False, progress_callback=None, force_grid_search=False):
        Utils.log_message(f"INFO: Training RF Agent for {self.ticker}")
        processed_data = self.preprocess_data(data)
        if processed_data.empty:
            Utils.log_message(f"ERROR: No valid preprocessed data for RF training")
            return
        
        # Prepare features and labels
        features = self._prepare_features(processed_data, apply_pca=False)
        labels = self._create_labels(processed_data)
        
        tscv = TimeSeriesSplit(n_splits=5)
        for fold, (train_idx, val_idx) in enumerate(tscv.split(features)):
            X_train, X_val = features[train_idx], features[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]
            self.scaler.fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
        
        Utils.log_message(f"DEBUG: Features shape: {features.shape}, Labels shape: {labels.shape}")
        if len(features) == 0 or len(labels) == 0:
            Utils.log_message(f"ERROR: No valid features or labels for RF training")
            return
        
        if len(features) != len(labels):
            Utils.log_message(f"ERROR: Feature-label mismatch: {len(features)} features, {len(labels)} labels")
            raise ValueError(f"Feature-label mismatch: {len(features)} features, {len(labels)} labels")
        
        # Split data into training and validation sets (70-30 split)
        X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.3, random_state=42, stratify=labels)
        Utils.log_message(f"INFO: Train set: {X_train.shape[0]} samples, Validation set: {X_val.shape[0]} samples")
        
        # Scale features
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        if use_smote:
            try:
                self.post_smote_label_distribution = Calculations.perform_smote(X_train_scaled, y_train, self.pre_smote_label_distribution)
                Utils.log_message(f"INFO: Post-SMOTE label distribution: Hold={self.post_smote_label_distribution.get(0, 0)}, Buy={self.post_smote_label_distribution.get(1, 0)}, Sell={self.post_smote_label_distribution.get(2, 0)}")
            except ValueError as e:
                Utils.log_message(f"WARNING: SMOTE failed: {e}. Proceeding with original training data.")
                self.post_smote_label_distribution = self.pre_smote_label_distribution
        else:
            # Skip SMOTE
            Utils.log_message(f"INFO: SMOTE skipped to match validation set's class distribution")
            self.post_smote_label_distribution = self.pre_smote_label_distribution

        # Apply PCA
        pca_results = Calculations.apply_pca(X_train_scaled, X_val_scaled)
        self.pca = pca_results['pca']
        X_train_scaled = pca_results['X_train']
        X_val_scaled = pca_results['X_val']
        self.n_pca_components = pca_results['n_pca_components']
        Utils.log_message(f"INFO: Applied PCA: {self.n_pca_components} components retained, explained variance ratio: {sum(self.pca.explained_variance_ratio_):.4f}")

        # Train KMeans on PCA-transformed training data
        best_score, best_n = -1, 3
        for n in range(2, 6):
            kmeans = KMeans(n_clusters=n, random_state=42)
            labels = kmeans.fit_predict(X_train_scaled)
            score = silhouette_score(X_train_scaled, labels)
            if score > best_score:
                best_score, best_n = score, n
        self.kmeans = KMeans(n_clusters=best_n, random_state=42)
        self.kmeans.fit(X_train_scaled)
        self.kmeans = KMeans(n_clusters=3, random_state=42)
        self.kmeans.fit(X_train_scaled)
        
        if self.best_params is not None and not force_grid_search:
            Utils.log_message(f"INFO: Using pre-loaded hyperparameters: {self.best_params}")
            self.rf_model = RandomForestClassifier(**self.best_params, random_state=42, class_weight='balanced')
            self.rf_model.fit(X_train_scaled, y_train)
            train_score = self.rf_model.score(X_train_scaled, y_train)
            val_score = self.rf_model.score(X_val_scaled, y_val)
            if val_score > self.best_score:
                self.best_score = val_score
                self.save_model()
                self._save_hyperparameters()
                Utils.log_message(f"INFO: New best RF model saved: Train Score={train_score:.4f}, Validation Score={val_score:.4f}, Params={self.best_params}")
                # Log feature importances (PCA components)
                importances = self.rf_model.feature_importances_
                Utils.log_message(f"INFO: PCA component importances: " + ", ".join(f"PC{i+1}: {imp:.4f}" for i, imp in enumerate(importances)))
            else:
                Utils.log_message(f"INFO: Pre-loaded hyperparameters did not improve best score: {self.best_score:.4f}")
            if progress_callback:
                progress_callback(
                    1.0,
                    params=self.best_params,
                    mean_cv_score=train_score,
                    best_params=self.best_params,
                    best_score=self.best_score,
                    training_phase="rf",
                    override_status_message=f"RF Training completed with loaded hyperparameters: Train Score={train_score:.4f}, Val Score={val_score:.4f}, Best Score={self.best_score:.4f}, Params={self.best_params}",
                    pre_smote_label_distribution=self.pre_smote_label_distribution,
                    post_smote_label_distribution=self.post_smote_label_distribution
                )
        else:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 3, 5, 7, 10],
                'min_samples_split': [5, 10, 15],
                'min_samples_leaf': [5, 10, 15],
                'max_features': ['sqrt', 'log2']
            }
            
            param_combinations = list(product(*param_grid.values()))
            total_combinations = len(param_combinations)
            
            # Log validation set details
            Utils.log_message(f"INFO: Validation set size: {X_val_scaled.shape[0]}")
            unique_val, counts_val = np.unique(y_val, return_counts=True)
            Utils.log_message(f"INFO: Validation set label distribution: {dict(zip(unique_val, counts_val))}")
            
            for i, params in enumerate(param_combinations):
                params_dict = dict(zip(param_grid.keys(), params))
                rf = RandomForestClassifier(**params_dict, random_state=42, class_weight='balanced')
                scores = cross_val_score(rf, X_train_scaled, y_train, cv=5, scoring='accuracy', n_jobs=-1)
                mean_test_score = np.mean(scores)
                
                # Fit the model to evaluate on validation set
                rf.fit(X_train_scaled, y_train)
                val_score = rf.score(X_val_scaled, y_val)
                
                # Log score comparison
                score_gap = mean_test_score - val_score
                Utils.log_message(f"INFO: Score comparison: CV Score={mean_test_score:.4f}, Validation Score={val_score:.4f}, Difference={score_gap:.4f}")
                
                # Save model if score gap is reasonable
                if val_score > self.best_score and score_gap < 0.1:  # Relaxed threshold
                    self.best_params = params_dict
                    self.best_score = val_score
                    self.rf_model = rf
                    self.save_model()
                    self._save_hyperparameters()
                    Utils.log_message(f"INFO: New best RF model saved: CV Score={mean_test_score:.4f}, Validation Score={val_score:.4f}, Params={params_dict}")
                    # Log feature importances (PCA components)
                    importances = self.rf_model.feature_importances_
                    Utils.log_message(f"INFO: PCA component importances: " + ", ".join(f"PC{i+1}: {imp:.4f}" for i, imp in enumerate(importances)))
                elif score_gap >= 0.1:
                    Utils.log_message(f"WARNING: Model skipped due to large score gap indicating overfitting: CV Score={mean_test_score:.4f}, Val Score={val_score:.4f}, Gap={score_gap:.4f}")
                
                progress = (i + 1) / total_combinations
                Utils.log_message(f"INFO: RF Grid Search: Progress {progress*100:.1f}%, Params: {params_dict}, CV Score: {mean_test_score:.4f}, Validation Score: {val_score:.4f}, Best Validation Score: {self.best_score:.4f}")
                if progress_callback:
                    progress_callback(
                        progress,
                        params=params_dict,
                        mean_cv_score=mean_test_score,
                        best_params=self.best_params,
                        best_score=self.best_score,
                        training_phase="rf",
                        pre_smote_label_distribution=self.pre_smote_label_distribution,
                        post_smote_label_distribution=self.post_smote_label_distribution
                    )
            
            if self.best_params is not None:
                self._save_hyperparameters()
                if self.rf_model:
                    importances = self.rf_model.feature_importances_
                    Utils.log_message(f"INFO: Final model PCA component importances: {", ".join(f"PC{i+1}: {imp:.4f}" for i, imp in enumerate(importances))}")
            if progress_callback:
                progress_callback(
                    1.0,
                    params=self.best_params,
                    mean_cv_score=self.best_score,
                    best_params=self.best_params,
                    best_score=self.best_score,
                    training_phase="rf",
                    #override_status_message=f"RF Training completed: Best Validation Score={self.best_score:.4f}, Best Params={self.best_params}",
                    pre_smote_label_distribution=self.pre_smote_label_distribution,
                    post_smote_label_distribution=self.post_smote_label_distribution
                )
        
        Utils.log_message(f"INFO: RF training completed: Best Validation Score={self.best_score:.4f}, Best Params={self.best_params}")

    def predict(self, state):
        if self.rf_model is None or self.kmeans is None or self.pca is None:
            Utils.log_message(f"WARNING: RF model, KMeans, or PCA not trained. Defaulting to Hold.")
            return 0, [0.33, 0.33, 0.33]
        
        expected_features = self.lookback * len(self.expected_feature_columns)
        if state.shape == (self.lookback, len(self.expected_feature_columns)):
            state = state.reshape(1, -1)
        elif state.shape != (1, expected_features):
            Utils.log_message(f"ERROR: Unexpected state shape: {state.shape}, expected (1, {expected_features})")
            return 0, [0.33, 0.33, 0.33]
        
        state_scaled = self.scaler.transform(state)
        state_pca = self.pca.transform(state_scaled)
        cluster = self.kmeans.predict(state_pca)[0]
        
        proba = self.rf_model.predict_proba(state_pca)[0]
        proba[1] *= (1 - self.trade_fee)
        proba[2] *= (1 - self.trade_fee)
        action = int(np.argmax(proba))

        return action, proba

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