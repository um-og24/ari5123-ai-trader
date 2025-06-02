import os
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import sys
sys.dont_write_bytecode = True

import logging
from logging.handlers import RotatingFileHandler
import re
import streamlit as st
import numpy as np
import pandas as pd

LOG_FILE_PATH = "system_logs/system_log.txt"

def _init_logger():
    os.makedirs("system_logs", exist_ok=True)
    logger = logging.getLogger("system_logger")
    logger.setLevel(logging.DEBUG)
    handler = RotatingFileHandler(LOG_FILE_PATH, maxBytes=5*1024*1024, backupCount=3, encoding="utf-8")
    formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    if not logger.hasHandlers():
        logger.addHandler(handler)
    return logger

logger = _init_logger()

REQUIRED_DATA_COLUMNS=['Open', 'High', 'Low', 'Close', 'Volume'] # OHLCV Features
FEATURE_COLUMNS=REQUIRED_DATA_COLUMNS + ['RSI', 'SMA20',  'SMA70', 'MACD', 'BB_Upper', 'BB_Lower', 'BB', 'BB_Penetration',
                'ATR', 'Returns', 'Volatility', 'RSI_Slope', 'Price_Slope', 'RSI_Divergence',
                'Lag1_Return', 'Lag3_Return', 'Lag5_Return', 'Skewness', 'Kurtosis']

SETTING_KEYS=['ticker', 'training_start_date', 'training_end_date', 'model_dir', 'capital_type', 'initial_cash', 'reference_capital', 'capital_percentage',
            'trade_fee', 'lookback', 'batch_size', 'risk_per_trade', 'epochs', 'max_trades_per_epoch', 'max_fee_per_epoch', 'confirmation_steps',
            'dqn_weight_scale', 'atr_multiplier', 'atr_period', 'atr_smoothing', 'use_smote', 'trading_simulation_delay', 'run_all_at_once']

LOG_LEVEL_METHODS = {
    "ERROR: ": logger.error,
    "WARNING: ": logger.warning,
    "INFO: ": logger.info,
    "DEBUG: ": logger.debug,
}
DEFAULT_LOG_METHOD = logger.info
IDENTIFIERS = ["ERROR: ", "WARNING: ", "INFO: ", "DEBUG: "]
ICONS = ["🚨", "⚠️", "ℹ️", "🐞", "📢"]
LOG_HANDLERS = {
    "ERROR: ": (st.error, ICONS[0]),
    "WARNING: ": (st.warning, ICONS[1]),
    "INFO: ": (st.info, ICONS[2]),
    "DEBUG: ": (st.warning, ICONS[3]),
}
DEFAULT_HEADER = (st.success, ICONS[4])


class Utils:
    """Centralized class for utility functions."""

    @staticmethod
    @st.cache_data(ttl=1800)
    def preprocess_data(df, required_columns=REQUIRED_DATA_COLUMNS, rsi_period=14,
                        macd_params=(12, 26, 9), bb_window=20, volatility_period=14, atr_smoothing=True,
                        rsi_divergence_params={'diff_period': 5, 'smooth_period': 3, 'rsi_threshold': 0.2}):
        """
        Preprocess financial data by adding technical indicators incrementally to avoid look-ahead bias.

        Args:
            df (pd.DataFrame): Input DataFrame with required columns.
            required_columns (list[str]): List of required columns.
            rsi_period (int): Period for RSI calculation.
            macd_params (tuple): MACD parameters (short_ema, long_ema, signal_ema).
            bb_window (int): Window for Bollinger Bands and SMA.
            volatility_period (int): Period for volatility and ATR.
            atr_smoothing (bool): Use EMA for ATR if True.
            rsi_divergence_params (dict): RSI divergence parameters.

        Returns:
            pd.DataFrame: Preprocessed DataFrame with indicators.
        """
        from calculations import Calculations
        try:
            if not isinstance(df, pd.DataFrame):
                Utils.log_message(f"ERROR: Input is not a pandas DataFrame: type={type(df)}")
                raise ValueError(f"Input must be a pandas DataFrame, got {type(df)}")

            Utils.log_message(f"DEBUG: Input DataFrame shape: {df.shape}, columns: {df.columns.tolist()}")

            if isinstance(df.columns, pd.MultiIndex):
                ticker = df.columns[0][1] if len(df.columns) > 0 else None
                if ticker:
                    try:
                        df = df.xs(ticker, level=1, axis=1)
                        df.columns = [col.split()[-1] for col in df.columns]
                    except KeyError:
                        Utils.log_message(f"ERROR: Ticker {ticker} not found in MultiIndex columns")
                        raise ValueError(f"Cannot extract ticker {ticker} from MultiIndex columns")
                else:
                    Utils.log_message(f"ERROR: Cannot determine ticker from MultiIndex columns")
                    raise ValueError("MultiIndex columns detected, but no ticker found")

            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                Utils.log_message(f"ERROR: Missing required columns: {missing_cols}")
                raise ValueError(f"Missing required columns: {missing_cols}")

            for col in required_columns:
                series = df[col]
                if not isinstance(series, pd.Series):
                    Utils.log_message(f"ERROR: Column {col} did not return a Series: type={type(series)}")
                    raise ValueError(f"Column {col} must be a pandas Series, got {type(series)}")
                if not pd.api.types.is_numeric_dtype(series):
                    Utils.log_message(f"ERROR: Column {col} is not numeric: dtype={series.dtype}")
                    raise ValueError(f"Column {col} must be numeric, got dtype {series.dtype}")

            if not df.index.is_monotonic_increasing:
                Utils.log_message(f"WARNING: Data index is not monotonic; sorting by index")
                df = df.sort_index()

            data = df[required_columns].copy()

            # Incremental indicator calculation to avoid look-ahead bias
            def compute_rolling_indicators(window_data):
                # Add technical indicators
                data = window_data.copy()
                data['RSI'] = Calculations.compute_rsi(data['Close'], rsi_period)
                data['SMA20'] = data['Close'].rolling(window=20, min_periods=1).mean()
                data['SMA70'] = data['Close'].rolling(window=70, min_periods=1).mean()
                data['MACD'] = Calculations.compute_macd(data['Close'], macd_params)
                data['BB_Upper'], data['BB_Lower'], data['BB'] = Calculations.compute_bollinger_band(data['Close'], bb_window)
                data['BB_Penetration'] = np.select(
                    [
                        data['Close'] > data['BB_Upper'],
                        data['Close'] < data['BB_Lower']
                    ],
                    [1, -1],
                    default=0
                )
                
                # Add ATR
                data['ATR'] = Calculations.compute_atr(data, period=volatility_period, smoothing=atr_smoothing)
                
                # Add returns and volatility
                data['Returns'] = data['Close'].pct_change() # Calculate arithmetic returns (percentage change)
                data['Volatility'] = data['Close'].pct_change().rolling(window=volatility_period, min_periods=1).std()
                
                # Calculate RSI Divergence
                data['RSI_Slope'] = data['RSI'].diff(rsi_divergence_params['diff_period']).rolling(rsi_divergence_params['smooth_period']).mean()
                data['Price_Slope'] = data['Close'].diff(rsi_divergence_params['diff_period']).rolling(rsi_divergence_params['smooth_period']).mean()
                data['RSI_Divergence'] = np.where(
                    (data['Price_Slope'] > 0) & (data['RSI_Slope'] < rsi_divergence_params['rsi_threshold']), 1,  # Bearish
                    np.where((data['Price_Slope'] < 0) & (data['RSI_Slope'] > -rsi_divergence_params['rsi_threshold']), -1, 0)  # Bullish
                )
                
                # Add lagged returns
                data['Lag1_Return'] = data['Close'].pct_change(1)
                data['Lag3_Return'] = data['Close'].pct_change(3)
                data['Lag5_Return'] = data['Close'].pct_change(5)
                
                # Add skewness and kurtosis
                data['Skewness'] = data['Returns'].rolling(window=20, min_periods=1).skew()
                data['Kurtosis'] = data['Returns'].rolling(window=20, min_periods=1).kurt()
                return data.iloc[-1][FEATURE_COLUMNS]

            # Apply indicators row by row for each window
            processed_data = []
            for i in range(len(data)):
                window = data.iloc[max(0, i - max(70, rsi_period, bb_window, volatility_period)):i + 1]
                if len(window) >= min(rsi_period, bb_window, volatility_period):
                    processed_row = compute_rolling_indicators(window)
                    processed_data.append(processed_row)
            
            data = pd.DataFrame(processed_data, index=data.index[-len(processed_data):])

            Utils.log_message(f"INFO: Input data: {len(df)} rows")
            data = data[FEATURE_COLUMNS]
            data = data.dropna()

            Utils.log_message(f"INFO: Preprocessed data: {len(data)} rows, columns: {data.columns.tolist()}")

            return data
        except Exception as e:
            Utils.log_message(f"ERROR: Error preprocessing data: {e}")
            raise RuntimeError(f"Error preprocessing data: {e}")


    @staticmethod
    def display_action(action, q_values, placeholder):
        with placeholder.container():
            st.write("**Action Taken**")
            action_map = {0: "Hold", 1: "Buy", 2: "Sell"}
            action_icons=["✋🏻", "🛒", "💰"]
            action_text = action_map.get(action, "Unknown")
            action_message=f"Action: {action_text} (Q-Values: Hold={q_values[0]:.2f}, Buy={q_values[1]:.2f}, Sell={q_values[2]:.2f})"
            if action_text == "Hold":
                st.error(action_message)
            else:
                st.success(action_message)
            st.toast(action_message, icon=action_icons[action])


    @staticmethod
    def check_and_restore_settings(agent, current_settings, comparison_keys, context=""):
        loaded_settings = agent.get_configuration_settings().copy() if agent else None

        if loaded_settings is None:
            Utils.log_message(f"INFO: No agent settings provided for {context} comparison")
            return True

        if current_settings is None:
            Utils.log_message(f"WARNING: No current settings found in st.session_state.user_settings for {context}")
            return False

        mismatches = {}
        for key in comparison_keys:
            agent_value = loaded_settings.get(key, None)
            current_value = current_settings.get(key, None)

            if key in ['training_start_date', 'training_end_date']:
                agent_value = pd.Timestamp(agent_value).date() if pd.notna(agent_value) else agent_value
                current_value = pd.Timestamp(current_value).date() if pd.notna(current_value) else current_value

            if key in ['model_dir']:
                agent_value = agent_value.replace(f"\\{agent.ticker}", "")

            if key in ['reference_capital', 'capital_percentage'] and loaded_settings.get('capital_type') == 'Explicit Value':
                continue

            if agent_value != current_value:
                mismatches.update({key: { f"{agent.ticker} Agent": agent_value, 'Current': current_value}})

        if mismatches:
            Utils.log_message(f"INFO: Settings mismatch detected in {context}: {mismatches}")
            with st.container(border=True):
                with st.expander(f"{context}: Settings mismatch between trained agent and current settings"):
                    st.dataframe(pd.DataFrame(mismatches).T.astype(str))
                cols=st.columns(2)
                if cols[0].button("Restore Agent's Settings", key=f"restore_checkpoint_{context.lower().replace(' ', '_')}", type="primary", use_container_width=True):
                    Utils.log_message(f"INFO: Restoring Agent's settings for {context}: {st.session_state.user_settings}")
                    current_settings.update(loaded_settings)
                    st.session_state.user_settings = current_settings
                    st.rerun()
                if cols[1].button("Use Current Settings", key=f"keep_current_{context.lower().replace(' ', '_')}", use_container_width=True):
                    Utils.log_message(f"INFO: Keeping current settings for {context}: {st.session_state.user_settings}")
                    if st.session_state.agent:
                        st.session_state.agent.update_settings(current_settings)
                    st.session_state.agent = None
                    st.rerun()
                return False
        return True


    @staticmethod
    def display_timer_until_rerun(wait_time, text="", placeholder=None):
        import time
        counter = 0
        if placeholder:
            placeholder.info(f"***{text} {int(wait_time-counter)} seconds...***")
        placeholder_toast=st.toast(f"***{text} {int(wait_time-counter)} seconds...***", icon='⌛')
        while True:
            time.sleep(1)
            counter+=1
            if counter >= wait_time:
                break
            if placeholder:
                placeholder.info(f"***{text} {int(wait_time-counter)} seconds...***")
            placeholder_toast.toast(f"***{text} {int(wait_time-counter)} seconds...***", icon='⌛')
        st.rerun()


    @staticmethod
    def perform_using_retries(delegate, max_attempts=3, delay=1):
        import time
        for attempt in range(max_attempts):
            try:
                return delegate()
            except Exception as e:
                Utils.log_message(f"ERROR: Attempt {attempt + 1} failed with error: {str(e)}")
                if attempt < max_attempts - 1:
                    time.sleep(delay)
                    delay *= 2
                else:
                    Utils.log_message(f"ERROR: Max retries reached. Raising exception: '{str(e)}'")
                    raise e


    @staticmethod
    def fetch_data(ticker, start_date, end_date):
        import os
        from datetime import datetime
        current_date = datetime.now().date()

        try:
            if start_date is None or end_date is None:
                Utils.log_message(f"ERROR: Start date or end date is None: training_start_date={start_date}, training_end_date={end_date}")
                return pd.DataFrame()

            if isinstance(start_date, str):
                start_date = pd.Timestamp(start_date).date()

            if isinstance(end_date, str):
                end_date = pd.Timestamp(end_date).date()
        except ValueError as e:
            Utils.log_message(f"ERROR: Invalid date format: start_date={start_date}, training_end_date={end_date}, error={e}")
            return pd.DataFrame()

        if end_date > current_date:
            Utils.log_message(f"WARNING: End date {end_date} is in the future. Setting to {current_date}")
            end_date = current_date

        if start_date >= end_date:
            Utils.log_message(f"ERROR: Invalid date range: start_date {start_date} >= training_end_date {end_date}")
            return pd.DataFrame()

        def get_from_local(ticker, start_date, end_date):
            data_dir = "data"
            os.makedirs(data_dir, exist_ok=True)

            # Check local cache
            filename = f"{ticker}_{start_date}_{end_date}.csv"
            full_path = os.path.join(data_dir, filename)
            if os.path.exists(full_path):
                try:
                    df = pd.read_csv(full_path, index_col=0, parse_dates=True)
                    # Validate required columns
                    if df.empty or not all(col in df.columns for col in REQUIRED_DATA_COLUMNS):
                        Utils.log_message(f"WARNING: Cached file {full_path} is empty or missing required columns {REQUIRED_DATA_COLUMNS}; ignoring")
                        return None
                    # Ensure no MultiIndex remnants
                    if isinstance(df.columns, pd.MultiIndex):
                        Utils.log_message(f"WARNING: Cached file {full_path} has MultiIndex columns; attempting to normalize")
                        df = df.xs(ticker, level=1, axis=1, drop_level=True) if ticker in df.columns.levels[1] else df
                        df.columns = [col.split()[-1] for col in df.columns]
                    Utils.log_message(f"INFO: Loading data from cache: {full_path}")
                    return df
                except Exception as e:
                    Utils.log_message(f"ERROR: Failed to load cached file {full_path}: {e}")
                    return None

            # Check if any existing file fully contains the range
            for file in os.listdir(data_dir):
                if not file.endswith(".csv") or not file.startswith(ticker + "_"):
                    continue
                try:
                    parts = file[:-4].split("_")
                    file_start = pd.to_datetime(parts[1]).date()
                    file_end = pd.to_datetime(parts[2]).date()
                    if file_start <= start_date and file_end >= end_date:
                        full_path = os.path.join(data_dir, file)
                        df = pd.read_csv(full_path, index_col=0, parse_dates=True)
                        if df.empty or not all(col in df.columns for col in REQUIRED_DATA_COLUMNS):
                            Utils.log_message(f"WARNING: Broader cached file {full_path} is empty or missing required columns; ignoring")
                            continue
                        if isinstance(df.columns, pd.MultiIndex):
                            Utils.log_message(f"WARNING: Broader cached file {full_path} has MultiIndex columns; attempting to normalize")
                            df = df.xs(ticker, level=1, axis=1, drop_level=True) if ticker in df.columns.levels[1] else df
                            df.columns = [col.split()[-1] for col in df.columns]
                        Utils.log_message(f"INFO: Found broader cached data: {file}")
                        df = df.loc[str(start_date):str(end_date)]
                        return df
                except Exception:
                    continue  # Skip malformed filenames

            return None

        def download_from_yfinance(ticker, start_date, end_date):
            import yfinance as yf

            data = yf.download(ticker, start=start_date, end=end_date, progress=True, auto_adjust=False)

            if data is None or data.empty or len(data) == 0:
                msg=f"ERROR: Data could not be obtained form yfinance for ticker {ticker} from {start_date} to {end_date}"
                Utils.log_message(msg)
                raise ValueError(msg)

            return data

        def download_from_pandas_datareader(ticker, start_date, end_date):
            import pandas_datareader as pdr
            data = pdr.get_data_stooq(ticker, start=start_date, end=end_date)

            if data is None or data.empty or len(data) == 0:
                msg=f"ERROR: Data could not be obtained from pandas_datareader for ticker {ticker} from {start_date} to {end_date}"
                Utils.log_message(msg)
                raise ValueError(msg)

            return data

        def save_data_to_csv(data, ticker, start_date, end_date):
            if data is None or data.empty or len(data) == 0:
                Utils.log_message(f"ERROR: Attempted to save empty data for {ticker} from {start_date} to {end_date}")
                return
            # Normalize DataFrame structure
            try:
                # Handle MultiIndex columns (from Stooq)
                if isinstance(data.columns, pd.MultiIndex):
                    Utils.log_message(f"INFO: Normalizing MultiIndex columns for {ticker}")
                    if ticker in data.columns.levels[1]:
                        data = data.xs(ticker, level=1, axis=1, drop_level=True)
                    else:
                        # Flatten columns by taking the last part of the name
                        data.columns = [col[-1] for col in data.columns]
                # Ensure standard column names
                data = data.rename(columns=lambda x: x.strip().capitalize())
                # Select only required columns, dropping extras like 'Adj Close'
                data = data[[col for col in REQUIRED_DATA_COLUMNS if col in data.columns]]
                if not all(col in data.columns for col in REQUIRED_DATA_COLUMNS):
                    Utils.log_message(f"ERROR: Normalized data for {ticker} missing required columns {REQUIRED_DATA_COLUMNS}")
                    return
                # Ensure Date is the index
                if data.index.name != 'Date':
                    if 'Date' in data.columns:
                        data = data.set_index('Date')
                    else:
                        data.index.name = 'Date'
                data.index = pd.to_datetime(data.index)
            except Exception as e:
                Utils.log_message(f"ERROR: Failed to normalize data for {ticker} before saving: {e}")
                return
            # Save to CSV
            data_dir = "data"
            os.makedirs(data_dir, exist_ok=True)
            filename = f"{ticker}_{start_date}_{end_date}.csv"
            full_path = os.path.join(data_dir, filename)
            data.to_csv(full_path)
            Utils.log_message(f"INFO: Saved data to {full_path}")

        try:
            data = get_from_local(ticker, start_date, end_date)

            if data is None or data.empty or len(data) == 0:
                data = Utils.perform_using_retries(lambda: download_from_pandas_datareader(ticker, start_date, end_date))
        except:
            data = Utils.perform_using_retries(lambda: download_from_yfinance(ticker, start_date, end_date))

        if data is not None and not data.empty:
            save_data_to_csv(data, ticker, start_date, end_date)

        return data


    @staticmethod
    def log_message(msg, ui_output=False, console_output=False, file_output=True, toast_output=False):
        if ui_output:
            systems_logs = st.session_state.setdefault('system_logs', [])
            systems_logs.append(msg)

        if console_output:
            print(msg)

        if toast_output:
            _, icon = next(
                ((identifier, icon) for identifier, icon in LOG_HANDLERS.items() if msg.startswith(identifier)),
                DEFAULT_HEADER
            )
            st.toast(msg, icon=icon)

        if file_output:
            for prefix, method in LOG_LEVEL_METHODS.items():
                if msg.startswith(prefix):
                    log_method = method
                    log_msg = msg
                    break
            else:
                log_method = DEFAULT_LOG_METHOD
                log_msg = msg

            log_method(log_msg)

    @staticmethod
    def load_system_logs_from_file(keep_timestamps=False):
        def swap_timestamp_prefix(log_line):
            pattern = r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (\w+:) (.*)$"
            match = re.match(pattern, log_line)
            if match:
                timestamp, prefix, message = match.groups()
                return f"{prefix} {timestamp} {message}"
            else:
                return log_line

        def remove_timestamp(log_line):
            pattern = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} (.*)$"
            match = re.match(pattern, log_line)
            if match:
                return match.group(1)
            else:
                return log_line

        logs = []
        if os.path.exists(LOG_FILE_PATH):
            with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if keep_timestamps:
                        line = swap_timestamp_prefix(line)
                    else:
                        line = remove_timestamp(line)
                    logs.append(line)
        return logs

    @staticmethod
    def clear_log_file():
        # Clear the content of the log file if it exists
        if os.path.exists(LOG_FILE_PATH):
            with open(LOG_FILE_PATH, "w", encoding="utf-8") as f:
                pass  # Just opening in 'w' mode truncates the file
