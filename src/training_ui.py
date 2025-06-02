import os
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import sys
sys.dont_write_bytecode = True

import streamlit as st
import time
import json

from chart_builder import ChartBuilder
from utils import Utils, SETTING_KEYS

def _inform_progress_callback(progress, params=None, mean_cv_score=None, best_params=None, best_score=None, training_phase=None, pre_smote_label_distribution=None, post_smote_label_distribution=None, override_status_message=None):
    if 'progress_bar' in st.session_state and 'status_text' in st.session_state:
        st.session_state.progress_bar.progress(min(progress, 1.0))
        if override_status_message:
            status = override_status_message
        else:
            status = f"RF Tuning Progress: {progress*100:.1f}%..."
            if best_score is not None and 'rf_metrics' in st.session_state:
                with st.session_state.rf_metrics.container():
                    st.write("**RF Agent Performance Metrics**")
                    if progress < 1.0:
                        cols = st.columns(6)
                        cols[0].metric("RF Last N Estimators", params['n_estimators'])
                        cols[1].metric("RF Last Max Depth", params['max_depth'])
                        cols[2].metric("RF Last Min Sample Split", params['min_samples_split'])
                        cols[3].metric("RF Last Min Sample Leaf", params['min_samples_leaf'])
                        cols[4].metric("RF Last Max Features", best_params['max_features'])
                        cols[5].metric("RF Last Mean CV Score", f"{mean_cv_score:.4f}", delta=f"{(mean_cv_score - best_score):.4f}")
                    if pre_smote_label_distribution and post_smote_label_distribution:
                        pre_dist = f"Hold={pre_smote_label_distribution.get(0, 0)}, Buy={pre_smote_label_distribution.get(1, 0)}, Sell={pre_smote_label_distribution.get(2, 0)}"
                        post_dist = f"Hold={post_smote_label_distribution.get(0, 0)}, Buy={post_smote_label_distribution.get(1, 0)}, Sell={post_smote_label_distribution.get(2, 0)}"
                        if pre_dist == post_dist:
                            st.write(f"**Label Distribution:** {pre_dist} -> SMOTE: Not applied!")
                        else:
                            st.write(f"**Label Distribution:** Pre-SMOTE: {pre_dist} -> Post-SMOTE: {post_dist}")
                    elif pre_smote_label_distribution:
                        st.write(f"**Label Distribution: Hold={pre_smote_label_distribution.get(0, 0)}, Buy={pre_smote_label_distribution.get(1, 0)}, Sell={pre_smote_label_distribution.get(2, 0)}")
                    best_cols = st.columns(6)
                    best_cols[0].metric("RF Best N Estimators", best_params['n_estimators'])
                    best_cols[1].metric("RF Best Max Depth", best_params['max_depth'])
                    best_cols[2].metric("RF Best Min Sample Split", best_params['min_samples_split'])
                    best_cols[3].metric("RF Best Min Sample Leaf", best_params['min_samples_leaf'])
                    best_cols[4].metric("RF Best Max Features", best_params['max_features'])
                    best_cols[5].metric("RF Best Accuracy Score", f"{best_score:.4f}", delta=f"{(best_score - mean_cv_score):.4f}")
                    st.divider()
                # Store RF metrics
                if 'rf_training_metrics' not in st.session_state:
                    st.session_state.rf_training_metrics = []
                st.session_state.rf_training_metrics.append({
                    "Iteration": len(st.session_state.rf_training_metrics) + 1,
                    "Mean CV Score": mean_cv_score,
                    "Best Score": best_score,
                    "N Estimators": params['n_estimators']
                })
            if training_phase and training_phase == "rf":
                st.session_state.rf_training_active = True
            else:
                st.session_state.rf_training_active = False
        st.session_state.status_text.text(status)

def _display_stored_ensemble_metrics(agent, placeholder):
    with placeholder:
        if os.path.exists(agent.metrics_save_path):
            try:
                with open(agent.metrics_save_path, "r") as f:
                    metrics_json = json.load(f)
                st.write("**Ensemble Performance Metrics**")
                cols = st.columns(6)
                cols[0].metric("Best Sharpe Ratio", f"{metrics_json.get('best_sharpe', 0.0):.4f}")
                cols[1].metric("Last Drawdown", f"€{metrics_json.get('last_drawdown', 0.0):.2f}")
                cols[2].metric("Total Trades", f"{int(metrics_json.get('total_trades', 0))}")
                cols[3].metric("DQN Temperature", f"{float(metrics_json.get('dqn_temperature', 0)):.3f}")
                cols[4].metric("RF Best Score", f"{float(metrics_json.get('rf_best_score', 0)):.4f}")
                cols[5].metric("RF N PCA Components", f"{int(metrics_json.get('rf_n_pca_components', 0))}")
                st.divider()
            except (json.JSONDecodeError, OSError) as e:
                Utils.log_message(f"ERROR: Failed to load metrics from {agent.metrics_save_path}: {e}")
                st.warning("Unable to load ensemble metrics.")

def _display_epoch_metrics(epoch_metrics_dict, placeholder, dqn_temperature):
    with placeholder.container():
        st.write("**Training Metrics**")
        cols = st.columns(6)
        cols[0].metric(f"Sharpe Ratio (Epoch {epoch_metrics_dict['Epoch']})", epoch_metrics_dict["Sharpe Ratio"], delta=f"{(float(epoch_metrics_dict['Sharpe Ratio']) - float(epoch_metrics_dict['Best Sharpe Ratio'])):.4f}")
        cols[1].metric("Max Drawdown", f"€{epoch_metrics_dict['Max Drawdown (€)']}")
        cols[2].metric("Cumulative Fees", f"€{epoch_metrics_dict['Cumulative Fees (€)']}")
        cols[3].metric("Best Sharpe Ratio", epoch_metrics_dict["Best Sharpe Ratio"], delta=f"{(float(epoch_metrics_dict['Best Sharpe Ratio']) - float(epoch_metrics_dict['Sharpe Ratio'])):.4f}")
        cols[4].metric("DQN Temperature", f"{dqn_temperature:.3f}")
        cols[5].metric("Epoch Fees", f"€{epoch_metrics_dict['Epoch Cumulative Fees (€)']}")
        st.divider()

def _display_metrics(agent, portfolio_values, portfolio_chart_placeholder, ensemble_metrics_placeholder, epoch_metrics, metrics_table_placeholder, epoch_metrics_chart_placeholder):
    _display_stored_ensemble_metrics(agent, ensemble_metrics_placeholder)
    if epoch_metrics and len(epoch_metrics) > 0:
        _display_epoch_metrics(epoch_metrics[-1], ensemble_metrics_placeholder, agent.dqn_agent.temperature)
    ChartBuilder.plot_training_metrics(epoch_metrics, metrics_table_placeholder, epoch_metrics_chart_placeholder)
    ChartBuilder.plot_portfolio_over_time(portfolio_values, portfolio_chart_placeholder)

def _handle_training_failed(agent, e, epoch, epoch_metrics, portfolio_values):
    error_msg = f"Training failed at epoch {epoch + 1}: {str(e)}"
    
    with st.container(border=True):
        agent_type = "RF" if "rf_agent" in str(e).lower() or "randomforest" in str(e).lower() else "DQN"
        Utils.log_message(f"ERROR: {error_msg} (Agent: {agent_type})")
        with st.container():
            st.error(f"{error_msg} (Agent: {agent_type})")
            with st.expander("Error Details and Recent Logs"):
                st.write(f"**Error Message**: {str(e)}")
                st.write(f"**Agent**: {agent_type}")
                recent_logs = Utils.load_system_logs_from_file(keep_timestamps=True)[-20:]
                st.write("**Recent Logs**:")
                for log in recent_logs:
                    st.text(log)
            col1, col2 = st.columns(2)
            if col1.button("Retry Epoch", key=f"retry_epoch_{epoch}", help="Retry the failed epoch", use_container_width=True):
                Utils.log_message(f"INFO: Retrying epoch {epoch + 1}")
                st.session_state.training_running = True
                st.session_state.training_paused = False
                # continue
            if col2.button("Stop Training", key=f"stop_epoch_{epoch}", help="Stop training and save checkpoint", use_container_width=True):
                st.session_state.training_running = False
                st.session_state.training_paused = False
                agent.save_checkpoint(
                    epoch,
                    agent.dqn_agent.epsilon,
                    epoch_metrics,
                    portfolio_values,
                    st.session_state.user_settings
                )
                st.success("Training stopped and checkpoint saved.")
                st.rerun()

def _handle_training(agent, settings, checkpoint_info, start_epoch, epoch_metrics, portfolio_values, ensemble_metrics, metrics_table_placeholder, epoch_metrics_chart, portfolio_chart):
    if st.session_state.training_running and not st.session_state.training_paused:
        epoch_metrics = epoch_metrics.copy() if epoch_metrics else []
        portfolio_values = portfolio_values.copy() if portfolio_values else []

        _display_metrics(agent, portfolio_values, portfolio_chart, ensemble_metrics.container(), epoch_metrics, metrics_table_placeholder, epoch_metrics_chart)

        for epoch in range(start_epoch, settings['epochs']):
            if st.session_state.training_paused:
                st.session_state.status_text.text(f"Training paused at epoch {epoch + 1}...")
                break

            status_message = f"Resuming from epoch {epoch + 1} of {settings['epochs']}..." if epoch == start_epoch and checkpoint_info else f"Processing epoch {epoch + 1} of {settings['epochs']}..."
            if not st.session_state.get('rf_training_active', False):
                st.session_state.status_text.text(status_message)
            st.session_state.progress_bar.progress(((epoch + 1) if epoch != start_epoch else epoch) / settings['epochs'])

            start_time = time.time()
            try:
                epoch_values, additional_metrics, epoch_metrics_dict = agent.train(
                    current_epoch=epoch,
                    max_epochs=settings['epochs'],
                    portfolio_values=portfolio_values,
                    progress_callback=_inform_progress_callback
                )
            except Exception as e:
                _handle_training_failed(agent, e, epoch, epoch_metrics, portfolio_values)
                raise e
            end_time = time.time()
            epoch_duration = end_time - start_time

            if epoch == start_epoch:
                portfolio_values = epoch_values
            else:
                portfolio_values.extend(epoch_values)

            st.session_state.progress_bar.progress(((epoch + 1) if epoch != start_epoch else epoch) / settings['epochs'])
            if len(portfolio_values) > 0:
                epoch_metrics_dict["Time Taken (s)"] = f"{epoch_duration:.2f}s"
                epoch_metrics.append(epoch_metrics_dict)
                _display_metrics(agent, portfolio_values, portfolio_chart, ensemble_metrics.container(), epoch_metrics, metrics_table_placeholder, epoch_metrics_chart)

            agent.save_checkpoint(
                epoch + 1,
                agent.dqn_agent.epsilon,
                epoch_metrics,
                portfolio_values,
                st.session_state.user_settings
            )

            if not st.session_state.training_running:
                break

        if not st.session_state.training_paused and st.session_state.training_running:
            # st.session_state.progress_bar.progress(1.0)
            agent.clear_checkpoints()
            st.session_state.training_running = False
            st.session_state.rf_training_active = False
            st.session_state.status_text.success("Training completed successfully!")
            st.session_state.pop('progress_bar', None)
            st.session_state.pop('status_text', None)
            st.session_state.pop('rf_metrics', None)

    elif st.session_state.training_running and st.session_state.training_paused:
        st.session_state.status_text.info(f"Training paused on epoch {start_epoch}. Click 'Resume Training' to continue.")
        _display_metrics(agent, portfolio_values, portfolio_chart, ensemble_metrics.container(), epoch_metrics, metrics_table_placeholder, epoch_metrics_chart)

def _build_ui_controls(agent, settings, checkpoint_info, start_epoch, epoch_metrics, portfolio_values):
    has_checkpoint = checkpoint_info is not None
    
    col1, col2 = st.columns(2)
    if st.session_state.training_running and not st.session_state.training_paused:
        train_button_label = "Pause Training"
        train_button_help = "Pause the current training session and save a checkpoint."
    elif st.session_state.training_paused or has_checkpoint:
        train_button_label = "Resume Training"
        train_button_help = "Resume training from the last checkpoint."
    else:
        train_button_label = "Start Training"
        train_button_help = "Start a new training session."
    train_pause_button = col1.button(train_button_label, help=train_button_help, use_container_width=True)
    stop_button = col2.button(
        "Stop Training",
        disabled=not (st.session_state.training_running or has_checkpoint),
        help="Stop training and clear all checkpoints.", use_container_width=True
    )
    
    with st.spinner("Training in progress..."):
        st.session_state.progress_bar = st.empty()
        st.session_state.status_text = st.empty()
        
        st.divider()
        
        st.session_state.rf_metrics = st.empty()
        st.session_state.rf_training_active = False
        ensemble_metrics = st.empty()
        metrics_table_placeholder = st.empty()
        epoch_metrics_chart_placeholder = st.empty()
        portfolio_chart_placeholder = st.empty()

        if train_pause_button:
            if st.session_state.training_running and not st.session_state.training_paused:
                st.session_state.training_paused = True
                agent.save_checkpoint(
                    start_epoch,
                    agent.dqn_agent.epsilon,
                    epoch_metrics,
                    portfolio_values,
                    st.session_state.user_settings
                )
                Utils.log_message(f"INFO: Training paused by user.")
                st.rerun()
            else:
                st.session_state.training_running = True
                st.session_state.training_paused = False
                st.session_state.rf_training_active = False
                if not has_checkpoint:
                    start_epoch = 0
                    epoch_metrics = []
                    portfolio_values = []
                    Utils.log_message(f"INFO: Starting new training session.")
                else:
                    checkpoint_info = agent.load_checkpoint(settings)
                    if checkpoint_info:
                        start_epoch, _, epoch_metrics, portfolio_values, checkpoint_settings = checkpoint_info
                        Utils.log_message(f"INFO: Resuming training from epoch {start_epoch  + 1}.")

                st.session_state.progress_bar.progress(0.0)
                st.session_state.status_text.text("Starting RF training...")
                st.rerun()

        if stop_button:
            st.session_state.training_running = False
            st.session_state.training_paused = False
            st.session_state.rf_training_active = False
            agent.clear_checkpoints()
            start_epoch = 0
            epoch_metrics = []
            portfolio_values = []
            st.session_state.pop('checkpoint_settings', None)
            st.session_state.pop('progress_bar', None)
            st.session_state.pop('status_text', None)
            st.session_state.pop('rf_metrics', None)
            Utils.log_message(f"INFO: Training stopped and checkpoints cleared.")
            st.success("Training stopped and checkpoints cleared!")
            st.rerun()

        _handle_training(agent, settings, checkpoint_info, start_epoch, epoch_metrics, portfolio_values, ensemble_metrics, metrics_table_placeholder, epoch_metrics_chart_placeholder, portfolio_chart_placeholder)

def _load_checkpoints(agent, settings):
    checkpoint_info = agent.load_checkpoint(settings)
    training_settings = None
    if checkpoint_info:
        start_epoch, _, epoch_metrics, portfolio_values, training_settings = checkpoint_info
        st.session_state.user_settings = training_settings
        st.warning(f"Resuming training from epoch {start_epoch} for {agent.ticker}.")
    else:
        start_epoch = 0
        epoch_metrics = []
        portfolio_values = []
        training_settings = settings
        Utils.log_message(f"INFO: No checkpoint found, starting fresh.")
    
    keys_to_compare = SETTING_KEYS.copy()
    Utils.check_and_restore_settings(agent, training_settings, comparison_keys=keys_to_compare, context="Training")
    
    return checkpoint_info, start_epoch, epoch_metrics, portfolio_values

def render_training(agent, settings):
    cols = st.columns([2, 3])
    with cols[0]:
        st.header("Model Training")
    with cols[1]:
        if agent.has_pretrained_model():
            st.success(f"Loaded pre-trained ensemble model for {agent.ticker} with best Sharpe Ratio of {agent.best_sharpe:.4f}.")
            if not hasattr(agent.rf_agent.scaler, 'mean_'):
                st.warning("RF Agent scaler is not fitted. Please start training to fit the model.")
        checkpoint_info, start_epoch, epoch_metrics, portfolio_values = _load_checkpoints(agent, settings)

    if agent:
        with st.expander(f"Agent's Training Dataset - {agent.ticker} ({agent.training_start_date} to {agent.training_end_date})"):
            tabs = st.tabs(["DataFrame", "Correlation Heatmap", "Indicator Signals", "Q-Q Plot", "Feature Distributions with KDE", "Indicator Time Series", "Feature Variability", 
                            "RF Feature Importance", "Rolling Volatility", "Pair Scatter"])
            with tabs[0]:
                st.dataframe(agent.training_data)
            with tabs[1]:
                ChartBuilder.plot_correlation_heatmap(agent.training_data, context="training")
            with tabs[2]:
                ChartBuilder.plot_indicator_signals_heatmap(agent.training_data, context="training")
            with tabs[3]:
                ChartBuilder.plot_qq_plot(agent.training_data, context="training")
            with tabs[4]:
                ChartBuilder.plot_feature_distribution_with_kde(agent.training_data, context="training")
            with tabs[5]:
                ChartBuilder.plot_indicator_timeseries(agent.training_data, context="training")
            with tabs[6]:
                ChartBuilder.plot_feature_boxplots(agent.training_data, context="training")
            with tabs[7]:
                ChartBuilder.plot_rf_feature_importance(agent, context="training")
            with tabs[8]:
                ChartBuilder.plot_rolling_volatility(agent.training_data, context="training")
            with tabs[9]:
                ChartBuilder.plot_pair_scatter(agent.training_data, context="training")

    _build_ui_controls(agent, settings, checkpoint_info, start_epoch, epoch_metrics, portfolio_values)
