"""
MLflow integration utilities for seamless experiment tracking.
"""

import os
import yaml
import mlflow
import mlflow.sklearn
from functools import wraps
from typing import Dict, Any, Optional, Callable, List
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from .experiment_tracker import ExperimentTracker, ModelVersionManager

logger = logging.getLogger(__name__)


def load_experiment_config(config_path: str = "configs/experiment_config.yaml") -> Dict[str, Any]:
    """Load experiment configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_mlflow_autolog(config: Optional[Dict[str, Any]] = None) -> None:
    """Setup MLflow autologging based on configuration."""
    if config is None:
        config = load_experiment_config()
    
    auto_log_config = config.get('experiment', {}).get('auto_log', {})
    
    if auto_log_config.get('sklearn', False):
        try:
            mlflow.sklearn.autolog()
        except Exception as e:
            logger.warning(f"MLflow sklearn autolog failed: {e}")
    
    if auto_log_config.get('xgboost', False):
        try:
            mlflow.xgboost.autolog()
        except Exception as e:
            logger.warning(f"MLflow xgboost autolog failed: {e}")
    
    if auto_log_config.get('lightgbm', False):
        try:
            mlflow.lightgbm.autolog()
        except Exception as e:
            logger.warning(f"MLflow lightgbm autolog failed: {e}")
    
    if auto_log_config.get('matplotlib', False):
        try:
            import mlflow.matplotlib
            mlflow.matplotlib.autolog()
        except (ImportError, AttributeError):
            logger.warning("MLflow matplotlib autolog not available")
    
    logger.info("MLflow autologging configured")


def mlflow_experiment(experiment_name: Optional[str] = None,
                     run_name: Optional[str] = None,
                     log_params: bool = True,
                     log_metrics: bool = True,
                     log_model: bool = True):
    """
    Decorator to automatically track experiments with MLflow.
    
    Args:
        experiment_name: Name of the experiment
        run_name: Name of the run
        log_params: Whether to log function parameters
        log_metrics: Whether to log returned metrics
        log_model: Whether to log returned model
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            config = load_experiment_config()
            
            # Initialize experiment tracker
            exp_name = experiment_name or config['experiment']['name']
            tracker = ExperimentTracker(exp_name)
            
            # Generate run name
            if run_name:
                final_run_name = run_name
            else:
                naming_config = config.get('run_naming', {})
                prefix = naming_config.get('prefix', 'run')
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                final_run_name = f"{prefix}_{timestamp}"
            
            # Start MLflow run
            run_id = tracker.start_run(final_run_name)
            
            try:
                # Log parameters if requested
                if log_params and kwargs:
                    tracker.log_params(kwargs)
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Log results based on type
                if isinstance(result, dict):
                    if log_metrics and 'metrics' in result:
                        tracker.log_metrics(result['metrics'])
                    
                    if log_model and 'model' in result:
                        model_type = result.get('model_type', 'sklearn')
                        tracker.log_model(result['model'], 'model', model_type)
                
                return result
                
            except Exception as e:
                mlflow.log_param("error", str(e))
                raise
            finally:
                tracker.end_run()
        
        return wrapper
    return decorator


class MLflowModelTracker:
    """Enhanced model tracking with MLflow integration."""
    
    def __init__(self, config_path: str = "configs/experiment_config.yaml"):
        """Initialize with configuration."""
        self.config = load_experiment_config(config_path)
        self.tracker = ExperimentTracker(self.config['experiment']['name'])
        self.version_manager = None
    
    def track_model_training(self, model_name: str, model_type: str,
                           model: Any, params: Dict[str, Any],
                           metrics: Dict[str, float],
                           feature_importance: Optional[Dict[str, float]] = None,
                           cv_results: Optional[Dict[str, Any]] = None,
                           predictions: Optional[Dict[str, np.ndarray]] = None) -> str:
        """
        Comprehensive model tracking.
        
        Args:
            model_name: Name of the model
            model_type: Type of model (xgboost, lightgbm, sklearn, etc.)
            model: Trained model object
            params: Model parameters
            metrics: Performance metrics
            feature_importance: Feature importance scores
            cv_results: Cross-validation results
            predictions: Predictions dictionary with 'y_true' and 'y_pred'
            
        Returns:
            Run ID
        """
        # Generate run name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f"{model_name}_{model_type}_{timestamp}"
        
        run_id = self.tracker.start_run(run_name)
        
        try:
            # Log parameters
            self.tracker.log_params(params)
            self.tracker.log_params({
                'model_name': model_name,
                'model_type': model_type,
                'timestamp': timestamp
            })
            
            # Log metrics
            self.tracker.log_metrics(metrics)
            
            # Log model
            model_uri = self.tracker.log_model(model, 'model', model_type)
            
            # Log feature importance
            if feature_importance:
                feature_names = list(feature_importance.keys())
                importance_values = list(feature_importance.values())
                self.tracker.log_feature_importance(feature_names, importance_values)
            
            # Log CV results
            if cv_results:
                self.tracker.log_validation_results(cv_results)
            
            # Log predictions
            if predictions and 'y_true' in predictions and 'y_pred' in predictions:
                self.tracker.log_predictions(
                    predictions['y_true'], 
                    predictions['y_pred']
                )
            
            logger.info(f"Tracked model training: {run_name} (ID: {run_id})")
            return run_id
            
        except Exception as e:
            logger.error(f"Error tracking model: {e}")
            raise
        finally:
            self.tracker.end_run()
    
    def register_best_model(self, metric: str = "wmape", 
                          minimize: bool = True) -> Optional[str]:
        """Register the best model to MLflow Model Registry."""
        try:
            best_run = self.tracker.get_best_run(metric, ascending=minimize)
            
            # Initialize version manager if not exists
            model_name = self.config['experiment']['model_registry']['base_model_name']
            if not self.version_manager:
                self.version_manager = ModelVersionManager(model_name)
            
            # Register model
            mlflow.register_model(
                f"runs:/{best_run.info.run_id}/model",
                model_name
            )
            
            logger.info(f"Registered best model (run: {best_run.info.run_id})")
            return best_run.info.run_id
            
        except Exception as e:
            logger.error(f"Error registering model: {e}")
            return None
    
    def compare_models(self, run_ids: Optional[List[str]] = None,
                      metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """Compare model performance across runs."""
        if metrics is None:
            metrics = self.config['experiment']['primary_metrics']
        
        if run_ids is None:
            # Get all runs from experiment
            runs_df = self.tracker.get_experiment_runs()
            run_ids = runs_df['run_id'].tolist()
        
        return self.tracker.compare_runs(run_ids, metrics)
    
    def get_model_leaderboard(self, metric: str = "wmape",
                            top_k: int = 10) -> pd.DataFrame:
        """Get top performing models leaderboard."""
        runs_df = self.tracker.get_experiment_runs()
        
        # Filter runs with the metric
        metric_col = f'metrics.{metric}'
        if metric_col not in runs_df.columns:
            return pd.DataFrame()
        
        # Sort by metric (assuming lower is better for most metrics)
        leaderboard = runs_df[runs_df[metric_col].notna()].copy()
        leaderboard = leaderboard.sort_values(metric_col, ascending=True)
        
        # Select relevant columns
        columns = ['run_id', 'start_time', metric_col, 'params.model_name', 'params.model_type']
        available_columns = [col for col in columns if col in leaderboard.columns]
        
        return leaderboard[available_columns].head(top_k)
    
    def export_experiment_results(self, output_path: str = "experiment_results.csv") -> None:
        """Export all experiment results to CSV."""
        runs_df = self.tracker.get_experiment_runs()
        runs_df.to_csv(output_path, index=False)
        logger.info(f"Exported experiment results to: {output_path}")


def create_experiment_report(tracker: ExperimentTracker,
                           output_path: str = "experiment_report.html") -> None:
    """Create an HTML report of experiment results."""
    runs_df = tracker.get_experiment_runs()
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Hackathon Forecast Model - Experiment Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .metric {{ font-weight: bold; color: #2e7d32; }}
        </style>
    </head>
    <body>
        <h1>Hackathon Forecast Model - Experiment Report</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Experiment Summary</h2>
        <p>Total Runs: {len(runs_df)}</p>
        <p>Experiment ID: {tracker.experiment_id}</p>
        
        <h2>Top Performing Models (by WMAPE)</h2>
        {_create_leaderboard_table(runs_df)}
        
        <h2>All Runs</h2>
        {runs_df.to_html(classes='table', table_id='all_runs')}
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Created experiment report: {output_path}")


def _create_leaderboard_table(runs_df: pd.DataFrame, metric: str = "wmape", top_k: int = 5) -> str:
    """Create HTML table for leaderboard."""
    metric_col = f'metrics.{metric}'
    
    if metric_col not in runs_df.columns:
        return "<p>No WMAPE metrics found</p>"
    
    # Get top performers
    top_runs = runs_df[runs_df[metric_col].notna()].nsmallest(top_k, metric_col)
    
    html = "<table><tr><th>Rank</th><th>Run ID</th><th>WMAPE</th><th>Model Type</th><th>Start Time</th></tr>"
    
    for i, (_, run) in enumerate(top_runs.iterrows(), 1):
        model_type = run.get('params.model_type', 'Unknown')
        wmape = f"{run[metric_col]:.4f}" if pd.notna(run[metric_col]) else "N/A"
        start_time = run['start_time'].strftime('%Y-%m-%d %H:%M') if pd.notna(run['start_time']) else "N/A"
        
        html += f"""
        <tr>
            <td>{i}</td>
            <td>{run['run_id'][:8]}...</td>
            <td class="metric">{wmape}</td>
            <td>{model_type}</td>
            <td>{start_time}</td>
        </tr>
        """
    
    html += "</table>"
    return html