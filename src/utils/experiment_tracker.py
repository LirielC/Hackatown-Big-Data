"""
Experiment tracking utilities using MLflow for the hackathon forecast model.
"""

import os
import json
import pickle
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from typing import Dict, Any, Optional, List, Union
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """MLflow-based experiment tracking for forecast models."""
    
    def __init__(self, experiment_name: str = "hackathon-forecast-2025", 
                 tracking_uri: Optional[str] = None):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking URI (defaults to local ./mlruns)
        """
        self.experiment_name = experiment_name
        
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            # Use local mlruns directory
            mlflow.set_tracking_uri("file:./mlruns")
        
        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        except mlflow.exceptions.MlflowException:
            # Experiment already exists
            experiment = mlflow.get_experiment_by_name(experiment_name)
            self.experiment_id = experiment.experiment_id
        
        mlflow.set_experiment(experiment_name)
        logger.info(f"Initialized experiment tracker for: {experiment_name}")
    
    def start_run(self, run_name: Optional[str] = None, 
                  nested: bool = False) -> str:
        """
        Start a new MLflow run.
        
        Args:
            run_name: Optional name for the run
            nested: Whether this is a nested run
            
        Returns:
            Run ID
        """
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        mlflow.start_run(run_name=run_name, nested=nested)
        run_id = mlflow.active_run().info.run_id
        logger.info(f"Started MLflow run: {run_name} (ID: {run_id})")
        return run_id
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to MLflow."""
        for key, value in params.items():
            # Convert complex objects to strings
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            elif not isinstance(value, (str, int, float, bool)):
                value = str(value)
            mlflow.log_param(key, value)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to MLflow."""
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
    
    def log_model(self, model: Any, model_name: str, 
                  model_type: str = "sklearn") -> str:
        """
        Log model to MLflow with automatic type detection.
        
        Args:
            model: Trained model object
            model_name: Name for the model
            model_type: Type of model (sklearn, xgboost, lightgbm, custom)
            
        Returns:
            Model URI
        """
        if model_type == "xgboost":
            model_info = mlflow.xgboost.log_model(model, model_name)
        elif model_type == "lightgbm":
            model_info = mlflow.lightgbm.log_model(model, model_name)
        elif model_type == "sklearn":
            model_info = mlflow.sklearn.log_model(model, model_name)
        else:
            # Custom model - use pickle
            model_info = mlflow.sklearn.log_model(model, model_name)
        
        logger.info(f"Logged {model_type} model: {model_name}")
        return model_info.model_uri
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log artifact to MLflow."""
        mlflow.log_artifact(local_path, artifact_path)
    
    def log_dict(self, dictionary: Dict[str, Any], artifact_file: str) -> None:
        """Log dictionary as JSON artifact."""
        mlflow.log_dict(dictionary, artifact_file)
    
    def log_dataframe(self, df: pd.DataFrame, artifact_file: str) -> None:
        """Log pandas DataFrame as CSV artifact."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        df.to_csv(temp_path, index=False)
        mlflow.log_artifact(temp_path, artifact_file)
        os.remove(temp_path)
    
    def log_feature_importance(self, feature_names: List[str], 
                             importance_values: List[float],
                             importance_type: str = "gain") -> None:
        """Log feature importance as both metrics and artifact."""
        # Log top 10 features as metrics
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_values
        }).sort_values('importance', ascending=False)
        
        for i, row in importance_df.head(10).iterrows():
            mlflow.log_metric(f"feature_importance_{row['feature']}", row['importance'])
        
        # Log full importance as artifact
        self.log_dict(importance_df.to_dict('records'), 
                     f"feature_importance_{importance_type}.json")
    
    def log_validation_results(self, cv_results: Dict[str, Any]) -> None:
        """Log cross-validation results."""
        # Log mean and std of CV scores
        for metric, scores in cv_results.items():
            if isinstance(scores, (list, np.ndarray)):
                mlflow.log_metric(f"cv_{metric}_mean", np.mean(scores))
                mlflow.log_metric(f"cv_{metric}_std", np.std(scores))
        
        # Log full CV results as artifact
        self.log_dict(cv_results, "cv_results.json")
    
    def log_predictions(self, y_true: np.ndarray, y_pred: np.ndarray,
                       dataset_name: str = "validation") -> None:
        """Log predictions vs actual values."""
        predictions_df = pd.DataFrame({
            'actual': y_true,
            'predicted': y_pred,
            'residual': y_true - y_pred
        })
        
        self.log_dataframe(predictions_df, f"predictions_{dataset_name}.csv")
        
        # Log prediction statistics
        mlflow.log_metric(f"{dataset_name}_mean_residual", predictions_df['residual'].mean())
        mlflow.log_metric(f"{dataset_name}_std_residual", predictions_df['residual'].std())
    
    def end_run(self) -> None:
        """End the current MLflow run."""
        mlflow.end_run()
        logger.info("Ended MLflow run")
    
    def get_experiment_runs(self) -> pd.DataFrame:
        """Get all runs from the current experiment."""
        runs = mlflow.search_runs(experiment_ids=[self.experiment_id])
        return runs
    
    def compare_runs(self, run_ids: List[str], 
                    metrics: List[str]) -> pd.DataFrame:
        """Compare specific runs on given metrics."""
        runs_data = []
        
        for run_id in run_ids:
            run = mlflow.get_run(run_id)
            run_data = {'run_id': run_id, 'run_name': run.info.run_name}
            
            for metric in metrics:
                run_data[metric] = run.data.metrics.get(metric, None)
            
            runs_data.append(run_data)
        
        return pd.DataFrame(runs_data)
    
    def get_best_run(self, metric: str, ascending: bool = False) -> mlflow.entities.Run:
        """Get the best run based on a specific metric."""
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"]
        )
        
        if len(runs) == 0:
            raise ValueError("No runs found in experiment")
        
        best_run_id = runs.iloc[0]['run_id']
        return mlflow.get_run(best_run_id)
    
    def load_model(self, run_id: str, model_name: str) -> Any:
        """Load a model from a specific run."""
        model_uri = f"runs:/{run_id}/{model_name}"
        return mlflow.sklearn.load_model(model_uri)
    
    def register_model(self, run_id: str, model_name: str, 
                      registered_model_name: str) -> None:
        """Register a model to MLflow Model Registry."""
        model_uri = f"runs:/{run_id}/{model_name}"
        mlflow.register_model(model_uri, registered_model_name)
        logger.info(f"Registered model: {registered_model_name}")


class ModelVersionManager:
    """Manage model versions and deployment."""
    
    def __init__(self, model_name: str):
        """
        Initialize model version manager.
        
        Args:
            model_name: Name of the registered model
        """
        self.model_name = model_name
        self.client = mlflow.tracking.MlflowClient()
    
    def create_model_version(self, run_id: str, model_path: str,
                           description: Optional[str] = None) -> int:
        """Create a new model version."""
        model_uri = f"runs:/{run_id}/{model_path}"
        
        model_version = self.client.create_model_version(
            name=self.model_name,
            source=model_uri,
            description=description
        )
        
        version_number = model_version.version
        logger.info(f"Created model version {version_number} for {self.model_name}")
        return version_number
    
    def transition_model_stage(self, version: int, stage: str) -> None:
        """
        Transition model version to a specific stage.
        
        Args:
            version: Model version number
            stage: Target stage (Staging, Production, Archived)
        """
        self.client.transition_model_version_stage(
            name=self.model_name,
            version=version,
            stage=stage
        )
        logger.info(f"Transitioned {self.model_name} v{version} to {stage}")
    
    def get_latest_version(self, stage: Optional[str] = None) -> Optional[int]:
        """Get the latest model version, optionally filtered by stage."""
        versions = self.client.get_latest_versions(
            name=self.model_name,
            stages=[stage] if stage else None
        )
        
        if versions:
            return int(versions[0].version)
        return None
    
    def load_model_version(self, version: Optional[int] = None, 
                          stage: Optional[str] = None) -> Any:
        """Load a specific model version or latest from stage."""
        if version:
            model_uri = f"models:/{self.model_name}/{version}"
        elif stage:
            model_uri = f"models:/{self.model_name}/{stage}"
        else:
            model_uri = f"models:/{self.model_name}/latest"
        
        return mlflow.sklearn.load_model(model_uri)
    
    def compare_model_versions(self, versions: List[int], 
                             metric: str) -> pd.DataFrame:
        """Compare performance of different model versions."""
        comparison_data = []
        
        for version in versions:
            model_version = self.client.get_model_version(self.model_name, version)
            run_id = model_version.run_id
            run = mlflow.get_run(run_id)
            
            comparison_data.append({
                'version': version,
                'run_id': run_id,
                'stage': model_version.current_stage,
                metric: run.data.metrics.get(metric, None),
                'created_at': model_version.creation_timestamp
            })
        
        return pd.DataFrame(comparison_data)


class ExperimentComparator:
    """Compare and analyze multiple experiments."""
    
    def __init__(self, experiment_tracker: ExperimentTracker):
        """Initialize with an experiment tracker."""
        self.tracker = experiment_tracker
    
    def compare_all_runs(self, metrics: List[str]) -> pd.DataFrame:
        """Compare all runs in the experiment on specified metrics."""
        runs_df = self.tracker.get_experiment_runs()
        
        # Select relevant columns
        columns = ['run_id', 'experiment_id', 'status', 'start_time', 'end_time']
        columns.extend([f'metrics.{metric}' for metric in metrics])
        
        return runs_df[columns].copy()
    
    def get_performance_summary(self, metric: str) -> Dict[str, float]:
        """Get performance summary statistics for a metric."""
        runs_df = self.tracker.get_experiment_runs()
        metric_col = f'metrics.{metric}'
        
        if metric_col not in runs_df.columns:
            return {}
        
        metric_values = runs_df[metric_col].dropna()
        
        return {
            'count': len(metric_values),
            'mean': metric_values.mean(),
            'std': metric_values.std(),
            'min': metric_values.min(),
            'max': metric_values.max(),
            'median': metric_values.median()
        }
    
    def find_pareto_optimal_runs(self, metrics: List[str], 
                               minimize: List[bool]) -> pd.DataFrame:
        """Find Pareto optimal runs for multi-objective optimization."""
        runs_df = self.tracker.get_experiment_runs()
        
        # Extract metric values
        metric_data = []
        for _, run in runs_df.iterrows():
            run_metrics = []
            for i, metric in enumerate(metrics):
                value = run.get(f'metrics.{metric}')
                if value is None:
                    break
                # Negate if we want to minimize
                run_metrics.append(-value if minimize[i] else value)
            
            if len(run_metrics) == len(metrics):
                metric_data.append({
                    'run_id': run['run_id'],
                    'metrics': run_metrics
                })
        
        # Find Pareto optimal solutions
        pareto_runs = []
        for i, run_a in enumerate(metric_data):
            is_dominated = False
            for j, run_b in enumerate(metric_data):
                if i != j:
                    # Check if run_b dominates run_a
                    dominates = all(b >= a for a, b in zip(run_a['metrics'], run_b['metrics']))
                    strictly_better = any(b > a for a, b in zip(run_a['metrics'], run_b['metrics']))
                    
                    if dominates and strictly_better:
                        is_dominated = True
                        break
            
            if not is_dominated:
                pareto_runs.append(run_a['run_id'])
        
        return runs_df[runs_df['run_id'].isin(pareto_runs)]