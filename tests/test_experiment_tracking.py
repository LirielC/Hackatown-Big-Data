"""
Unit tests for experiment tracking functionality.
"""

import pytest
import tempfile
import shutil
import os
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import mlflow

from src.utils.experiment_tracker import ExperimentTracker, ModelVersionManager, ExperimentComparator
from src.utils.mlflow_integration import MLflowModelTracker, load_experiment_config


class TestExperimentTracker:
    """Test cases for ExperimentTracker class."""
    
    @pytest.fixture
    def temp_mlruns_dir(self):
        """Create temporary MLflow tracking directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def tracker(self, temp_mlruns_dir):
        """Create ExperimentTracker instance with temporary directory."""
        tracking_uri = f"file://{temp_mlruns_dir}"
        return ExperimentTracker("test-experiment", tracking_uri)
    
    def test_tracker_initialization(self, tracker):
        """Test tracker initialization."""
        assert tracker.experiment_name == "test-experiment"
        assert tracker.experiment_id is not None
    
    def test_start_and_end_run(self, tracker):
        """Test starting and ending MLflow runs."""
        run_id = tracker.start_run("test-run")
        assert run_id is not None
        assert mlflow.active_run() is not None
        
        tracker.end_run()
        assert mlflow.active_run() is None
    
    def test_log_params(self, tracker):
        """Test parameter logging."""
        tracker.start_run("param-test")
        
        params = {
            'learning_rate': 0.1,
            'n_estimators': 100,
            'model_type': 'xgboost',
            'nested_param': {'key': 'value'}
        }
        
        tracker.log_params(params)
        
        # Verify params are logged
        run = mlflow.active_run()
        assert run.data.params['learning_rate'] == '0.1'
        assert run.data.params['model_type'] == 'xgboost'
        
        tracker.end_run()
    
    def test_log_metrics(self, tracker):
        """Test metrics logging."""
        tracker.start_run("metrics-test")
        
        metrics = {
            'mae': 10.5,
            'rmse': 15.2,
            'wmape': 8.3
        }
        
        tracker.log_metrics(metrics)
        
        # Verify metrics are logged
        run = mlflow.active_run()
        assert abs(run.data.metrics['mae'] - 10.5) < 1e-6
        assert abs(run.data.metrics['wmape'] - 8.3) < 1e-6
        
        tracker.end_run()
    
    def test_log_model(self, tracker):
        """Test model logging."""
        from sklearn.linear_model import LinearRegression
        
        tracker.start_run("model-test")
        
        # Create and train a simple model
        model = LinearRegression()
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        model.fit(X, y)
        
        model_uri = tracker.log_model(model, "test_model", "sklearn")
        assert model_uri is not None
        
        tracker.end_run()
    
    def test_log_feature_importance(self, tracker):
        """Test feature importance logging."""
        tracker.start_run("importance-test")
        
        feature_names = ['feature_1', 'feature_2', 'feature_3']
        importance_values = [0.5, 0.3, 0.2]
        
        tracker.log_feature_importance(feature_names, importance_values)
        
        # Check that metrics were logged for top features
        run = mlflow.active_run()
        assert 'feature_importance_feature_1' in run.data.metrics
        
        tracker.end_run()
    
    def test_log_predictions(self, tracker):
        """Test predictions logging."""
        tracker.start_run("predictions-test")
        
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        
        tracker.log_predictions(y_true, y_pred, "test")
        
        # Check that residual statistics were logged
        run = mlflow.active_run()
        assert 'test_mean_residual' in run.data.metrics
        assert 'test_std_residual' in run.data.metrics
        
        tracker.end_run()
    
    def test_get_experiment_runs(self, tracker):
        """Test retrieving experiment runs."""
        # Create a few test runs
        for i in range(3):
            tracker.start_run(f"test-run-{i}")
            tracker.log_metrics({'test_metric': i * 10})
            tracker.end_run()
        
        runs_df = tracker.get_experiment_runs()
        assert len(runs_df) >= 3
        assert 'run_id' in runs_df.columns
        assert 'metrics.test_metric' in runs_df.columns


class TestModelVersionManager:
    """Test cases for ModelVersionManager class."""
    
    @pytest.fixture
    def temp_mlruns_dir(self):
        """Create temporary MLflow tracking directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def version_manager(self, temp_mlruns_dir):
        """Create ModelVersionManager instance."""
        tracking_uri = f"file://{temp_mlruns_dir}"
        mlflow.set_tracking_uri(tracking_uri)
        return ModelVersionManager("test-model")
    
    def test_version_manager_initialization(self, version_manager):
        """Test version manager initialization."""
        assert version_manager.model_name == "test-model"
        assert version_manager.client is not None


class TestMLflowIntegration:
    """Test cases for MLflow integration utilities."""
    
    def test_load_experiment_config(self):
        """Test loading experiment configuration."""
        # Create temporary config file
        config_content = """
        experiment:
          name: "test-experiment"
          primary_metrics:
            - "wmape"
            - "mae"
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            config_path = f.name
        
        try:
            config = load_experiment_config(config_path)
            assert config['experiment']['name'] == "test-experiment"
            assert 'wmape' in config['experiment']['primary_metrics']
        finally:
            os.unlink(config_path)
    
    @patch('mlflow.sklearn.autolog')
    @patch('mlflow.xgboost.autolog')
    def test_setup_mlflow_autolog(self, mock_xgb_autolog, mock_sklearn_autolog):
        """Test MLflow autolog setup."""
        from src.utils.mlflow_integration import setup_mlflow_autolog
        
        config = {
            'experiment': {
                'auto_log': {
                    'sklearn': True,
                    'xgboost': True,
                    'lightgbm': False
                }
            }
        }
        
        setup_mlflow_autolog(config)
        
        mock_sklearn_autolog.assert_called_once()
        mock_xgb_autolog.assert_called_once()


class TestMLflowModelTracker:
    """Test cases for MLflowModelTracker class."""
    
    @pytest.fixture
    def temp_config_file(self):
        """Create temporary config file."""
        config_content = """
        experiment:
          name: "test-experiment"
          primary_metrics:
            - "wmape"
            - "mae"
          model_registry:
            base_model_name: "test-model"
        run_naming:
          prefix: "test"
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            yield f.name
        
        os.unlink(f.name)
    
    @pytest.fixture
    def temp_mlruns_dir(self):
        """Create temporary MLflow tracking directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_model_tracker_initialization(self, temp_config_file, temp_mlruns_dir):
        """Test MLflowModelTracker initialization."""
        # Set tracking URI
        tracking_uri = f"file://{temp_mlruns_dir}"
        mlflow.set_tracking_uri(tracking_uri)
        
        tracker = MLflowModelTracker(temp_config_file)
        assert tracker.config is not None
        assert tracker.tracker is not None
    
    def test_calculate_wmape(self):
        """Test WMAPE calculation."""
        from examples.test_experiment_tracking import calculate_wmape
        
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 190, 310])
        
        wmape = calculate_wmape(y_true, y_pred)
        expected_wmape = (10 + 10 + 10) / (100 + 200 + 300) * 100
        
        assert abs(wmape - expected_wmape) < 1e-6


class TestExperimentComparator:
    """Test cases for ExperimentComparator class."""
    
    @pytest.fixture
    def temp_mlruns_dir(self):
        """Create temporary MLflow tracking directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def tracker_with_runs(self, temp_mlruns_dir):
        """Create tracker with some test runs."""
        tracking_uri = f"file://{temp_mlruns_dir}"
        tracker = ExperimentTracker("test-experiment", tracking_uri)
        
        # Create test runs
        for i in range(3):
            tracker.start_run(f"test-run-{i}")
            tracker.log_metrics({
                'wmape': 10 + i,
                'mae': 5 + i * 0.5
            })
            tracker.end_run()
        
        return tracker
    
    def test_experiment_comparator_initialization(self, tracker_with_runs):
        """Test ExperimentComparator initialization."""
        comparator = ExperimentComparator(tracker_with_runs)
        assert comparator.tracker == tracker_with_runs
    
    def test_compare_all_runs(self, tracker_with_runs):
        """Test comparing all runs."""
        comparator = ExperimentComparator(tracker_with_runs)
        comparison = comparator.compare_all_runs(['wmape', 'mae'])
        
        assert len(comparison) >= 3
        assert 'metrics.wmape' in comparison.columns
        assert 'metrics.mae' in comparison.columns
    
    def test_get_performance_summary(self, tracker_with_runs):
        """Test performance summary generation."""
        comparator = ExperimentComparator(tracker_with_runs)
        summary = comparator.get_performance_summary('wmape')
        
        assert 'count' in summary
        assert 'mean' in summary
        assert 'std' in summary
        assert summary['count'] >= 3


def test_integration_with_real_models():
    """Integration test with real ML models."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error
    
    # Create sample data
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randn(100) * 10 + 50
    
    # Create temporary tracking directory
    with tempfile.TemporaryDirectory() as temp_dir:
        tracking_uri = f"file://{temp_dir}"
        tracker = ExperimentTracker("integration-test", tracking_uri)
        
        # Train model and track
        run_id = tracker.start_run("integration-test-run")
        
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        y_pred = model.predict(X)
        
        # Log everything
        tracker.log_params({'n_estimators': 10, 'random_state': 42})
        tracker.log_metrics({'mae': mean_absolute_error(y, y_pred)})
        tracker.log_model(model, 'rf_model', 'sklearn')
        
        tracker.end_run()
        
        # Verify run was created
        runs_df = tracker.get_experiment_runs()
        assert len(runs_df) >= 1
        assert runs_df.iloc[0]['run_id'] == run_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])