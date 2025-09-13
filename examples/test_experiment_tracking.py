"""
Example usage of the experiment tracking system.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb

from src.utils.experiment_tracker import ExperimentTracker, ModelVersionManager
from src.utils.mlflow_integration import (
    MLflowModelTracker, 
    setup_mlflow_autolog,
    mlflow_experiment,
    create_experiment_report
)


def calculate_wmape(y_true, y_pred):
    """Calculate Weighted Mean Absolute Percentage Error."""
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100


@mlflow_experiment(run_name="test_sklearn_model")
def train_sklearn_model():
    """Example of training a sklearn model with automatic tracking."""
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(1000, 10)
    y = np.random.randn(1000) * 100 + 1000
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    metrics = {
        'mae': mean_absolute_error(y, y_pred),
        'rmse': np.sqrt(mean_squared_error(y, y_pred)),
        'wmape': calculate_wmape(y, y_pred)
    }
    
    return {
        'model': model,
        'model_type': 'sklearn',
        'metrics': metrics
    }


def test_comprehensive_tracking():
    """Test comprehensive model tracking with MLflowModelTracker."""
    print("Testing comprehensive model tracking...")
    
    # Initialize tracker
    tracker = MLflowModelTracker()
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(1000, 10)
    y = np.random.randn(1000) * 100 + 1000
    
    # Test XGBoost tracking
    print("Training XGBoost model...")
    xgb_params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42
    }
    
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X, y)
    y_pred_xgb = xgb_model.predict(X)
    
    xgb_metrics = {
        'mae': mean_absolute_error(y, y_pred_xgb),
        'rmse': np.sqrt(mean_squared_error(y, y_pred_xgb)),
        'wmape': calculate_wmape(y, y_pred_xgb)
    }
    
    # Get feature importance
    feature_importance = dict(zip(
        [f'feature_{i}' for i in range(X.shape[1])],
        xgb_model.feature_importances_
    ))
    
    # Cross-validation
    cv_scores = cross_val_score(xgb_model, X, y, cv=5, scoring='neg_mean_absolute_error')
    cv_results = {
        'cv_mae_scores': -cv_scores,
        'cv_mae_mean': -cv_scores.mean(),
        'cv_mae_std': cv_scores.std()
    }
    
    # Track XGBoost model
    xgb_run_id = tracker.track_model_training(
        model_name='xgboost_forecast',
        model_type='xgboost',
        model=xgb_model,
        params=xgb_params,
        metrics=xgb_metrics,
        feature_importance=feature_importance,
        cv_results=cv_results,
        predictions={'y_true': y, 'y_pred': y_pred_xgb}
    )
    
    print(f"XGBoost run ID: {xgb_run_id}")
    
    # Test LightGBM tracking
    print("Training LightGBM model...")
    lgb_params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42
    }
    
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(X, y)
    y_pred_lgb = lgb_model.predict(X)
    
    lgb_metrics = {
        'mae': mean_absolute_error(y, y_pred_lgb),
        'rmse': np.sqrt(mean_squared_error(y, y_pred_lgb)),
        'wmape': calculate_wmape(y, y_pred_lgb)
    }
    
    # Track LightGBM model
    lgb_run_id = tracker.track_model_training(
        model_name='lightgbm_forecast',
        model_type='lightgbm',
        model=lgb_model,
        params=lgb_params,
        metrics=lgb_metrics,
        feature_importance=dict(zip(
            [f'feature_{i}' for i in range(X.shape[1])],
            lgb_model.feature_importances_
        )),
        predictions={'y_true': y, 'y_pred': y_pred_lgb}
    )
    
    print(f"LightGBM run ID: {lgb_run_id}")
    
    # Compare models
    print("\nComparing models...")
    comparison = tracker.compare_models([xgb_run_id, lgb_run_id])
    print(comparison)
    
    # Get leaderboard
    print("\nModel leaderboard:")
    leaderboard = tracker.get_model_leaderboard()
    print(leaderboard)
    
    return tracker


def test_model_versioning():
    """Test model versioning functionality."""
    print("Testing model versioning...")
    
    # Initialize tracker and train a model
    tracker = test_comprehensive_tracking()
    
    # Register best model
    best_run_id = tracker.register_best_model(metric='wmape', minimize=True)
    
    if best_run_id:
        print(f"Registered best model from run: {best_run_id}")
        
        # Test version management
        version_manager = ModelVersionManager("hackathon-forecast-model")
        
        # Get latest version
        latest_version = version_manager.get_latest_version()
        print(f"Latest model version: {latest_version}")
        
        if latest_version:
            # Transition to staging
            version_manager.transition_model_stage(latest_version, "Staging")
            print(f"Transitioned version {latest_version} to Staging")
    
    return tracker


def test_experiment_comparison():
    """Test experiment comparison and analysis."""
    print("Testing experiment comparison...")
    
    # Initialize tracker
    tracker_instance = ExperimentTracker("hackathon-forecast-2025")
    
    # Get all runs
    runs_df = tracker_instance.get_experiment_runs()
    print(f"Total runs in experiment: {len(runs_df)}")
    
    if len(runs_df) > 0:
        # Performance summary
        from src.utils.mlflow_integration import MLflowModelTracker
        tracker = MLflowModelTracker()
        
        performance_summary = tracker.tracker.get_experiment_runs()
        print("\nExperiment summary:")
        print(performance_summary[['run_id', 'start_time', 'metrics.wmape', 'params.model_type']].head())
        
        # Export results
        tracker.export_experiment_results("test_experiment_results.csv")
        print("Exported experiment results to test_experiment_results.csv")
        
        # Create report
        create_experiment_report(tracker.tracker, "test_experiment_report.html")
        print("Created experiment report: test_experiment_report.html")


def main():
    """Run all experiment tracking tests."""
    print("=== Testing Experiment Tracking System ===\n")
    
    # Setup MLflow autologging
    setup_mlflow_autolog()
    
    # Test 1: Basic sklearn model with decorator
    print("1. Testing sklearn model with decorator...")
    train_sklearn_model()
    print("✓ Completed\n")
    
    # Test 2: Comprehensive tracking
    print("2. Testing comprehensive model tracking...")
    tracker = test_comprehensive_tracking()
    print("✓ Completed\n")
    
    # Test 3: Model versioning
    print("3. Testing model versioning...")
    test_model_versioning()
    print("✓ Completed\n")
    
    # Test 4: Experiment comparison
    print("4. Testing experiment comparison...")
    test_experiment_comparison()
    print("✓ Completed\n")
    
    print("=== All tests completed successfully! ===")
    print("\nTo view results:")
    print("1. Run 'mlflow ui' in the terminal")
    print("2. Open http://localhost:5000 in your browser")
    print("3. Check the 'hackathon-forecast-2025' experiment")


if __name__ == "__main__":
    main()