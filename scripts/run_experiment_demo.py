#!/usr/bin/env python3
"""
Demo script for experiment tracking system.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

from src.utils.mlflow_integration import MLflowModelTracker, setup_mlflow_autolog


def calculate_wmape(y_true, y_pred):
    """Calculate Weighted Mean Absolute Percentage Error."""
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100


def generate_sample_data(n_samples=1000, n_features=10, noise=0.1):
    """Generate sample sales-like data."""
    np.random.seed(42)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Create realistic sales pattern
    # Base sales influenced by seasonality and trends
    seasonal_pattern = np.sin(np.arange(n_samples) * 2 * np.pi / 52)  # Weekly seasonality
    trend = np.arange(n_samples) * 0.01  # Small upward trend
    
    # Feature influence
    feature_effects = X @ np.random.randn(n_features) * 10
    
    # Combine effects
    y = 1000 + seasonal_pattern * 100 + trend * 50 + feature_effects + np.random.randn(n_samples) * noise * 100
    
    # Ensure positive values (sales can't be negative)
    y = np.maximum(y, 0)
    
    return X, y


def demo_experiment_tracking():
    """Demonstrate experiment tracking with multiple models."""
    print("=== DEMO: Sistema de Experiment Tracking ===\n")
    
    # Setup MLflow autologging
    setup_mlflow_autolog()
    
    # Initialize tracker
    tracker = MLflowModelTracker()
    
    # Generate sample data
    print("Gerando dados de exemplo...")
    X, y = generate_sample_data(n_samples=2000, n_features=15)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Dados de treino: {X_train.shape}")
    print(f"Dados de teste: {X_test.shape}")
    print(f"Média das vendas: {y.mean():.2f}")
    print()
    
    # Model 1: Random Forest
    print("1. Treinando Random Forest...")
    rf_params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'random_state': 42
    }
    
    rf_model = RandomForestRegressor(**rf_params)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    
    rf_metrics = {
        'mae': mean_absolute_error(y_test, rf_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, rf_pred)),
        'wmape': calculate_wmape(y_test, rf_pred)
    }
    
    rf_feature_importance = dict(zip(
        [f'feature_{i}' for i in range(X.shape[1])],
        rf_model.feature_importances_
    ))
    
    rf_run_id = tracker.track_model_training(
        model_name='random_forest_demo',
        model_type='sklearn',
        model=rf_model,
        params=rf_params,
        metrics=rf_metrics,
        feature_importance=rf_feature_importance,
        predictions={'y_true': y_test, 'y_pred': rf_pred}
    )
    
    print(f"   WMAPE: {rf_metrics['wmape']:.4f}")
    print(f"   Run ID: {rf_run_id[:8]}...")
    print()
    
    # Model 2: XGBoost
    print("2. Treinando XGBoost...")
    xgb_params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }
    
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    
    xgb_metrics = {
        'mae': mean_absolute_error(y_test, xgb_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, xgb_pred)),
        'wmape': calculate_wmape(y_test, xgb_pred)
    }
    
    xgb_feature_importance = dict(zip(
        [f'feature_{i}' for i in range(X.shape[1])],
        xgb_model.feature_importances_
    ))
    
    xgb_run_id = tracker.track_model_training(
        model_name='xgboost_demo',
        model_type='xgboost',
        model=xgb_model,
        params=xgb_params,
        metrics=xgb_metrics,
        feature_importance=xgb_feature_importance,
        predictions={'y_true': y_test, 'y_pred': xgb_pred}
    )
    
    print(f"   WMAPE: {xgb_metrics['wmape']:.4f}")
    print(f"   Run ID: {xgb_run_id[:8]}...")
    print()
    
    # Model 3: XGBoost with different parameters
    print("3. Treinando XGBoost (configuração 2)...")
    xgb_params_2 = {
        'n_estimators': 200,
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'random_state': 42
    }
    
    xgb_model_2 = xgb.XGBRegressor(**xgb_params_2)
    xgb_model_2.fit(X_train, y_train)
    xgb_pred_2 = xgb_model_2.predict(X_test)
    
    xgb_metrics_2 = {
        'mae': mean_absolute_error(y_test, xgb_pred_2),
        'rmse': np.sqrt(mean_squared_error(y_test, xgb_pred_2)),
        'wmape': calculate_wmape(y_test, xgb_pred_2)
    }
    
    xgb_run_id_2 = tracker.track_model_training(
        model_name='xgboost_demo_v2',
        model_type='xgboost',
        model=xgb_model_2,
        params=xgb_params_2,
        metrics=xgb_metrics_2,
        feature_importance=dict(zip(
            [f'feature_{i}' for i in range(X.shape[1])],
            xgb_model_2.feature_importances_
        )),
        predictions={'y_true': y_test, 'y_pred': xgb_pred_2}
    )
    
    print(f"   WMAPE: {xgb_metrics_2['wmape']:.4f}")
    print(f"   Run ID: {xgb_run_id_2[:8]}...")
    print()
    
    # Compare models
    print("=== COMPARAÇÃO DE MODELOS ===")
    comparison = tracker.compare_models(
        [rf_run_id, xgb_run_id, xgb_run_id_2],
        ['wmape', 'mae', 'rmse']
    )
    
    print("Resultados:")
    for _, row in comparison.iterrows():
        run_id = row['run_id'][:8]
        wmape = row.get('wmape', 'N/A')
        mae = row.get('mae', 'N/A')
        print(f"  {run_id}: WMAPE={wmape:.4f}, MAE={mae:.2f}")
    
    print()
    
    # Show leaderboard
    print("=== LEADERBOARD ===")
    leaderboard = tracker.get_model_leaderboard(metric='wmape', top_k=5)
    
    if not leaderboard.empty:
        for i, (_, row) in enumerate(leaderboard.iterrows(), 1):
            run_id = row['run_id'][:8]
            wmape = row.get('metrics.wmape', 'N/A')
            model_type = row.get('params.model_type', 'Unknown')
            print(f"  {i}. {run_id} - WMAPE: {wmape:.4f} - Tipo: {model_type}")
    
    print()
    
    # Register best model
    print("=== REGISTRO DO MELHOR MODELO ===")
    best_run_id = tracker.register_best_model(metric='wmape', minimize=True)
    if best_run_id:
        print(f"Melhor modelo registrado: {best_run_id[:8]}...")
    
    # Export results
    tracker.export_experiment_results("demo_experiment_results.csv")
    print("Resultados exportados para: demo_experiment_results.csv")
    
    # Create report
    from src.utils.mlflow_integration import create_experiment_report
    create_experiment_report(tracker.tracker, "demo_experiment_report.html")
    print("Relatório HTML criado: demo_experiment_report.html")
    
    print("\n=== COMO VISUALIZAR OS RESULTADOS ===")
    print("1. Execute no terminal: mlflow ui")
    print("2. Abra no navegador: http://localhost:5000")
    print("3. Selecione o experimento: hackathon-forecast-2025")
    print("\nDemo concluída com sucesso!")


if __name__ == "__main__":
    demo_experiment_tracking()