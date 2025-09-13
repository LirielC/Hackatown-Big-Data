"""
Exemplo de uso do módulo de ensemble para combinação de modelos.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import yaml
import logging
from datetime import datetime

from src.models.training import XGBoostModel, LightGBMModel
from src.models.ensemble import (
    WeightedEnsemble, StackingEnsemble, EnsembleManager, EnsembleValidator
)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_sample_data(n_samples=1000, n_features=10, noise_level=0.1):
    """Gera dados sintéticos para demonstração."""
    np.random.seed(42)
    
    # Gerar features
    X = pd.DataFrame({
        f'feature_{i}': np.random.randn(n_samples) for i in range(n_features)
    })
    
    # Adicionar features temporais
    dates = pd.date_range('2022-01-01', periods=n_samples, freq='D')
    X['semana'] = dates.isocalendar().week
    X['mes'] = dates.month
    X['trimestre'] = dates.quarter
    
    # Gerar target com padrão não-linear
    y_values = (
        2 * X['feature_0'].values + 
        1.5 * X['feature_1'].values * X['feature_2'].values +
        0.5 * np.sin(X['semana'].values * 2 * np.pi / 52) +  # Sazonalidade anual
        noise_level * np.random.randn(n_samples)
    )
    
    # Garantir valores positivos e válidos
    y_values = np.maximum(y_values, 0.1)
    y_values = np.clip(y_values, 0.1, 1000.0)
    y = pd.Series(y_values)
    
    return X, pd.Series(y)


def load_config():
    """Carrega configuração do modelo."""
    config_path = 'configs/model_config.yaml'
    
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        # Configuração padrão para demonstração
        config = {
            'general': {
                'random_seed': 42,
                'n_jobs': -1
            },
            'models': {
                'xgboost': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42
                },
                'lightgbm': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42
                }
            },
            'ensemble': {
                'weighted': {
                    'enabled': True,
                    'optimize_weights': True,
                    'optimization_trials': 50
                },
                'stacking': {
                    'enabled': True,
                    'meta_learner': 'ridge',
                    'ridge_alpha': 1.0
                }
            },
            'validation': {
                'n_splits': 5,
                'test_size': 50
            },
            'hyperparameter_tuning': {
                'enabled': False
            }
        }
    
    return config


def train_base_models(X_train, y_train, X_val, y_val, config):
    """Treina modelos base para o ensemble."""
    logger.info("Treinando modelos base...")
    
    models = {}
    
    # Treinar XGBoost
    logger.info("Treinando XGBoost...")
    xgb_model = XGBoostModel(config)
    xgb_model.fit(X_train, y_train, X_val, y_val, optimize_hyperparams=False)
    models['xgboost'] = xgb_model
    
    # Treinar LightGBM
    logger.info("Treinando LightGBM...")
    lgb_model = LightGBMModel(config)
    lgb_model.fit(X_train, y_train, X_val, y_val, optimize_hyperparams=False)
    models['lightgbm'] = lgb_model
    
    logger.info(f"Modelos base treinados: {list(models.keys())}")
    return models


def demonstrate_weighted_ensemble(models, X_train, y_train, X_test, y_test, config):
    """Demonstra uso do WeightedEnsemble."""
    logger.info("\n=== Demonstração WeightedEnsemble ===")
    
    # Criar e treinar ensemble ponderado
    weighted_ensemble = WeightedEnsemble(config)
    weighted_ensemble.fit(models, X_train, y_train)
    
    # Mostrar pesos otimizados
    logger.info("Pesos otimizados:")
    for name, weight in weighted_ensemble.ensemble_weights.items():
        logger.info(f"  {name}: {weight:.4f}")
    
    # Gerar previsões
    predictions = weighted_ensemble.predict(X_test)
    
    # Calcular métricas
    from src.models.training import ModelEvaluator
    metrics = ModelEvaluator.evaluate_model(y_test.values, predictions)
    
    logger.info("Performance do WeightedEnsemble:")
    for metric, value in metrics.items():
        logger.info(f"  {metric.upper()}: {value:.4f}")
    
    return weighted_ensemble, metrics


def demonstrate_stacking_ensemble(models, X_train, y_train, X_test, y_test, config):
    """Demonstra uso do StackingEnsemble."""
    logger.info("\n=== Demonstração StackingEnsemble ===")
    
    # Criar e treinar ensemble de stacking
    stacking_ensemble = StackingEnsemble(config)
    stacking_ensemble.fit(models, X_train, y_train)
    
    # Gerar previsões
    predictions = stacking_ensemble.predict(X_test)
    
    # Calcular métricas
    from src.models.training import ModelEvaluator
    metrics = ModelEvaluator.evaluate_model(y_test.values, predictions)
    
    logger.info("Performance do StackingEnsemble:")
    for metric, value in metrics.items():
        logger.info(f"  {metric.upper()}: {value:.4f}")
    
    return stacking_ensemble, metrics


def demonstrate_ensemble_manager(models, X_train, y_train, X_test, y_test, config):
    """Demonstra uso do EnsembleManager."""
    logger.info("\n=== Demonstração EnsembleManager ===")
    
    # Criar manager
    manager = EnsembleManager(config)
    
    # Criar múltiplos ensembles
    ensembles = manager.create_ensembles(models, X_train, y_train)
    logger.info(f"Ensembles criados: {list(ensembles.keys())}")
    
    # Avaliar ensembles
    evaluation_results = manager.evaluate_ensembles(X_test, y_test)
    
    # Mostrar resultados
    logger.info("Resultados da avaliação:")
    for name, metrics in evaluation_results.items():
        logger.info(f"  {name}:")
        for metric, value in metrics.items():
            logger.info(f"    {metric.upper()}: {value:.4f}")
    
    # Selecionar melhor ensemble
    best_ensemble = manager.select_best_ensemble('wmape')
    
    # Mostrar resumo comparativo
    summary = manager.get_ensemble_summary()
    logger.info("\nResumo comparativo:")
    logger.info(summary.round(4))
    
    return manager, best_ensemble


def demonstrate_ensemble_validation(models, ensembles, X_test):
    """Demonstra validação de ensembles."""
    logger.info("\n=== Validação de Ensembles ===")
    
    # Validar diversidade dos modelos base
    diversity_metrics = EnsembleValidator.validate_ensemble_diversity(models, X_test)
    
    logger.info("Métricas de diversidade dos modelos base:")
    for metric, value in diversity_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Validar estabilidade dos ensembles
    for name, ensemble in ensembles.items():
        stability_metrics = EnsembleValidator.validate_ensemble_stability(
            ensemble, X_test, n_bootstrap=20
        )
        
        logger.info(f"\nMétricas de estabilidade - {name}:")
        for metric, value in stability_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")


def compare_individual_vs_ensemble_performance(models, ensembles, X_test, y_test):
    """Compara performance individual vs ensemble."""
    logger.info("\n=== Comparação Individual vs Ensemble ===")
    
    from src.models.training import ModelEvaluator
    
    results = {}
    
    # Avaliar modelos individuais
    logger.info("Performance dos modelos individuais:")
    for name, model in models.items():
        predictions = model.predict(X_test)
        metrics = ModelEvaluator.evaluate_model(y_test.values, predictions)
        results[name] = metrics
        logger.info(f"  {name} - WMAPE: {metrics['wmape']:.4f}")
    
    # Avaliar ensembles
    logger.info("\nPerformance dos ensembles:")
    for name, ensemble in ensembles.items():
        predictions = ensemble.predict(X_test)
        metrics = ModelEvaluator.evaluate_model(y_test.values, predictions)
        results[f"ensemble_{name}"] = metrics
        logger.info(f"  {name} - WMAPE: {metrics['wmape']:.4f}")
    
    # Criar DataFrame comparativo
    comparison_df = pd.DataFrame(results).T
    logger.info("\nComparação completa:")
    logger.info(comparison_df.round(4))
    
    return comparison_df


def main():
    """Função principal de demonstração."""
    logger.info("Iniciando demonstração do módulo de ensemble")
    
    # Carregar configuração
    config = load_config()
    
    # Gerar dados sintéticos
    logger.info("Gerando dados sintéticos...")
    X, y = generate_sample_data(n_samples=1000, n_features=8)
    
    # Dividir dados
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    logger.info(f"Dados divididos - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    try:
        # Treinar modelos base
        models = train_base_models(X_train, y_train, X_val, y_val, config)
        
        # Demonstrar WeightedEnsemble
        weighted_ensemble, weighted_metrics = demonstrate_weighted_ensemble(
            models, X_train, y_train, X_test, y_test, config
        )
        
        # Demonstrar StackingEnsemble
        stacking_ensemble, stacking_metrics = demonstrate_stacking_ensemble(
            models, X_train, y_train, X_test, y_test, config
        )
        
        # Demonstrar EnsembleManager
        ensembles = {'weighted': weighted_ensemble, 'stacking': stacking_ensemble}
        manager, best_ensemble = demonstrate_ensemble_manager(
            models, X_train, y_train, X_test, y_test, config
        )
        
        # Validação de ensembles
        demonstrate_ensemble_validation(models, ensembles, X_test)
        
        # Comparação final
        comparison_results = compare_individual_vs_ensemble_performance(
            models, ensembles, X_test, y_test
        )
        
        logger.info("\nDemonstração concluída com sucesso!")
        
    except Exception as e:
        logger.error(f"Erro durante a demonstração: {str(e)}")
        raise


if __name__ == "__main__":
    main()