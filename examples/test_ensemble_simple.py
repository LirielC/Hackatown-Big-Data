"""
Exemplo simples de uso do módulo de ensemble.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import logging
from sklearn.datasets import make_regression

from src.models.ensemble import WeightedEnsemble, StackingEnsemble, EnsembleManager
from src.models.training import BaseModel

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleModel(BaseModel):
    """Modelo simples para demonstração."""
    
    def __init__(self, config, model_type='linear'):
        super().__init__(config)
        self.model_type = model_type
        self.coefficients = None
        
    def fit(self, X, y, **kwargs):
        """Treina modelo simples."""
        if self.model_type == 'linear':
            # Regressão linear simples
            X_with_bias = np.column_stack([np.ones(len(X)), X])
            self.coefficients = np.linalg.lstsq(X_with_bias, y, rcond=None)[0]
        elif self.model_type == 'mean':
            # Modelo que sempre prediz a média
            self.coefficients = np.mean(y)
        elif self.model_type == 'random':
            # Modelo com coeficientes aleatórios
            np.random.seed(42)
            self.coefficients = np.random.randn(X.shape[1] + 1) * 0.1
            
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Gera previsões."""
        if not self.is_fitted:
            raise ValueError("Modelo deve ser treinado")
            
        if self.model_type == 'linear' or self.model_type == 'random':
            X_with_bias = np.column_stack([np.ones(len(X)), X])
            return X_with_bias @ self.coefficients
        elif self.model_type == 'mean':
            return np.full(len(X), self.coefficients)


def create_sample_data():
    """Cria dados sintéticos simples."""
    X, y = make_regression(
        n_samples=1000,
        n_features=5,
        noise=0.1,
        random_state=42
    )
    
    # Converter para DataFrame/Series
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y = pd.Series(y)
    
    return X, y


def demonstrate_weighted_ensemble():
    """Demonstra WeightedEnsemble."""
    logger.info("=== Demonstração WeightedEnsemble ===")
    
    # Criar dados
    X, y = create_sample_data()
    
    # Dividir dados
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Criar modelos simples
    config = {'general': {'n_jobs': 1}}
    models = {
        'linear': SimpleModel(config, 'linear'),
        'mean': SimpleModel(config, 'mean'),
        'random': SimpleModel(config, 'random')
    }
    
    # Treinar modelos
    for name, model in models.items():
        model.fit(X_train, y_train)
        logger.info(f"Modelo {name} treinado")
    
    # Configuração do ensemble
    ensemble_config = {
        'ensemble': {
            'weighted': {
                'enabled': True,
                'optimize_weights': True,
                'optimization_trials': 20
            }
        },
        'validation': {
            'n_splits': 3,
            'test_size': 50
        }
    }
    
    # Criar e treinar ensemble
    ensemble = WeightedEnsemble(ensemble_config)
    ensemble.fit(models, X_train, y_train)
    
    # Mostrar pesos
    logger.info("Pesos otimizados:")
    for name, weight in ensemble.ensemble_weights.items():
        logger.info(f"  {name}: {weight:.4f}")
    
    # Gerar previsões
    predictions = ensemble.predict(X_test)
    
    # Calcular erro simples
    mae = np.mean(np.abs(y_test - predictions))
    logger.info(f"MAE do ensemble: {mae:.4f}")
    
    return ensemble


def demonstrate_stacking_ensemble():
    """Demonstra StackingEnsemble."""
    logger.info("\n=== Demonstração StackingEnsemble ===")
    
    # Criar dados
    X, y = create_sample_data()
    
    # Dividir dados
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Criar modelos simples
    config = {'general': {'n_jobs': 1}}
    models = {
        'linear': SimpleModel(config, 'linear'),
        'mean': SimpleModel(config, 'mean'),
        'random': SimpleModel(config, 'random')
    }
    
    # Treinar modelos
    for name, model in models.items():
        model.fit(X_train, y_train)
    
    # Configuração do ensemble
    ensemble_config = {
        'ensemble': {
            'stacking': {
                'enabled': True,
                'meta_learner': 'ridge',
                'ridge_alpha': 1.0
            }
        },
        'validation': {
            'n_splits': 3,
            'test_size': 50
        }
    }
    
    # Criar e treinar ensemble
    ensemble = StackingEnsemble(ensemble_config)
    ensemble.fit(models, X_train, y_train)
    
    # Gerar previsões
    predictions = ensemble.predict(X_test)
    
    # Calcular erro simples
    mae = np.mean(np.abs(y_test - predictions))
    logger.info(f"MAE do ensemble: {mae:.4f}")
    
    return ensemble


def demonstrate_ensemble_manager():
    """Demonstra EnsembleManager."""
    logger.info("\n=== Demonstração EnsembleManager ===")
    
    # Criar dados
    X, y = create_sample_data()
    
    # Dividir dados
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Criar modelos simples
    config = {'general': {'n_jobs': 1}}
    models = {
        'linear': SimpleModel(config, 'linear'),
        'mean': SimpleModel(config, 'mean'),
        'random': SimpleModel(config, 'random')
    }
    
    # Treinar modelos
    for name, model in models.items():
        model.fit(X_train, y_train)
    
    # Configuração completa
    ensemble_config = {
        'ensemble': {
            'weighted': {
                'enabled': True,
                'optimize_weights': True,
                'optimization_trials': 20
            },
            'stacking': {
                'enabled': True,
                'meta_learner': 'ridge',
                'ridge_alpha': 1.0
            }
        },
        'validation': {
            'n_splits': 3,
            'test_size': 50
        }
    }
    
    # Criar manager
    manager = EnsembleManager(ensemble_config)
    
    # Criar ensembles
    ensembles = manager.create_ensembles(models, X_train, y_train)
    logger.info(f"Ensembles criados: {list(ensembles.keys())}")
    
    # Avaliar ensembles
    evaluation_results = manager.evaluate_ensembles(X_test, y_test)
    
    # Mostrar resultados
    logger.info("Resultados da avaliação:")
    for name, metrics in evaluation_results.items():
        logger.info(f"  {name}: MAE = {metrics['mae']:.4f}")
    
    # Selecionar melhor ensemble
    best_ensemble = manager.select_best_ensemble('mae')
    
    return manager, best_ensemble


def main():
    """Função principal."""
    logger.info("Iniciando demonstração simples do módulo de ensemble")
    
    try:
        # Demonstrar WeightedEnsemble
        weighted_ensemble = demonstrate_weighted_ensemble()
        
        # Demonstrar StackingEnsemble
        stacking_ensemble = demonstrate_stacking_ensemble()
        
        # Demonstrar EnsembleManager
        manager, best_ensemble = demonstrate_ensemble_manager()
        
        logger.info("\nDemonstração concluída com sucesso!")
        
    except Exception as e:
        logger.error(f"Erro durante a demonstração: {str(e)}")
        raise


if __name__ == "__main__":
    main()