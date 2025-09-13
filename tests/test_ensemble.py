"""
Testes para o módulo de ensemble.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import tempfile
import os

from src.models.ensemble import (
    WeightedEnsemble, StackingEnsemble, EnsembleManager, EnsembleValidator
)
from src.models.training import BaseModel


class MockModel(BaseModel):
    """Modelo mock para testes."""
    
    def __init__(self, config, predictions=None):
        super().__init__(config)
        self.predictions = predictions if predictions is not None else np.random.rand(100)
        self.is_fitted = True
    
    def fit(self, X, y, **kwargs):
        self.is_fitted = True
        return self
    
    def predict(self, X):
        return self.predictions[:len(X)]


@pytest.fixture
def sample_data():
    """Dados de exemplo para testes."""
    np.random.seed(42)
    n_samples = 100
    
    X = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples)
    })
    
    y = pd.Series(np.random.rand(n_samples) * 100)
    
    return X, y


@pytest.fixture
def mock_models():
    """Modelos mock para testes."""
    config = {'general': {'n_jobs': 1}}
    
    models = {
        'model1': MockModel(config, np.random.rand(100) * 80 + 10),
        'model2': MockModel(config, np.random.rand(100) * 90 + 5),
        'model3': MockModel(config, np.random.rand(100) * 70 + 15)
    }
    
    return models


@pytest.fixture
def ensemble_config():
    """Configuração para ensemble."""
    return {
        'ensemble': {
            'weighted': {
                'enabled': True,
                'optimize_weights': True,
                'optimization_trials': 10
            },
            'stacking': {
                'enabled': True,
                'meta_learner': 'ridge',
                'ridge_alpha': 1.0
            }
        },
        'validation': {
            'n_splits': 3,
            'test_size': 10
        }
    }


class TestWeightedEnsemble:
    """Testes para WeightedEnsemble."""
    
    def test_init(self, ensemble_config):
        """Testa inicialização do WeightedEnsemble."""
        ensemble = WeightedEnsemble(ensemble_config)
        
        assert ensemble.config == ensemble_config
        assert ensemble.models == {}
        assert not ensemble.is_fitted
        assert ensemble.ensemble_weights is None
    
    def test_fit_with_optimization(self, ensemble_config, mock_models, sample_data):
        """Testa treinamento com otimização de pesos."""
        X, y = sample_data
        ensemble = WeightedEnsemble(ensemble_config)
        
        # Treinar ensemble
        ensemble.fit(mock_models, X, y)
        
        assert ensemble.is_fitted
        assert ensemble.models == mock_models
        assert ensemble.ensemble_weights is not None
        assert len(ensemble.ensemble_weights) == len(mock_models)
        
        # Verificar que pesos somam aproximadamente 1
        total_weight = sum(ensemble.ensemble_weights.values())
        assert abs(total_weight - 1.0) < 0.01
    
    def test_fit_without_optimization(self, ensemble_config, mock_models, sample_data):
        """Testa treinamento sem otimização (pesos uniformes)."""
        X, y = sample_data
        
        # Desabilitar otimização
        ensemble_config['ensemble']['weighted']['optimize_weights'] = False
        ensemble = WeightedEnsemble(ensemble_config)
        
        ensemble.fit(mock_models, X, y)
        
        assert ensemble.is_fitted
        expected_weight = 1.0 / len(mock_models)
        for weight in ensemble.ensemble_weights.values():
            assert abs(weight - expected_weight) < 0.01
    
    def test_predict(self, ensemble_config, mock_models, sample_data):
        """Testa geração de previsões."""
        X, y = sample_data
        ensemble = WeightedEnsemble(ensemble_config)
        ensemble.fit(mock_models, X, y)
        
        # Gerar previsões
        predictions = ensemble.predict(X)
        
        assert len(predictions) == len(X)
        assert isinstance(predictions, np.ndarray)
        assert not np.isnan(predictions).any()
    
    def test_fit_untrained_models(self, ensemble_config, sample_data):
        """Testa erro com modelos não treinados."""
        X, y = sample_data
        
        # Criar modelo não treinado
        untrained_model = MockModel({'general': {'n_jobs': 1}})
        untrained_model.is_fitted = False
        
        models = {'untrained': untrained_model}
        ensemble = WeightedEnsemble(ensemble_config)
        
        with pytest.raises(ValueError, match="deve estar treinado"):
            ensemble.fit(models, X, y)
    
    def test_predict_before_fit(self, ensemble_config, sample_data):
        """Testa erro ao prever antes de treinar."""
        X, y = sample_data
        ensemble = WeightedEnsemble(ensemble_config)
        
        with pytest.raises(ValueError, match="deve ser treinado"):
            ensemble.predict(X)


class TestStackingEnsemble:
    """Testes para StackingEnsemble."""
    
    def test_init(self, ensemble_config):
        """Testa inicialização do StackingEnsemble."""
        ensemble = StackingEnsemble(ensemble_config)
        
        assert ensemble.config == ensemble_config
        assert ensemble.models == {}
        assert not ensemble.is_fitted
        assert ensemble.meta_learner is None
    
    def test_fit_ridge_meta_learner(self, ensemble_config, mock_models, sample_data):
        """Testa treinamento com meta-learner Ridge."""
        X, y = sample_data
        ensemble = StackingEnsemble(ensemble_config)
        
        ensemble.fit(mock_models, X, y)
        
        assert ensemble.is_fitted
        assert ensemble.models == mock_models
        assert ensemble.meta_learner is not None
        assert hasattr(ensemble.meta_learner, 'predict')
    
    def test_fit_linear_meta_learner(self, ensemble_config, mock_models, sample_data):
        """Testa treinamento com meta-learner Linear."""
        X, y = sample_data
        
        # Configurar meta-learner linear
        ensemble_config['ensemble']['stacking']['meta_learner'] = 'linear'
        ensemble = StackingEnsemble(ensemble_config)
        
        ensemble.fit(mock_models, X, y)
        
        assert ensemble.is_fitted
        assert ensemble.meta_learner is not None
    
    def test_fit_random_forest_meta_learner(self, ensemble_config, mock_models, sample_data):
        """Testa treinamento com meta-learner Random Forest."""
        X, y = sample_data
        
        # Configurar meta-learner Random Forest
        ensemble_config['ensemble']['stacking']['meta_learner'] = 'random_forest'
        ensemble_config['ensemble']['stacking']['rf_n_estimators'] = 50
        ensemble_config['ensemble']['stacking']['rf_max_depth'] = 3
        ensemble = StackingEnsemble(ensemble_config)
        
        ensemble.fit(mock_models, X, y)
        
        assert ensemble.is_fitted
        assert ensemble.meta_learner is not None
    
    def test_predict(self, ensemble_config, mock_models, sample_data):
        """Testa geração de previsões."""
        X, y = sample_data
        ensemble = StackingEnsemble(ensemble_config)
        ensemble.fit(mock_models, X, y)
        
        predictions = ensemble.predict(X)
        
        assert len(predictions) == len(X)
        assert isinstance(predictions, np.ndarray)
        assert not np.isnan(predictions).any()
    
    def test_invalid_meta_learner(self, ensemble_config, mock_models, sample_data):
        """Testa erro com meta-learner inválido."""
        X, y = sample_data
        
        ensemble_config['ensemble']['stacking']['meta_learner'] = 'invalid'
        ensemble = StackingEnsemble(ensemble_config)
        
        with pytest.raises(ValueError, match="Meta-learner não suportado"):
            ensemble.fit(mock_models, X, y)


class TestEnsembleManager:
    """Testes para EnsembleManager."""
    
    def test_init(self, ensemble_config):
        """Testa inicialização do EnsembleManager."""
        manager = EnsembleManager(ensemble_config)
        
        assert manager.config == ensemble_config
        assert manager.ensembles == {}
        assert manager.best_ensemble is None
        assert manager.evaluation_results == {}
    
    def test_create_ensembles(self, ensemble_config, mock_models, sample_data):
        """Testa criação de múltiplos ensembles."""
        X, y = sample_data
        manager = EnsembleManager(ensemble_config)
        
        ensembles = manager.create_ensembles(mock_models, X, y)
        
        assert 'weighted' in ensembles
        assert 'stacking' in ensembles
        assert len(manager.ensembles) == 2
        
        for ensemble in ensembles.values():
            assert ensemble.is_fitted
    
    def test_evaluate_ensembles(self, ensemble_config, mock_models, sample_data):
        """Testa avaliação de ensembles."""
        X, y = sample_data
        manager = EnsembleManager(ensemble_config)
        
        # Criar ensembles
        manager.create_ensembles(mock_models, X, y)
        
        # Avaliar ensembles
        results = manager.evaluate_ensembles(X, y)
        
        assert len(results) == len(manager.ensembles)
        for name, metrics in results.items():
            assert 'wmape' in metrics
            assert 'mae' in metrics
            assert 'rmse' in metrics
            assert 'mape' in metrics
    
    def test_select_best_ensemble(self, ensemble_config, mock_models, sample_data):
        """Testa seleção do melhor ensemble."""
        X, y = sample_data
        manager = EnsembleManager(ensemble_config)
        
        # Criar e avaliar ensembles
        manager.create_ensembles(mock_models, X, y)
        manager.evaluate_ensembles(X, y)
        
        # Selecionar melhor ensemble
        best_ensemble = manager.select_best_ensemble('wmape')
        
        assert best_ensemble is not None
        assert manager.best_ensemble == best_ensemble
    
    def test_get_ensemble_summary(self, ensemble_config, mock_models, sample_data):
        """Testa resumo comparativo dos ensembles."""
        X, y = sample_data
        manager = EnsembleManager(ensemble_config)
        
        # Criar e avaliar ensembles
        manager.create_ensembles(mock_models, X, y)
        manager.evaluate_ensembles(X, y)
        
        # Obter resumo
        summary = manager.get_ensemble_summary()
        
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == len(manager.ensembles)
        assert 'wmape' in summary.columns


class TestEnsembleValidator:
    """Testes para EnsembleValidator."""
    
    def test_validate_ensemble_diversity(self, mock_models, sample_data):
        """Testa validação de diversidade do ensemble."""
        X, y = sample_data
        
        diversity_metrics = EnsembleValidator.validate_ensemble_diversity(mock_models, X)
        
        assert 'avg_correlation' in diversity_metrics
        assert 'min_correlation' in diversity_metrics
        assert 'max_correlation' in diversity_metrics
        assert 'diversity_score' in diversity_metrics
        
        # Verificar que métricas estão em ranges válidos
        assert -1 <= diversity_metrics['avg_correlation'] <= 1
        assert 0 <= diversity_metrics['diversity_score'] <= 2
    
    def test_validate_ensemble_stability(self, ensemble_config, mock_models, sample_data):
        """Testa validação de estabilidade do ensemble."""
        X, y = sample_data
        
        # Criar ensemble
        ensemble = WeightedEnsemble(ensemble_config)
        ensemble.fit(mock_models, X, y)
        
        # Validar estabilidade
        stability_metrics = EnsembleValidator.validate_ensemble_stability(
            ensemble, X, n_bootstrap=10
        )
        
        assert 'prediction_std' in stability_metrics
        assert 'prediction_cv' in stability_metrics
        assert 'stability_score' in stability_metrics
        
        # Verificar que métricas são positivas
        assert stability_metrics['prediction_std'] >= 0
        assert stability_metrics['stability_score'] > 0


class TestEnsembleSaveLoad:
    """Testes para salvar e carregar ensembles."""
    
    def test_save_load_weighted_ensemble(self, ensemble_config, mock_models, sample_data):
        """Testa salvar e carregar WeightedEnsemble."""
        X, y = sample_data
        
        # Treinar ensemble
        ensemble = WeightedEnsemble(ensemble_config)
        ensemble.fit(mock_models, X, y)
        
        # Salvar ensemble
        with tempfile.NamedTemporaryFile(suffix='.pkl') as f:
            ensemble.save_ensemble(f.name)
            
            # Carregar ensemble
            new_ensemble = WeightedEnsemble(ensemble_config)
            new_ensemble.load_ensemble(f.name)
            
            # Verificar que ensemble foi carregado corretamente
            assert new_ensemble.is_fitted
            assert new_ensemble.ensemble_weights == ensemble.ensemble_weights
            
            # Verificar que previsões são iguais
            pred1 = ensemble.predict(X)
            pred2 = new_ensemble.predict(X)
            np.testing.assert_array_almost_equal(pred1, pred2)
    
    def test_save_before_fit_error(self, ensemble_config):
        """Testa erro ao salvar ensemble não treinado."""
        ensemble = WeightedEnsemble(ensemble_config)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl') as f:
            with pytest.raises(ValueError, match="deve ser treinado"):
                ensemble.save_ensemble(f.name)


if __name__ == "__main__":
    pytest.main([__file__])