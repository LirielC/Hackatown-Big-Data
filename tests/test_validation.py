"""
Testes para o módulo de validação e avaliação de modelos.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import tempfile
import os

from src.models.validation import (
    WalkForwardValidator,
    ResidualAnalyzer,
    BaselineComparator,
    ValidationManager
)
from src.models.training import BaseModel


class MockModel(BaseModel):
    """Modelo mock para testes."""
    
    def __init__(self, config):
        super().__init__(config)
        self.model = Mock()
        
    def fit(self, X, y, **kwargs):
        self.is_fitted = True
        self.feature_importance_ = pd.DataFrame({
            'feature': X.columns,
            'importance': np.random.random(len(X.columns))
        })
        return self
        
    def predict(self, X):
        # Previsão simples baseada na primeira coluna + ruído
        if len(X.columns) > 0:
            base_pred = X.iloc[:, 0] * 0.8 + np.random.normal(0, 0.1, len(X))
        else:
            base_pred = np.random.normal(100, 10, len(X))
        return base_pred.values


@pytest.fixture
def sample_config():
    """Configuração de exemplo para testes."""
    return {
        'validation': {
            'walk_forward': {
                'initial_train_size': 0.7,
                'step_size': 4,
                'min_train_size': 20
            },
            'n_splits': 3,
            'test_size': 4
        },
        'baseline': {
            'moving_average_window': 4,
            'seasonal_period': 12
        },
        'visualization': {
            'save_plots': True
        }
    }


@pytest.fixture
def sample_data():
    """Dados de exemplo para testes."""
    np.random.seed(42)
    n_samples = 100
    
    # Criar dados temporais
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='W')
    
    # Features
    X = pd.DataFrame({
        'data': dates,
        'feature1': np.random.normal(100, 20, n_samples),
        'feature2': np.random.normal(50, 10, n_samples),
        'feature3': np.random.normal(0, 1, n_samples),
        'segmento': np.random.choice(['A', 'B', 'C'], n_samples)
    })
    
    # Target com tendência e sazonalidade
    trend = np.linspace(100, 150, n_samples)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(n_samples) / 12)
    noise = np.random.normal(0, 5, n_samples)
    y = pd.Series(trend + seasonal + noise + X['feature1'] * 0.3)
    
    return X, y


class TestWalkForwardValidator:
    """Testes para WalkForwardValidator."""
    
    def test_init(self, sample_config):
        """Testa inicialização do validador."""
        validator = WalkForwardValidator(sample_config)
        assert validator.config == sample_config
        assert validator.validation_config == sample_config['validation']
    
    def test_validate_model(self, sample_config, sample_data):
        """Testa validação walk-forward."""
        X, y = sample_data
        model = MockModel(sample_config)
        model.fit(X.drop(['data', 'segmento'], axis=1), y)
        
        validator = WalkForwardValidator(sample_config)
        results = validator.validate_model(model, X, y, 'data')
        
        # Verificar estrutura dos resultados
        assert 'fold_results' in results
        assert 'predictions' in results
        assert 'actuals' in results
        assert 'overall_metrics' in results
        
        # Verificar que temos pelo menos um fold
        assert len(results['fold_results']) > 0
        
        # Verificar métricas
        assert 'wmape' in results['overall_metrics']
        assert 'mae' in results['overall_metrics']
        assert results['overall_metrics']['wmape'] >= 0
    
    def test_validate_model_with_index_date(self, sample_config, sample_data):
        """Testa validação com data no índice."""
        X, y = sample_data
        X_indexed = X.drop('data', axis=1).set_index(X['data'])
        
        model = MockModel(sample_config)
        model.fit(X_indexed.drop('segmento', axis=1), y)
        
        validator = WalkForwardValidator(sample_config)
        results = validator.validate_model(model, X_indexed, y)
        
        assert len(results['fold_results']) > 0
        assert 'overall_metrics' in results


class TestResidualAnalyzer:
    """Testes para ResidualAnalyzer."""
    
    def test_init(self, sample_config):
        """Testa inicialização do analisador."""
        analyzer = ResidualAnalyzer(sample_config)
        assert analyzer.config == sample_config
    
    def test_analyze_residuals_basic(self, sample_config):
        """Testa análise básica de resíduos."""
        np.random.seed(42)
        y_true = np.random.normal(100, 20, 100)
        y_pred = y_true + np.random.normal(0, 5, 100)
        
        analyzer = ResidualAnalyzer(sample_config)
        results = analyzer.analyze_residuals(y_true, y_pred)
        
        # Verificar estrutura dos resultados
        assert 'basic_stats' in results
        assert 'normality_tests' in results
        assert 'autocorrelation' in results
        assert 'outliers' in results
        assert 'bias_analysis' in results
        
        # Verificar estatísticas básicas
        assert 'mean' in results['basic_stats']
        assert 'std' in results['basic_stats']
        assert abs(results['basic_stats']['mean']) < 2  # Deve ser próximo de 0
    
    def test_analyze_residuals_with_segments(self, sample_config, sample_data):
        """Testa análise de resíduos por segmento."""
        X, y = sample_data
        y_pred = y + np.random.normal(0, 5, len(y))
        
        analyzer = ResidualAnalyzer(sample_config)
        results = analyzer.analyze_residuals(y.values, y_pred.values, X, 'segmento')
        
        assert 'segment_analysis' in results
        assert len(results['segment_analysis']) > 0
        
        # Verificar que cada segmento tem métricas
        for segment_name, segment_data in results['segment_analysis'].items():
            assert 'metrics' in segment_data
            assert 'residual_stats' in segment_data
    
    def test_outlier_detection(self, sample_config):
        """Testa detecção de outliers."""
        np.random.seed(42)
        # Dados normais + alguns outliers
        residuals = np.concatenate([
            np.random.normal(0, 1, 95),
            np.array([10, -10, 15, -15, 20])  # Outliers óbvios
        ])
        
        analyzer = ResidualAnalyzer(sample_config)
        results = analyzer.analyze_residuals(residuals, np.zeros_like(residuals))
        
        # Deve detectar outliers
        assert results['outliers']['z_score_outliers']['count'] > 0
        assert results['outliers']['iqr_outliers']['count'] > 0


class TestBaselineComparator:
    """Testes para BaselineComparator."""
    
    def test_init(self, sample_config):
        """Testa inicialização do comparador."""
        comparator = BaselineComparator(sample_config)
        assert comparator.config == sample_config
    
    def test_create_baselines(self, sample_config, sample_data):
        """Testa criação de baselines."""
        X, y = sample_data
        
        comparator = BaselineComparator(sample_config)
        baselines = comparator.create_baselines(X, y, 'segmento')
        
        # Verificar que todos os baselines foram criados
        expected_baselines = [
            'historical_mean', 'last_value', 'moving_average',
            'linear_trend', 'seasonal_naive', 'segment_baselines'
        ]
        
        for baseline in expected_baselines:
            assert baseline in baselines
        
        # Verificar valores dos baselines
        assert baselines['historical_mean']['value'] > 0
        assert baselines['last_value']['value'] > 0
        assert 'slope' in baselines['linear_trend']
    
    def test_compare_with_baselines(self, sample_config, sample_data):
        """Testa comparação com baselines."""
        X, y = sample_data
        
        # Criar previsões do modelo (ligeiramente melhores que média)
        model_pred = np.full(len(y), y.mean()) + np.random.normal(0, 1, len(y))
        
        # Criar previsões dos baselines
        baseline_pred = {
            'historical_mean': np.full(len(y), y.mean()),
            'last_value': np.full(len(y), y.iloc[-1])
        }
        
        comparator = BaselineComparator(sample_config)
        baselines = comparator.create_baselines(X, y)
        
        results = comparator.compare_with_baselines(
            model_pred, y.values, baselines, baseline_pred
        )
        
        # Verificar estrutura dos resultados
        assert 'model_metrics' in results
        assert 'baseline_metrics' in results
        assert 'improvements' in results
        assert 'model_rank' in results
        
        # Verificar métricas
        assert 'wmape' in results['model_metrics']
        assert len(results['baseline_metrics']) > 0


# Removed PerformanceVisualizer tests as it's not implemented in the simplified version


class TestValidationManager:
    """Testes para ValidationManager."""
    
    def test_init(self, sample_config):
        """Testa inicialização do gerenciador."""
        manager = ValidationManager(sample_config)
        assert manager.config == sample_config
        assert isinstance(manager.walk_forward_validator, WalkForwardValidator)
        assert isinstance(manager.residual_analyzer, ResidualAnalyzer)
        assert isinstance(manager.baseline_comparator, BaselineComparator)
        # Visualizer not implemented in simplified version
    
    @patch('matplotlib.pyplot.savefig')
    def test_run_complete_validation(self, mock_savefig, sample_config, sample_data):
        """Testa validação completa."""
        X, y = sample_data
        model = MockModel(sample_config)
        model.fit(X.drop(['data', 'segmento'], axis=1), y)
        
        manager = ValidationManager(sample_config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            results = manager.run_complete_validation(
                model, X, y, 'data', 'segmento', temp_dir
            )
        
        # Verificar estrutura completa dos resultados
        assert 'validation' in results
        assert 'residual_analysis' in results
        assert 'baseline_comparison' in results
        assert 'executive_summary' in results
        
        # Verificar resumo executivo
        summary = results['executive_summary']
        assert 'model_performance' in summary
        assert 'validation_quality' in summary
        assert 'baseline_comparison' in summary
        assert 'recommendations' in summary
        
        # Verificar que métricas estão presentes
        assert 'wmape' in summary['model_performance']
        assert isinstance(summary['recommendations'], list)


class TestIntegration:
    """Testes de integração do módulo de validação."""
    
    def test_full_validation_pipeline(self, sample_config, sample_data):
        """Testa pipeline completo de validação."""
        X, y = sample_data
        
        # Treinar modelo
        model = MockModel(sample_config)
        model.fit(X.drop(['data', 'segmento'], axis=1), y)
        
        # Executar validação completa
        manager = ValidationManager(sample_config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            results = manager.run_complete_validation(
                model, X, y, 'data', 'segmento', temp_dir
            )
        
        # Verificar que o pipeline executou sem erros
        assert results is not None
        assert len(results) == 5  # Todos os componentes principais
        
        # Verificar qualidade dos resultados
        validation_metrics = results['validation']['overall_metrics']
        assert validation_metrics['wmape'] > 0
        assert validation_metrics['mae'] > 0
        
        # Verificar que baselines foram superados (ou pelo menos comparados)
        baseline_rank = results['baseline_comparison']['model_rank']['wmape']
        assert baseline_rank >= 1  # Modelo deve estar rankeado
        
        # Verificar que recomendações foram geradas
        recommendations = results['executive_summary']['recommendations']
        assert isinstance(recommendations, list)


if __name__ == '__main__':
    pytest.main([__file__])