"""
Testes para Sistema de Múltiplas Submissões - Hackathon 2025
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import yaml
import json

from src.utils.submission_manager import (
    SubmissionManager, SubmissionVersionManager, SubmissionValidator, 
    PerformanceComparator, SubmissionMetadata
)
from src.utils.fast_submission_pipeline import (
    FastSubmissionPipeline, FeatureCache, ModelCache
)


@pytest.fixture
def sample_config():
    """Configuração de teste."""
    return {
        'strategies': {
            'test_strategy': {
                'name': 'Test Strategy',
                'description': 'Strategy for testing',
                'models': {
                    'xgboost': {
                        'enabled': True,
                        'weight': 0.6,
                        'hyperparameters': {
                            'n_estimators': 100,
                            'max_depth': 3
                        }
                    },
                    'lightgbm': {
                        'enabled': True,
                        'weight': 0.4,
                        'hyperparameters': {
                            'n_estimators': 100,
                            'max_depth': 3
                        }
                    }
                },
                'features': {
                    'lag_periods': [1, 2],
                    'rolling_windows': [4, 8],
                    'max_features': 20
                },
                'post_processing': {
                    'smoothing_factor': 0.1,
                    'outlier_cap_multiplier': 2.5
                }
            }
        },
        'submission': {
            'output_dir': 'test_submissions',
            'filename_template': 'test_{strategy}_{version}_{timestamp}',
            'formats': ['csv'],
            'validation': {
                'check_format': True,
                'check_completeness': True
            },
            'backup': {
                'enabled': False,
                'keep_versions': 5
            }
        },
        'performance_comparison': {
            'metrics': {
                'primary': 'wmape',
                'secondary': ['mae', 'rmse']
            }
        },
        'fast_execution': {
            'optimizations': {
                'use_cached_features': True,
                'use_cached_models': True,
                'parallel_training': False
            },
            'cache': {
                'features_cache_dir': 'test_cache/features',
                'models_cache_dir': 'test_cache/models',
                'ttl_hours': 1
            }
        }
    }


@pytest.fixture
def sample_predictions():
    """Dados de previsão de teste."""
    np.random.seed(42)
    
    # Gerar dados para 5 semanas, 10 PDVs, 20 produtos
    data = []
    for semana in range(1, 6):
        for pdv in range(1, 11):
            for produto in range(1, 21):
                quantidade = np.random.poisson(50)  # Valores positivos
                data.append({
                    'semana': semana,
                    'pdv': pdv,
                    'produto': produto,
                    'quantidade': quantidade
                })
    
    return pd.DataFrame(data)


@pytest.fixture
def temp_dir():
    """Diretório temporário para testes."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


class TestSubmissionValidator:
    """Testes para validador de submissões."""
    
    def test_valid_submission(self, sample_predictions):
        """Testa validação de submissão válida."""
        validator = SubmissionValidator()
        result = validator.validate_submission(sample_predictions)
        
        assert result['is_valid'] is True
        assert len(result['errors']) == 0
        assert 'total_records' in result['stats']
    
    def test_missing_columns(self):
        """Testa validação com colunas faltantes."""
        df = pd.DataFrame({
            'semana': [1, 2],
            'pdv': [1, 2]
            # Faltando 'produto' e 'quantidade'
        })
        
        validator = SubmissionValidator()
        result = validator.validate_submission(df)
        
        assert result['is_valid'] is False
        assert any('faltantes' in error for error in result['errors'])
    
    def test_negative_values(self, sample_predictions):
        """Testa validação com valores negativos."""
        sample_predictions.loc[0, 'quantidade'] = -10
        
        validator = SubmissionValidator()
        result = validator.validate_submission(sample_predictions)
        
        assert result['is_valid'] is False
        assert any('negativos' in error for error in result['errors'])
    
    def test_wrong_data_types(self):
        """Testa validação com tipos de dados incorretos."""
        df = pd.DataFrame({
            'semana': ['1', '2'],  # String em vez de int
            'pdv': [1, 2],
            'produto': [1, 2],
            'quantidade': [10, 20]
        })
        
        validator = SubmissionValidator()
        result = validator.validate_submission(df)
        
        assert result['is_valid'] is False


class TestSubmissionVersionManager:
    """Testes para gerenciador de versões."""
    
    def test_version_increment(self, temp_dir):
        """Testa incremento de versões."""
        manager = SubmissionVersionManager(temp_dir)
        
        # Primeira versão
        v1 = manager.get_next_version('test_strategy')
        assert v1 == 'v1.0.0'
        
        # Segunda versão (patch)
        v2 = manager.get_next_version('test_strategy')
        assert v2 == 'v1.0.1'
        
        # Versão minor
        v3 = manager.increment_minor_version('test_strategy')
        assert v3 == 'v1.1.0'
        
        # Versão major
        v4 = manager.increment_major_version('test_strategy')
        assert v4 == 'v2.0.0'
    
    def test_metadata_save_load(self, temp_dir):
        """Testa salvamento e carregamento de metadados."""
        manager = SubmissionVersionManager(temp_dir)
        
        metadata = SubmissionMetadata(
            strategy_name='test_strategy',
            version='v1.0.0',
            timestamp='2025-01-10T10:00:00',
            config_hash='abc123',
            performance_metrics={'wmape': 0.1234}
        )
        
        # Salvar
        manager.save_metadata(metadata)
        
        # Carregar
        loaded = manager.load_metadata('test_strategy', 'v1.0.0')
        
        assert loaded is not None
        assert loaded.strategy_name == 'test_strategy'
        assert loaded.version == 'v1.0.0'
        assert loaded.performance_metrics['wmape'] == 0.1234
    
    def test_list_submissions(self, temp_dir):
        """Testa listagem de submissões."""
        manager = SubmissionVersionManager(temp_dir)
        
        # Criar algumas submissões
        for i in range(3):
            metadata = SubmissionMetadata(
                strategy_name='test_strategy',
                version=f'v1.0.{i}',
                timestamp=f'2025-01-10T10:0{i}:00',
                config_hash=f'hash{i}',
                performance_metrics={'wmape': 0.1 + i * 0.01}
            )
            manager.save_metadata(metadata)
        
        # Listar todas
        submissions = manager.list_submissions()
        assert len(submissions) == 3
        
        # Listar por estratégia
        strategy_submissions = manager.list_submissions('test_strategy')
        assert len(strategy_submissions) == 3
        
        # Listar estratégia inexistente
        empty_submissions = manager.list_submissions('nonexistent')
        assert len(empty_submissions) == 0
    
    def test_best_submission(self, temp_dir):
        """Testa busca pela melhor submissão."""
        manager = SubmissionVersionManager(temp_dir)
        
        # Criar submissões com diferentes performances
        performances = [0.15, 0.12, 0.18, 0.10]  # Melhor é 0.10
        
        for i, wmape in enumerate(performances):
            metadata = SubmissionMetadata(
                strategy_name='test_strategy',
                version=f'v1.0.{i}',
                timestamp=f'2025-01-10T10:0{i}:00',
                config_hash=f'hash{i}',
                performance_metrics={'wmape': wmape}
            )
            manager.save_metadata(metadata)
        
        # Buscar melhor
        best = manager.get_best_submission('test_strategy', 'wmape')
        
        assert best is not None
        assert best.performance_metrics['wmape'] == 0.10
        assert best.version == 'v1.0.3'


class TestPerformanceComparator:
    """Testes para comparador de performance."""
    
    def test_compare_submissions(self, temp_dir):
        """Testa comparação de submissões."""
        manager = SubmissionVersionManager(temp_dir)
        comparator = PerformanceComparator(manager)
        
        # Criar submissões de teste
        submissions = []
        for i in range(3):
            metadata = SubmissionMetadata(
                strategy_name=f'strategy_{i}',
                version='v1.0.0',
                timestamp=f'2025-01-10T10:0{i}:00',
                config_hash=f'hash{i}',
                performance_metrics={
                    'wmape': 0.1 + i * 0.01,
                    'mae': 10 + i * 2
                }
            )
            submissions.append(metadata)
        
        # Comparar
        comparison_df = comparator.compare_submissions(submissions)
        
        assert len(comparison_df) == 3
        assert 'metric_wmape' in comparison_df.columns
        assert 'metric_mae' in comparison_df.columns
        
        # Verificar ordenação (menor WMAPE primeiro)
        assert comparison_df.iloc[0]['metric_wmape'] == 0.1
        assert comparison_df.iloc[-1]['metric_wmape'] == 0.12


class TestFeatureCache:
    """Testes para cache de features."""
    
    def test_cache_key_generation(self, temp_dir):
        """Testa geração de chave de cache."""
        cache = FeatureCache(temp_dir)
        
        data_hash = 'abc123'
        config = {'lag_periods': [1, 2], 'max_features': 20}
        
        key1 = cache.get_cache_key(data_hash, config)
        key2 = cache.get_cache_key(data_hash, config)
        
        # Mesma configuração deve gerar mesma chave
        assert key1 == key2
        
        # Configuração diferente deve gerar chave diferente
        config2 = {'lag_periods': [1, 3], 'max_features': 20}
        key3 = cache.get_cache_key(data_hash, config2)
        assert key1 != key3
    
    def test_cache_save_load(self, temp_dir):
        """Testa salvamento e carregamento do cache."""
        cache = FeatureCache(temp_dir)
        
        # Dados de teste
        features_df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })
        metadata = {'test': 'data'}
        cache_key = 'test_key'
        
        # Salvar
        cache.save_features(cache_key, features_df, metadata)
        
        # Verificar se existe
        assert cache.is_cache_valid(cache_key, ttl_hours=24)
        
        # Carregar
        loaded_df, loaded_metadata = cache.load_features(cache_key)
        
        pd.testing.assert_frame_equal(features_df, loaded_df)
        assert loaded_metadata['test'] == 'data'


class TestModelCache:
    """Testes para cache de modelos."""
    
    def test_model_key_generation(self, temp_dir):
        """Testa geração de chave para modelo."""
        cache = ModelCache(temp_dir)
        
        model_type = 'xgboost'
        config = {'n_estimators': 100, 'max_depth': 3}
        data_hash = 'abc123'
        
        key1 = cache.get_model_key(model_type, config, data_hash)
        key2 = cache.get_model_key(model_type, config, data_hash)
        
        # Mesma configuração deve gerar mesma chave
        assert key1 == key2
        
        # Configuração diferente deve gerar chave diferente
        config2 = {'n_estimators': 200, 'max_depth': 3}
        key3 = cache.get_model_key(model_type, config2, data_hash)
        assert key1 != key3
    
    def test_model_save_load(self, temp_dir):
        """Testa salvamento e carregamento de modelo."""
        cache = ModelCache(temp_dir)
        
        # Mock de modelo
        mock_model = Mock()
        mock_model.predict = Mock(return_value=[1, 2, 3])
        
        metadata = {'model_type': 'test'}
        model_key = 'test_model_key'
        
        # Salvar
        cache.save_model(model_key, mock_model, metadata)
        
        # Verificar se existe
        assert cache.is_model_cached(model_key, ttl_hours=24)
        
        # Carregar
        loaded_model, loaded_metadata = cache.load_model(model_key)
        
        assert loaded_model.predict([1, 2, 3]) == [1, 2, 3]
        assert loaded_metadata['model_type'] == 'test'


class TestSubmissionManager:
    """Testes para gerenciador de submissões."""
    
    def test_initialization(self, temp_dir, sample_config):
        """Testa inicialização do gerenciador."""
        # Salvar configuração temporária
        config_file = Path(temp_dir) / 'test_config.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(sample_config, f)
        
        manager = SubmissionManager(str(config_file))
        
        assert 'test_strategy' in manager.list_strategies()
        
        strategy_config = manager.get_strategy_config('test_strategy')
        assert strategy_config['name'] == 'Test Strategy'
    
    def test_create_submission(self, temp_dir, sample_config, sample_predictions):
        """Testa criação de submissão."""
        # Salvar configuração temporária
        config_file = Path(temp_dir) / 'test_config.yaml'
        
        # Ajustar diretório de saída
        sample_config['submission']['output_dir'] = str(Path(temp_dir) / 'submissions')
        
        with open(config_file, 'w') as f:
            yaml.dump(sample_config, f)
        
        manager = SubmissionManager(str(config_file))
        
        # Criar submissão
        performance_metrics = {'wmape': 0.1234, 'mae': 45.67}
        
        submission = manager.create_submission(
            strategy_name='test_strategy',
            predictions_df=sample_predictions,
            performance_metrics=performance_metrics
        )
        
        assert submission.strategy_name == 'test_strategy'
        assert submission.version == 'v1.0.0'
        assert submission.performance_metrics['wmape'] == 0.1234
        assert submission.file_path is not None
        
        # Verificar se arquivo foi criado
        assert Path(submission.file_path).exists()


@pytest.mark.integration
class TestFastSubmissionPipeline:
    """Testes de integração para pipeline rápido."""
    
    @patch('src.utils.fast_submission_pipeline.FastSubmissionPipeline._create_features')
    @patch('src.utils.fast_submission_pipeline.FastSubmissionPipeline._train_model')
    def test_process_strategy_with_mocks(self, mock_train, mock_features, 
                                       temp_dir, sample_config, sample_predictions):
        """Testa processamento de estratégia com mocks."""
        # Configurar mocks
        mock_features.return_value = sample_predictions
        mock_model = Mock()
        mock_model.predict = Mock(return_value=sample_predictions['quantidade'].values)
        mock_train.return_value = mock_model
        
        # Salvar configuração temporária
        config_file = Path(temp_dir) / 'test_config.yaml'
        sample_config['submission']['output_dir'] = str(Path(temp_dir) / 'submissions')
        sample_config['fast_execution']['cache']['features_cache_dir'] = str(Path(temp_dir) / 'cache/features')
        sample_config['fast_execution']['cache']['models_cache_dir'] = str(Path(temp_dir) / 'cache/models')
        
        with open(config_file, 'w') as f:
            yaml.dump(sample_config, f)
        
        # Criar pipeline
        pipeline = FastSubmissionPipeline(str(config_file))
        
        # Processar estratégia
        with patch.object(pipeline, '_calculate_performance_metrics') as mock_metrics:
            mock_metrics.return_value = {'wmape': 0.1234}
            
            submission = pipeline._process_strategy(
                'test_strategy', 
                sample_predictions, 
                'test_hash'
            )
        
        assert submission.strategy_name == 'test_strategy'
        assert submission.performance_metrics['wmape'] == 0.1234


def test_config_validation(sample_config):
    """Testa validação de configuração."""
    # Configuração válida
    assert 'strategies' in sample_config
    assert 'test_strategy' in sample_config['strategies']
    
    # Verificar estrutura da estratégia
    strategy = sample_config['strategies']['test_strategy']
    assert 'name' in strategy
    assert 'models' in strategy
    assert 'features' in strategy


if __name__ == '__main__':
    pytest.main([__file__, '-v'])