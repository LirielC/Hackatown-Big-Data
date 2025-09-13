"""
Testes para o módulo de geração de previsões.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from pathlib import Path

from src.models.prediction import (
    PredictionGenerator, 
    PredictionValidator, 
    PredictionError
)
from src.models.training import BaseModel


class TestPredictionGenerator:
    """Testes para a classe PredictionGenerator."""
    
    @pytest.fixture
    def config(self):
        """Configuração de teste."""
        return {
            'prediction': {
                'target_weeks': [1, 2, 3, 4, 5],
                'post_processing': {
                    'ensure_positive': True,
                    'apply_bounds': True,
                    'max_multiplier': 3.0
                },
                'output': {
                    'format': 'csv',
                    'separator': ';',
                    'encoding': 'utf-8'
                }
            }
        }
    
    @pytest.fixture
    def prediction_generator(self, config):
        """Instância do PredictionGenerator."""
        return PredictionGenerator(config)
    
    @pytest.fixture
    def mock_model(self):
        """Mock de modelo treinado."""
        model = Mock(spec=BaseModel)
        model.is_fitted = True
        model.predict.return_value = np.array([10.5, 20.3, 15.7, 8.2, 25.1])
        return model
    
    @pytest.fixture
    def sample_features(self):
        """DataFrame de features de exemplo."""
        return pd.DataFrame({
            'pdv': ['001', '001', '002', '002', '003'],
            'produto': ['A', 'B', 'A', 'B', 'A'],
            'semana': [1, 1, 2, 2, 3],
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature2': [0.5, 1.5, 2.5, 3.5, 4.5]
        })
    
    @pytest.fixture
    def sample_historical_data(self):
        """Dados históricos de exemplo."""
        return pd.DataFrame({
            'pdv': ['001', '001', '002', '002', '003'] * 10,
            'produto': ['A', 'B', 'A', 'B', 'A'] * 10,
            'quantidade': np.random.randint(1, 50, 50)
        })
    
    def test_init(self, config):
        """Testa inicialização do PredictionGenerator."""
        generator = PredictionGenerator(config)
        
        assert generator.config == config
        assert generator.ensure_positive is True
        assert generator.apply_bounds is True
        assert generator.max_multiplier == 3.0
        assert generator.target_weeks == [1, 2, 3, 4, 5]
    
    def test_generate_predictions_success(self, prediction_generator, mock_model, sample_features):
        """Testa geração bem-sucedida de previsões."""
        predictions = prediction_generator.generate_predictions(mock_model, sample_features)
        
        # Verificar estrutura do resultado
        assert isinstance(predictions, pd.DataFrame)
        assert list(predictions.columns) == ['semana', 'pdv', 'produto', 'quantidade']
        assert len(predictions) == len(sample_features)
        
        # Verificar tipos de dados
        assert predictions['semana'].dtype == int
        assert predictions['quantidade'].dtype == int
        
        # Verificar valores não-negativos
        assert (predictions['quantidade'] >= 0).all()
        
        # Verificar que modelo foi chamado
        mock_model.predict.assert_called_once()
    
    def test_generate_predictions_with_historical_data(self, prediction_generator, mock_model, 
                                                     sample_features, sample_historical_data):
        """Testa geração de previsões com dados históricos."""
        predictions = prediction_generator.generate_predictions(
            mock_model, sample_features, sample_historical_data
        )
        
        assert isinstance(predictions, pd.DataFrame)
        assert len(predictions) == len(sample_features)
        assert (predictions['quantidade'] >= 0).all()
    
    def test_validate_prediction_inputs_model_not_fitted(self, prediction_generator, sample_features):
        """Testa validação com modelo não treinado."""
        mock_model = Mock(spec=BaseModel)
        mock_model.is_fitted = False
        
        with pytest.raises(PredictionError, match="Modelo deve estar treinado"):
            prediction_generator._validate_prediction_inputs(mock_model, sample_features)
    
    def test_validate_prediction_inputs_empty_dataframe(self, prediction_generator, mock_model):
        """Testa validação com DataFrame vazio."""
        empty_df = pd.DataFrame()
        
        with pytest.raises(PredictionError, match="DataFrame de features está vazio"):
            prediction_generator._validate_prediction_inputs(mock_model, empty_df)
    
    def test_validate_prediction_inputs_missing_columns(self, prediction_generator, mock_model):
        """Testa validação com colunas obrigatórias ausentes."""
        invalid_df = pd.DataFrame({'feature1': [1, 2, 3]})
        
        with pytest.raises(PredictionError, match="Colunas obrigatórias ausentes"):
            prediction_generator._validate_prediction_inputs(mock_model, invalid_df)
    
    def test_generate_raw_predictions(self, prediction_generator, mock_model, sample_features):
        """Testa geração de previsões brutas."""
        predictions = prediction_generator._generate_raw_predictions(mock_model, sample_features)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(sample_features)
        mock_model.predict.assert_called_once()
    
    def test_prepare_prediction_features(self, prediction_generator, sample_features):
        """Testa preparação de features para predição."""
        features = prediction_generator._prepare_prediction_features(sample_features)
        
        # Verificar que colunas de identificação foram removidas
        excluded_cols = ['pdv', 'produto', 'semana', 'quantidade', 'data', 'data_semana']
        for col in excluded_cols:
            assert col not in features.columns
        
        # Verificar que features foram mantidas
        assert 'feature1' in features.columns
        assert 'feature2' in features.columns
        assert len(features) == len(sample_features)
    
    def test_prepare_prediction_features_no_features(self, prediction_generator):
        """Testa preparação quando não há features válidas."""
        invalid_df = pd.DataFrame({
            'pdv': ['001', '002'],
            'produto': ['A', 'B'],
            'semana': [1, 2]
        })
        
        with pytest.raises(PredictionError, match="Nenhuma feature válida encontrada"):
            prediction_generator._prepare_prediction_features(invalid_df)
    
    def test_apply_post_processing_ensure_positive(self, prediction_generator, sample_features):
        """Testa pós-processamento para garantir valores positivos."""
        # Previsões com valores negativos
        predictions = np.array([-5.0, 10.0, -2.0, 15.0, 8.0])
        
        processed = prediction_generator._apply_post_processing(predictions, sample_features)
        
        # Verificar que valores negativos foram convertidos para zero
        assert (processed >= 0).all()
        assert processed[0] == 0  # -5.0 -> 0
        assert processed[2] == 0  # -2.0 -> 0
        assert processed[1] == 10  # 10.0 -> 10 (arredondado)
    
    def test_apply_post_processing_with_historical_bounds(self, prediction_generator, 
                                                        sample_features, sample_historical_data):
        """Testa pós-processamento com limites históricos."""
        # Previsões muito altas
        predictions = np.array([1000.0, 2000.0, 500.0, 1500.0, 800.0])
        
        processed = prediction_generator._apply_post_processing(
            predictions, sample_features, sample_historical_data
        )
        
        # Verificar que valores foram limitados
        assert (processed <= predictions).all()
        assert isinstance(processed, np.ndarray)
    
    def test_apply_smoothing(self, prediction_generator):
        """Testa suavização de valores extremos."""
        # Criar array com valores extremos
        predictions = np.array([10, 20, 30, 1000, 15, 25, 2000, 18])
        
        smoothed = prediction_generator._apply_smoothing(predictions)
        
        # Verificar que valores extremos foram reduzidos
        assert smoothed.max() < predictions.max()
        assert len(smoothed) == len(predictions)
    
    def test_format_predictions(self, prediction_generator, sample_features):
        """Testa formatação das previsões."""
        predictions = np.array([10.5, 20.3, 15.7, 8.2, 25.1])
        
        formatted = prediction_generator._format_predictions(predictions, sample_features)
        
        # Verificar estrutura
        assert list(formatted.columns) == ['semana', 'pdv', 'produto', 'quantidade']
        assert len(formatted) == len(sample_features)
        
        # Verificar tipos
        assert formatted['semana'].dtype == int
        assert formatted['quantidade'].dtype == int
        
        # Verificar ordenação
        assert formatted['semana'].is_monotonic_increasing or len(formatted['semana'].unique()) == 1
    
    def test_validate_predictions_success(self, prediction_generator):
        """Testa validação bem-sucedida de previsões."""
        valid_predictions = pd.DataFrame({
            'semana': [1, 1, 2, 2, 3],
            'pdv': ['001', '002', '001', '002', '001'],
            'produto': ['A', 'A', 'B', 'B', 'A'],
            'quantidade': [10, 15, 20, 8, 12]
        })
        
        # Não deve levantar exceção
        prediction_generator._validate_predictions(valid_predictions)
    
    def test_validate_predictions_empty_dataframe(self, prediction_generator):
        """Testa validação com DataFrame vazio."""
        empty_df = pd.DataFrame()
        
        with pytest.raises(PredictionError, match="DataFrame de previsões está vazio"):
            prediction_generator._validate_predictions(empty_df)
    
    def test_validate_predictions_missing_columns(self, prediction_generator):
        """Testa validação com colunas ausentes."""
        invalid_df = pd.DataFrame({
            'semana': [1, 2],
            'pdv': ['001', '002']
            # Faltam 'produto' e 'quantidade'
        })
        
        with pytest.raises(PredictionError, match="Colunas obrigatórias ausentes"):
            prediction_generator._validate_predictions(invalid_df)
    
    def test_validate_predictions_null_values(self, prediction_generator):
        """Testa validação com valores nulos."""
        invalid_df = pd.DataFrame({
            'semana': [1, 2, 3],
            'pdv': ['001', '002', None],
            'produto': ['A', 'B', 'C'],
            'quantidade': [10, 15, 20]
        })
        
        with pytest.raises(PredictionError, match="Valores nulos encontrados"):
            prediction_generator._validate_predictions(invalid_df)
    
    def test_validate_predictions_negative_quantities(self, prediction_generator):
        """Testa validação com quantidades negativas."""
        invalid_df = pd.DataFrame({
            'semana': [1, 2, 3],
            'pdv': ['001', '002', '003'],
            'produto': ['A', 'B', 'C'],
            'quantidade': [10, -5, 20]
        })
        
        with pytest.raises(PredictionError, match="quantidades negativas"):
            prediction_generator._validate_predictions(invalid_df)
    
    def test_validate_predictions_duplicates(self, prediction_generator):
        """Testa validação com duplicatas."""
        invalid_df = pd.DataFrame({
            'semana': [1, 1, 2],
            'pdv': ['001', '001', '002'],
            'produto': ['A', 'A', 'B'],  # Duplicata: semana 1, pdv 001, produto A
            'quantidade': [10, 15, 20]
        })
        
        with pytest.raises(PredictionError, match="combinações duplicadas"):
            prediction_generator._validate_predictions(invalid_df)
    
    def test_save_predictions_csv(self, prediction_generator):
        """Testa salvamento em formato CSV."""
        predictions_df = pd.DataFrame({
            'semana': [1, 2, 3],
            'pdv': ['001', '002', '003'],
            'produto': ['A', 'B', 'C'],
            'quantidade': [10, 15, 20]
        })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'predictions.csv')
            
            saved_path = prediction_generator.save_predictions(predictions_df, output_path, 'csv')
            
            # Verificar que arquivo foi criado
            assert os.path.exists(saved_path)
            assert saved_path == output_path
            
            # Verificar conteúdo
            loaded_df = pd.read_csv(saved_path, sep=';', dtype={'pdv': str, 'produto': str})
            pd.testing.assert_frame_equal(loaded_df, predictions_df)
    
    def test_save_predictions_parquet(self, prediction_generator):
        """Testa salvamento em formato Parquet."""
        predictions_df = pd.DataFrame({
            'semana': [1, 2, 3],
            'pdv': ['001', '002', '003'],
            'produto': ['A', 'B', 'C'],
            'quantidade': [10, 15, 20]
        })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'predictions.parquet')
            
            saved_path = prediction_generator.save_predictions(predictions_df, output_path, 'parquet')
            
            # Verificar que arquivo foi criado
            assert os.path.exists(saved_path)
            
            # Verificar conteúdo
            loaded_df = pd.read_parquet(saved_path)
            pd.testing.assert_frame_equal(loaded_df, predictions_df)
    
    def test_save_predictions_invalid_format(self, prediction_generator):
        """Testa salvamento com formato inválido."""
        predictions_df = pd.DataFrame({
            'semana': [1, 2],
            'pdv': ['001', '002'],
            'produto': ['A', 'B'],
            'quantidade': [10, 15]
        })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'predictions.txt')
            
            with pytest.raises(PredictionError, match="Formato não suportado"):
                prediction_generator.save_predictions(predictions_df, output_path, 'txt')
    
    def test_generate_prediction_summary(self, prediction_generator):
        """Testa geração de resumo das previsões."""
        predictions_df = pd.DataFrame({
            'semana': [1, 1, 2, 2, 3, 3],
            'pdv': ['001', '002', '001', '002', '001', '002'],
            'produto': ['A', 'A', 'B', 'B', 'A', 'A'],
            'quantidade': [10, 15, 20, 8, 12, 25]
        })
        
        summary = prediction_generator.generate_prediction_summary(predictions_df)
        
        # Verificar estrutura do resumo
        assert 'total_predictions' in summary
        assert 'prediction_period' in summary
        assert 'coverage' in summary
        assert 'quantity_statistics' in summary
        assert 'distribution' in summary
        assert 'weekly_statistics' in summary
        
        # Verificar valores
        assert summary['total_predictions'] == 6
        assert summary['coverage']['unique_pdvs'] == 2
        assert summary['coverage']['unique_products'] == 2
        assert summary['quantity_statistics']['total_quantity'] == 90


class TestPredictionValidator:
    """Testes para a classe PredictionValidator."""
    
    def test_validate_prediction_format_success(self):
        """Testa validação bem-sucedida de formato."""
        valid_df = pd.DataFrame({
            'semana': [1, 2, 3],
            'pdv': ['001', '002', '003'],
            'produto': ['A', 'B', 'C'],
            'quantidade': [10, 15, 20]
        })
        
        result = PredictionValidator.validate_prediction_format(valid_df)
        
        assert result['is_valid'] is True
        assert len(result['errors']) == 0
        assert 'format_check' in result
    
    def test_validate_prediction_format_missing_columns(self):
        """Testa validação com colunas ausentes."""
        invalid_df = pd.DataFrame({
            'semana': [1, 2],
            'pdv': ['001', '002']
            # Faltam 'produto' e 'quantidade'
        })
        
        result = PredictionValidator.validate_prediction_format(invalid_df)
        
        assert result['is_valid'] is False
        assert any('Colunas obrigatórias ausentes' in error for error in result['errors'])
    
    def test_validate_prediction_format_invalid_quantity_type(self):
        """Testa validação com tipo inválido para quantidade."""
        invalid_df = pd.DataFrame({
            'semana': [1, 2, 3],
            'pdv': ['001', '002', '003'],
            'produto': ['A', 'B', 'C'],
            'quantidade': ['10', '15', '20']  # String em vez de numérico
        })
        
        result = PredictionValidator.validate_prediction_format(invalid_df)
        
        assert result['is_valid'] is False
        assert any('deve ser numérica' in error for error in result['errors'])
    
    def test_validate_prediction_format_negative_quantities(self):
        """Testa validação com quantidades negativas."""
        invalid_df = pd.DataFrame({
            'semana': [1, 2, 3],
            'pdv': ['001', '002', '003'],
            'produto': ['A', 'B', 'C'],
            'quantidade': [10, -5, 20]
        })
        
        result = PredictionValidator.validate_prediction_format(invalid_df)
        
        assert result['is_valid'] is False
        assert any('quantidades negativas' in error for error in result['errors'])
    
    def test_validate_prediction_format_null_quantities(self):
        """Testa validação com quantidades nulas."""
        invalid_df = pd.DataFrame({
            'semana': [1, 2, 3],
            'pdv': ['001', '002', '003'],
            'produto': ['A', 'B', 'C'],
            'quantidade': [10, np.nan, 20]
        })
        
        result = PredictionValidator.validate_prediction_format(invalid_df)
        
        assert result['is_valid'] is False
        assert any('valores nulos' in error for error in result['errors'])
    
    def test_validate_prediction_format_duplicates(self):
        """Testa validação com duplicatas."""
        invalid_df = pd.DataFrame({
            'semana': [1, 1, 2],
            'pdv': ['001', '001', '002'],
            'produto': ['A', 'A', 'B'],  # Duplicata
            'quantidade': [10, 15, 20]
        })
        
        result = PredictionValidator.validate_prediction_format(invalid_df)
        
        assert result['is_valid'] is False
        assert any('combinações duplicadas' in error for error in result['errors'])
    
    def test_compare_with_baseline(self):
        """Testa comparação com baseline."""
        predictions_df = pd.DataFrame({
            'quantidade': [10, 15, 20]
        })
        
        baseline_df = pd.DataFrame({
            'quantidade': [8, 12, 18]
        })
        
        comparison = PredictionValidator.compare_with_baseline(predictions_df, baseline_df)
        
        assert 'prediction_total' in comparison
        assert 'baseline_total' in comparison
        assert 'difference_absolute' in comparison
        assert 'difference_percentage' in comparison
        
        assert comparison['prediction_total'] == 45
        assert comparison['baseline_total'] == 38
        assert comparison['difference_absolute'] == 7
    
    def test_compare_with_baseline_no_quantity_column(self):
        """Testa comparação com baseline sem coluna quantidade."""
        predictions_df = pd.DataFrame({
            'quantidade': [10, 15, 20]
        })
        
        baseline_df = pd.DataFrame({
            'other_column': [1, 2, 3]
        })
        
        comparison = PredictionValidator.compare_with_baseline(predictions_df, baseline_df)
        
        assert comparison['baseline_total'] == 0
        assert comparison['difference_absolute'] == 45
    
    def test_validate_business_rules_success(self):
        """Testa validação bem-sucedida de regras de negócio."""
        valid_df = pd.DataFrame({
            'quantidade': [10, 15, 20, 12, 18, 25, 8, 22]
        })
        
        result = PredictionValidator.validate_business_rules(valid_df)
        
        assert result['is_valid'] is True
        assert 'statistics' in result
    
    def test_validate_business_rules_too_many_zeros(self):
        """Testa validação com muitas previsões zero."""
        invalid_df = pd.DataFrame({
            'quantidade': [0, 0, 0, 0, 0, 0, 10, 15]  # 75% zeros
        })
        
        result = PredictionValidator.validate_business_rules(invalid_df)
        
        assert result['is_valid'] is False
        assert any('Muitas previsões zero' in violation for violation in result['violations'])
    
    def test_validate_business_rules_high_quantities(self):
        """Testa validação com quantidades suspeitas."""
        # Criar dados com outliers extremos
        normal_data = [10, 15, 20, 12, 18] * 20  # 100 valores normais
        outliers = [10000, 15000]  # 2 outliers extremos
        
        invalid_df = pd.DataFrame({
            'quantidade': normal_data + outliers
        })
        
        result = PredictionValidator.validate_business_rules(invalid_df)
        
        # Pode ou não ser inválido dependendo da distribuição, mas deve detectar outliers
        assert 'statistics' in result
        assert 'high_quantity_count' in result['statistics']


class TestPredictionIntegration:
    """Testes de integração para o módulo de predição."""
    
    @pytest.fixture
    def full_config(self):
        """Configuração completa para testes de integração."""
        return {
            'prediction': {
                'target_weeks': [1, 2, 3, 4, 5],
                'post_processing': {
                    'ensure_positive': True,
                    'apply_bounds': True,
                    'max_multiplier': 3.0
                },
                'output': {
                    'format': 'csv',
                    'separator': ';',
                    'encoding': 'utf-8',
                    'filename': 'submission_{timestamp}.csv'
                }
            }
        }
    
    @pytest.fixture
    def complete_features_df(self):
        """DataFrame completo de features para teste de integração."""
        np.random.seed(42)
        
        # Criar combinações únicas de PDV/produto/semana
        data = []
        pdvs = ['001', '002', '003']
        produtos = ['A', 'B', 'C', 'D']
        semanas = [1, 2, 3, 4, 5]
        
        for semana in semanas:
            for pdv in pdvs:
                for produto in produtos:
                    data.append({
                        'pdv': pdv,
                        'produto': produto,
                        'semana': semana,
                        'lag_1': np.random.normal(10, 3),
                        'lag_2': np.random.normal(8, 2),
                        'media_movel_4': np.random.normal(12, 4),
                        'sazonalidade': np.random.uniform(0.8, 1.2),
                        'tendencia': np.random.normal(0, 0.1)
                    })
        
        return pd.DataFrame(data)
    
    def test_end_to_end_prediction_pipeline(self, full_config, complete_features_df):
        """Testa pipeline completo de predição."""
        # Criar mock de modelo
        mock_model = Mock(spec=BaseModel)
        mock_model.is_fitted = True
        mock_model.predict.return_value = np.random.normal(15, 5, len(complete_features_df))
        
        # Criar gerador de previsões
        generator = PredictionGenerator(full_config)
        
        # Gerar previsões
        predictions = generator.generate_predictions(mock_model, complete_features_df)
        
        # Validar resultado
        assert isinstance(predictions, pd.DataFrame)
        assert len(predictions) == len(complete_features_df)
        assert (predictions['quantidade'] >= 0).all()
        
        # Validar formato
        validator_result = PredictionValidator.validate_prediction_format(predictions)
        assert validator_result['is_valid'] is True
        
        # Gerar resumo
        summary = generator.generate_prediction_summary(predictions)
        assert summary['total_predictions'] == len(predictions)
    
    def test_prediction_with_file_output(self, full_config, complete_features_df):
        """Testa predição com salvamento em arquivo."""
        mock_model = Mock(spec=BaseModel)
        mock_model.is_fitted = True
        mock_model.predict.return_value = np.random.normal(15, 5, len(complete_features_df))
        
        generator = PredictionGenerator(full_config)
        predictions = generator.generate_predictions(mock_model, complete_features_df)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'test_predictions.csv')
            
            # Salvar previsões
            saved_path = generator.save_predictions(predictions, output_path)
            
            # Verificar arquivo
            assert os.path.exists(saved_path)
            
            # Carregar e validar
            loaded_predictions = pd.read_csv(saved_path, sep=';', dtype={'pdv': str, 'produto': str})
            pd.testing.assert_frame_equal(loaded_predictions, predictions)