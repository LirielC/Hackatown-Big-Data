"""
Testes para o módulo de formatação de saída.

Este módulo testa a geração de arquivos CSV e Parquet no formato específico
do hackathon, incluindo validações de formato e integridade.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
from unittest.mock import patch, MagicMock

from src.models.output_formatter import (
    SubmissionFormatter, 
    SubmissionValidator,
    OutputFormatterError
)


class TestSubmissionFormatter:
    """Testes para a classe SubmissionFormatter."""
    
    @pytest.fixture
    def sample_config(self):
        """Configuração de exemplo para testes."""
        return {
            'output': {
                'csv_separator': ';',
                'csv_encoding': 'utf-8',
                'csv_decimal': '.',
                'parquet_compression': 'snappy',
                'validate_format': True,
                'add_timestamp': False  # Desabilitar para testes determinísticos
            }
        }
    
    @pytest.fixture
    def sample_predictions(self):
        """DataFrame de previsões de exemplo."""
        return pd.DataFrame({
            'semana': [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'pdv': ['001', '001', '002', '001', '001', '002', '001', '001', '002'],
            'produto': ['A', 'B', 'A', 'A', 'B', 'A', 'A', 'B', 'A'],
            'quantidade': [10, 15, 8, 12, 18, 9, 11, 16, 7]
        })
    
    @pytest.fixture
    def invalid_predictions(self):
        """DataFrame com dados inválidos para testes."""
        return pd.DataFrame({
            'semana': [1, 2, 3],
            'pdv': ['001', '002', '003'],
            'produto': ['A', 'B', 'C'],
            'quantidade': [-5, 10, None]  # Valor negativo e nulo
        })
    
    def test_formatter_initialization(self, sample_config):
        """Testa inicialização do formatador."""
        formatter = SubmissionFormatter(sample_config)
        
        assert formatter.csv_separator == ';'
        assert formatter.csv_encoding == 'utf-8'
        assert formatter.parquet_compression == 'snappy'
        assert formatter.validate_format is True
    
    def test_format_submission_csv_success(self, sample_config, sample_predictions):
        """Testa formatação bem-sucedida de CSV."""
        formatter = SubmissionFormatter(sample_config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'test_submission.csv')
            
            result_path = formatter.format_submission_csv(sample_predictions, output_path)
            
            # Verificar se arquivo foi criado
            assert Path(result_path).exists()
            
            # Verificar conteúdo do arquivo
            loaded_df = pd.read_csv(result_path, sep=';', encoding='utf-8')
            
            assert len(loaded_df) == len(sample_predictions)
            assert list(loaded_df.columns) == ['semana', 'pdv', 'produto', 'quantidade']
            assert loaded_df['quantidade'].dtype == 'int64'
    
    def test_format_submission_parquet_success(self, sample_config, sample_predictions):
        """Testa formatação bem-sucedida de Parquet."""
        formatter = SubmissionFormatter(sample_config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'test_submission.parquet')
            
            result_path = formatter.format_submission_parquet(sample_predictions, output_path)
            
            # Verificar se arquivo foi criado
            assert Path(result_path).exists()
            
            # Verificar conteúdo do arquivo
            loaded_df = pd.read_parquet(result_path)
            
            assert len(loaded_df) == len(sample_predictions)
            assert list(loaded_df.columns) == ['semana', 'pdv', 'produto', 'quantidade']
    
    def test_format_csv_with_invalid_data(self, sample_config, invalid_predictions):
        """Testa formatação CSV com dados inválidos."""
        formatter = SubmissionFormatter(sample_config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'test_invalid.csv')
            
            with pytest.raises(OutputFormatterError):
                formatter.format_submission_csv(invalid_predictions, output_path)
    
    def test_format_parquet_with_invalid_data(self, sample_config, invalid_predictions):
        """Testa formatação Parquet com dados inválidos."""
        formatter = SubmissionFormatter(sample_config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'test_invalid.parquet')
            
            with pytest.raises(OutputFormatterError):
                formatter.format_submission_parquet(invalid_predictions, output_path)
    
    def test_validate_submission_format_success(self, sample_config, sample_predictions):
        """Testa validação bem-sucedida de formato."""
        formatter = SubmissionFormatter(sample_config)
        
        # Não deve levantar exceção
        formatter._validate_submission_format(sample_predictions)
    
    def test_validate_submission_format_missing_columns(self, sample_config):
        """Testa validação com colunas ausentes."""
        formatter = SubmissionFormatter(sample_config)
        
        df_missing_cols = pd.DataFrame({
            'semana': [1, 2],
            'pdv': ['001', '002']
            # Faltam 'produto' e 'quantidade'
        })
        
        with pytest.raises(OutputFormatterError, match="Colunas obrigatórias ausentes"):
            formatter._validate_submission_format(df_missing_cols)
    
    def test_validate_submission_format_null_values(self, sample_config):
        """Testa validação com valores nulos."""
        formatter = SubmissionFormatter(sample_config)
        
        df_with_nulls = pd.DataFrame({
            'semana': [1, 2, 3],
            'pdv': ['001', '002', None],
            'produto': ['A', 'B', 'C'],
            'quantidade': [10, 15, 20]
        })
        
        with pytest.raises(OutputFormatterError, match="Valores nulos encontrados"):
            formatter._validate_submission_format(df_with_nulls)
    
    def test_validate_submission_format_negative_quantities(self, sample_config):
        """Testa validação com quantidades negativas."""
        formatter = SubmissionFormatter(sample_config)
        
        df_negative = pd.DataFrame({
            'semana': [1, 2, 3],
            'pdv': ['001', '002', '003'],
            'produto': ['A', 'B', 'C'],
            'quantidade': [10, -5, 20]
        })
        
        with pytest.raises(OutputFormatterError, match="quantidades negativas"):
            formatter._validate_submission_format(df_negative)
    
    def test_validate_submission_format_duplicates(self, sample_config):
        """Testa validação com duplicatas."""
        formatter = SubmissionFormatter(sample_config)
        
        df_duplicates = pd.DataFrame({
            'semana': [1, 1, 2],
            'pdv': ['001', '001', '002'],
            'produto': ['A', 'A', 'B'],  # Duplicata: semana=1, pdv=001, produto=A
            'quantidade': [10, 15, 20]
        })
        
        with pytest.raises(OutputFormatterError, match="combinações duplicadas"):
            formatter._validate_submission_format(df_duplicates)
    
    def test_validate_submission_format_invalid_weeks(self, sample_config):
        """Testa validação com semanas inválidas."""
        formatter = SubmissionFormatter(sample_config)
        
        df_invalid_weeks = pd.DataFrame({
            'semana': [0, 6, 10],  # Semanas inválidas (devem ser 1-5)
            'pdv': ['001', '002', '003'],
            'produto': ['A', 'B', 'C'],
            'quantidade': [10, 15, 20]
        })
        
        with pytest.raises(OutputFormatterError, match="Semanas inválidas"):
            formatter._validate_submission_format(df_invalid_weeks)
    
    def test_prepare_csv_format(self, sample_config, sample_predictions):
        """Testa preparação de formato CSV."""
        formatter = SubmissionFormatter(sample_config)
        
        csv_df = formatter._prepare_csv_format(sample_predictions)
        
        # Verificar ordem das colunas
        assert list(csv_df.columns) == ['semana', 'pdv', 'produto', 'quantidade']
        
        # Verificar tipos de dados
        assert csv_df['semana'].dtype == 'int64'
        assert csv_df['pdv'].dtype == 'object'  # string
        assert csv_df['produto'].dtype == 'object'  # string
        assert csv_df['quantidade'].dtype == 'int64'
        
        # Verificar ordenação
        assert csv_df['semana'].is_monotonic_increasing
    
    def test_prepare_parquet_format(self, sample_config, sample_predictions):
        """Testa preparação de formato Parquet."""
        formatter = SubmissionFormatter(sample_config)
        
        parquet_df = formatter._prepare_parquet_format(sample_predictions)
        
        # Verificar ordem das colunas
        assert list(parquet_df.columns) == ['semana', 'pdv', 'produto', 'quantidade']
        
        # Verificar tipos otimizados
        assert parquet_df['semana'].dtype == 'int8'
        assert parquet_df['pdv'].dtype.name == 'category'
        assert parquet_df['produto'].dtype.name == 'category'
        assert parquet_df['quantidade'].dtype == 'int32'
    
    def test_add_timestamp_to_path(self, sample_config):
        """Testa adição de timestamp ao caminho."""
        formatter = SubmissionFormatter(sample_config)
        
        original_path = "/path/to/file.csv"
        timestamped_path = formatter._add_timestamp_to_path(original_path)
        
        # Verificar que timestamp foi adicionado
        assert "file_" in timestamped_path
        assert timestamped_path.endswith(".csv")
        assert timestamped_path != original_path
    
    def test_generate_submission_summary(self, sample_config, sample_predictions):
        """Testa geração de resumo da submissão."""
        formatter = SubmissionFormatter(sample_config)
        
        summary = formatter.generate_submission_summary(sample_predictions)
        
        # Verificar estrutura do resumo
        assert 'submission_info' in summary
        assert 'data_coverage' in summary
        assert 'quantity_summary' in summary
        assert 'weekly_breakdown' in summary
        
        # Verificar dados específicos
        assert summary['submission_info']['total_predictions'] == len(sample_predictions)
        assert summary['data_coverage']['weeks'] == [1, 2, 3]
        assert summary['quantity_summary']['total_quantity'] == sample_predictions['quantidade'].sum()
    
    def test_csv_encoding_and_separator(self, sample_predictions):
        """Testa configurações específicas de CSV."""
        config = {
            'output': {
                'csv_separator': ',',
                'csv_encoding': 'latin-1',
                'validate_format': True,
                'add_timestamp': False
            }
        }
        
        formatter = SubmissionFormatter(config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'test_custom.csv')
            
            result_path = formatter.format_submission_csv(sample_predictions, output_path)
            
            # Verificar se arquivo foi criado com configurações corretas
            with open(result_path, 'r', encoding='latin-1') as f:
                content = f.read()
                assert ',' in content  # Separador personalizado
                assert ';' not in content  # Não deve ter separador padrão


class TestSubmissionValidator:
    """Testes para a classe SubmissionValidator."""
    
    def test_validate_csv_format_success(self):
        """Testa validação bem-sucedida de CSV."""
        # Criar arquivo CSV temporário válido
        df = pd.DataFrame({
            'semana': [1, 2, 3],
            'pdv': ['001', '002', '003'],
            'produto': ['A', 'B', 'C'],
            'quantidade': [10, 15, 20]
        })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, 'valid.csv')
            df.to_csv(csv_path, sep=';', encoding='utf-8', index=False)
            
            result = SubmissionValidator.validate_csv_format(csv_path)
            
            assert result['is_valid'] is True
            assert len(result['errors']) == 0
            assert result['data_info']['total_rows'] == 3
    
    def test_validate_csv_format_file_not_found(self):
        """Testa validação com arquivo inexistente."""
        result = SubmissionValidator.validate_csv_format('/path/inexistente.csv')
        
        assert result['is_valid'] is False
        assert any('não encontrado' in error for error in result['errors'])
    
    def test_validate_csv_format_wrong_columns(self):
        """Testa validação com colunas incorretas."""
        df = pd.DataFrame({
            'week': [1, 2, 3],  # Nome incorreto
            'store': ['001', '002', '003'],  # Nome incorreto
            'product': ['A', 'B', 'C'],  # Nome incorreto
            'qty': [10, 15, 20]  # Nome incorreto
        })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, 'wrong_columns.csv')
            df.to_csv(csv_path, sep=';', encoding='utf-8', index=False)
            
            result = SubmissionValidator.validate_csv_format(csv_path)
            
            assert result['is_valid'] is False
            assert any('Colunas incorretas' in error for error in result['errors'])
    
    def test_validate_parquet_format_success(self):
        """Testa validação bem-sucedida de Parquet."""
        df = pd.DataFrame({
            'semana': [1, 2, 3],
            'pdv': ['001', '002', '003'],
            'produto': ['A', 'B', 'C'],
            'quantidade': [10, 15, 20]
        })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            parquet_path = os.path.join(temp_dir, 'valid.parquet')
            df.to_parquet(parquet_path, index=False)
            
            result = SubmissionValidator.validate_parquet_format(parquet_path)
            
            assert result['is_valid'] is True
            assert len(result['errors']) == 0
            assert result['data_info']['total_rows'] == 3
    
    def test_validate_parquet_format_file_not_found(self):
        """Testa validação Parquet com arquivo inexistente."""
        result = SubmissionValidator.validate_parquet_format('/path/inexistente.parquet')
        
        assert result['is_valid'] is False
        assert any('não encontrado' in error for error in result['errors'])
    
    def test_compare_formats_consistent(self):
        """Testa comparação entre formatos consistentes."""
        df = pd.DataFrame({
            'semana': [1, 2, 3],
            'pdv': ['001', '002', '003'],
            'produto': ['A', 'B', 'C'],
            'quantidade': [10, 15, 20]
        })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, 'test.csv')
            parquet_path = os.path.join(temp_dir, 'test.parquet')
            
            df.to_csv(csv_path, sep=';', encoding='utf-8', index=False)
            df.to_parquet(parquet_path, index=False)
            
            result = SubmissionValidator.compare_formats(csv_path, parquet_path)
            
            assert result['files_consistent'] is True
            assert len(result['differences']) == 0
    
    def test_compare_formats_inconsistent(self):
        """Testa comparação entre formatos inconsistentes."""
        df1 = pd.DataFrame({
            'semana': [1, 2, 3],
            'pdv': ['001', '002', '003'],
            'produto': ['A', 'B', 'C'],
            'quantidade': [10, 15, 20]
        })
        
        df2 = pd.DataFrame({
            'semana': [1, 2],  # Menos linhas
            'pdv': ['001', '002'],
            'produto': ['A', 'B'],
            'quantidade': [10, 15]
        })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, 'test1.csv')
            parquet_path = os.path.join(temp_dir, 'test2.parquet')
            
            df1.to_csv(csv_path, sep=';', encoding='utf-8', index=False)
            df2.to_parquet(parquet_path, index=False)
            
            result = SubmissionValidator.compare_formats(csv_path, parquet_path)
            
            assert result['files_consistent'] is False
            assert len(result['differences']) > 0


class TestOutputFormatterIntegration:
    """Testes de integração para formatação de saída."""
    
    def test_end_to_end_csv_workflow(self):
        """Testa fluxo completo de formatação CSV."""
        config = {
            'output': {
                'csv_separator': ';',
                'csv_encoding': 'utf-8',
                'validate_format': True,
                'add_timestamp': False
            }
        }
        
        # Dados de teste
        predictions = pd.DataFrame({
            'semana': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            'pdv': ['001', '002', '001', '002', '001', '002', '001', '002', '001', '002'],
            'produto': ['A', 'A', 'B', 'B', 'A', 'A', 'B', 'B', 'A', 'A'],
            'quantidade': [10, 12, 15, 18, 8, 9, 20, 22, 11, 13]
        })
        
        formatter = SubmissionFormatter(config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'submission.csv')
            
            # Formatar e salvar
            result_path = formatter.format_submission_csv(predictions, output_path)
            
            # Validar arquivo salvo
            validation = SubmissionValidator.validate_csv_format(result_path)
            
            assert validation['is_valid'] is True
            assert validation['data_info']['total_rows'] == 10
            assert validation['data_info']['weeks'] == [1, 2, 3, 4, 5]
    
    def test_end_to_end_parquet_workflow(self):
        """Testa fluxo completo de formatação Parquet."""
        config = {
            'output': {
                'parquet_compression': 'snappy',
                'validate_format': True,
                'add_timestamp': False
            }
        }
        
        # Dados de teste
        predictions = pd.DataFrame({
            'semana': [1, 2, 3, 4, 5],
            'pdv': ['001', '002', '003', '004', '005'],
            'produto': ['A', 'B', 'C', 'D', 'E'],
            'quantidade': [100, 200, 150, 300, 250]
        })
        
        formatter = SubmissionFormatter(config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'submission.parquet')
            
            # Formatar e salvar
            result_path = formatter.format_submission_parquet(predictions, output_path)
            
            # Validar arquivo salvo
            validation = SubmissionValidator.validate_parquet_format(result_path)
            
            assert validation['is_valid'] is True
            assert validation['data_info']['total_rows'] == 5
            assert validation['data_info']['weeks'] == [1, 2, 3, 4, 5]
    
    def test_both_formats_consistency(self):
        """Testa consistência entre formatos CSV e Parquet."""
        config = {
            'output': {
                'csv_separator': ';',
                'csv_encoding': 'utf-8',
                'parquet_compression': 'snappy',
                'validate_format': True,
                'add_timestamp': False
            }
        }
        
        # Dados de teste
        predictions = pd.DataFrame({
            'semana': [1, 2, 3],
            'pdv': ['001', '002', '003'],
            'produto': ['A', 'B', 'C'],
            'quantidade': [50, 75, 100]
        })
        
        formatter = SubmissionFormatter(config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, 'submission.csv')
            parquet_path = os.path.join(temp_dir, 'submission.parquet')
            
            # Formatar ambos os formatos
            formatter.format_submission_csv(predictions, csv_path)
            formatter.format_submission_parquet(predictions, parquet_path)
            
            # Comparar consistência
            comparison = SubmissionValidator.compare_formats(csv_path, parquet_path)
            
            assert comparison['files_consistent'] is True
            assert len(comparison['differences']) == 0


if __name__ == '__main__':
    pytest.main([__file__])