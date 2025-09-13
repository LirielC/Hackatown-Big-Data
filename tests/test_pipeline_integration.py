"""
Testes de integração para o pipeline completo do modelo de previsão.
"""

import pytest
import tempfile
import shutil
import yaml
import json
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
import sys
import os

# Adicionar src ao path para imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from main import (
    load_config, validate_config, setup_environment, 
    run_data_ingestion, run_data_preprocessing,
    PipelineExecutionTracker, run_full_pipeline
)


class TestPipelineIntegration:
    """Testes de integração para o pipeline completo."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Cria workspace temporário para testes."""
        temp_dir = tempfile.mkdtemp()
        
        # Criar estrutura de diretórios
        dirs = [
            'data/raw',
            'data/processed', 
            'data/features',
            'configs',
            'models',
            'output'
        ]
        
        for dir_path in dirs:
            Path(temp_dir, dir_path).mkdir(parents=True, exist_ok=True)
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def sample_config(self, temp_workspace):
        """Cria configuração de teste."""
        config = {
            'general': {
                'project_name': 'test-hackathon-forecast',
                'random_seed': 42,
                'n_jobs': 1,
                'verbose': True
            },
            'data': {
                'raw_data_path': str(Path(temp_workspace) / 'data' / 'raw'),
                'processed_data_path': str(Path(temp_workspace) / 'data' / 'processed'),
                'features_path': str(Path(temp_workspace) / 'data' / 'features'),
                'aggregation': {
                    'frequency': 'W',
                    'start_date': '2022-01-01',
                    'end_date': '2022-12-31'
                }
            },
            'models': {
                'xgboost': {
                    'n_estimators': 10,
                    'max_depth': 3,
                    'random_state': 42
                }
            },
            'experiment_tracking': {
                'enabled': False,
                'experiment_name': 'test-experiment'
            },
            'performance': {
                'use_polars': False
            }
        }
        
        # Salvar configuração
        config_path = Path(temp_workspace) / 'configs' / 'test_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        return str(config_path), config
    
    @pytest.fixture
    def sample_data(self, temp_workspace):
        """Cria dados de teste."""
        # Dados de transação
        np.random.seed(42)
        n_records = 1000
        
        # Gerar datas de 2022
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2022, 12, 31)
        date_range = pd.date_range(start_date, end_date, freq='D')
        
        transactions_data = {
            'transaction_date': np.random.choice(date_range, n_records),
            'internal_store_id': np.random.randint(1, 11, n_records),
            'internal_product_id': np.random.randint(1, 51, n_records),
            'quantity': np.random.randint(1, 20, n_records),
            'revenue': np.random.uniform(10, 500, n_records)
        }
        
        transactions_df = pd.DataFrame(transactions_data)
        
        # Dados de PDV
        stores_data = {
            'pdv': range(1, 11),
            'categoria_pdv': ['c-store'] * 5 + ['g-store'] * 3 + ['liquor'] * 2,
            'zipcode': [f'0000{i}' for i in range(1, 11)]
        }
        stores_df = pd.DataFrame(stores_data)
        
        # Dados de produto
        products_data = {
            'produto': range(1, 51),
            'categoria': ['categoria_' + str(i % 5) for i in range(1, 51)],
            'marca': ['marca_' + str(i % 10) for i in range(1, 51)]
        }
        products_df = pd.DataFrame(products_data)
        
        # Salvar dados como Parquet
        raw_path = Path(temp_workspace) / 'data' / 'raw'
        
        transactions_df.to_parquet(raw_path / 'transactions.parquet', index=False)
        stores_df.to_parquet(raw_path / 'stores.parquet', index=False)
        products_df.to_parquet(raw_path / 'products.parquet', index=False)
        
        return {
            'transactions': transactions_df,
            'stores': stores_df,
            'products': products_df
        }
    
    def test_load_config(self, sample_config):
        """Testa carregamento de configuração."""
        config_path, expected_config = sample_config
        
        loaded_config = load_config(config_path)
        
        assert loaded_config['general']['project_name'] == expected_config['general']['project_name']
        assert loaded_config['general']['random_seed'] == expected_config['general']['random_seed']
        assert loaded_config['data']['raw_data_path'] == expected_config['data']['raw_data_path']
    
    def test_validate_config_valid(self, sample_config):
        """Testa validação de configuração válida."""
        _, config = sample_config
        
        # Não deve levantar exceção
        validate_config(config)
    
    def test_validate_config_missing_section(self, sample_config):
        """Testa validação com seção obrigatória faltando."""
        _, config = sample_config
        
        # Remover seção obrigatória
        del config['general']
        
        with pytest.raises(ValueError, match="Seção obrigatória 'general'"):
            validate_config(config)
    
    def test_validate_config_missing_random_seed(self, sample_config):
        """Testa validação com random_seed faltando."""
        _, config = sample_config
        
        # Remover random_seed
        del config['general']['random_seed']
        
        with pytest.raises(ValueError, match="random_seed não configurado"):
            validate_config(config)
    
    def test_setup_environment_reproducibility(self, sample_config):
        """Testa configuração de ambiente para reprodutibilidade."""
        _, config = sample_config
        
        # Configurar ambiente
        tracker = setup_environment(config)
        
        # Verificar se seeds foram configurados
        import random
        import numpy as np
        
        # Gerar números aleatórios
        random_nums_1 = [random.random() for _ in range(5)]
        numpy_nums_1 = np.random.random(5)
        
        # Reconfigurar ambiente
        setup_environment(config)
        
        # Gerar números novamente
        random_nums_2 = [random.random() for _ in range(5)]
        numpy_nums_2 = np.random.random(5)
        
        # Devem ser iguais (reprodutibilidade)
        assert random_nums_1 == random_nums_2
        assert np.array_equal(numpy_nums_1, numpy_nums_2)
    
    def test_pipeline_execution_tracker(self):
        """Testa rastreador de execução do pipeline."""
        tracker = PipelineExecutionTracker()
        
        # Iniciar pipeline
        tracker.start_pipeline()
        assert tracker.start_time is not None
        
        # Executar etapas
        tracker.start_step("Teste 1")
        tracker.end_step(success=True)
        
        tracker.start_step("Teste 2")
        tracker.end_step(success=False, error="Erro de teste")
        
        # Finalizar pipeline
        tracker.end_pipeline()
        assert tracker.end_time is not None
        
        # Verificar resumo
        summary = tracker.get_summary()
        assert summary['total_steps'] == 2
        assert summary['successful_steps'] == 1
        assert summary['success_rate'] == 0.5
        assert len(summary['steps']) == 2
        
        # Verificar detalhes das etapas
        assert summary['steps'][0]['name'] == "Teste 1"
        assert summary['steps'][0]['success'] is True
        assert summary['steps'][1]['name'] == "Teste 2"
        assert summary['steps'][1]['success'] is False
        assert summary['steps'][1]['error'] == "Erro de teste"
    
    @patch('main.setup_mlflow_autolog')
    @patch('main.ExperimentTracker')
    def test_data_ingestion_integration(self, mock_tracker, mock_mlflow, sample_config, sample_data):
        """Testa integração da ingestão de dados."""
        config_path, config = sample_config
        
        # Mock experiment tracker
        mock_tracker.return_value = None
        
        # Executar ingestão
        run_data_ingestion(config)
        
        # Verificar se não houve erros (função deve completar sem exceções)
        assert True
    
    @patch('main.setup_mlflow_autolog')
    @patch('main.ExperimentTracker')
    def test_data_preprocessing_integration(self, mock_tracker, mock_mlflow, sample_config, sample_data):
        """Testa integração do pré-processamento."""
        config_path, config = sample_config
        
        # Mock experiment tracker
        mock_tracker.return_value = None
        
        # Executar pré-processamento
        run_data_preprocessing(config)
        
        # Verificar se arquivo processado foi criado
        processed_path = Path(config['data']['processed_data_path'])
        output_file = processed_path / "weekly_sales_processed.parquet"
        summary_file = processed_path / "preprocessing_summary.json"
        
        assert output_file.exists()
        assert summary_file.exists()
        
        # Verificar conteúdo do resumo
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        assert 'preprocessing_summary' in summary
        assert 'validation_results' in summary
        assert 'data_shape' in summary
    
    @patch('main.setup_mlflow_autolog')
    @patch('main.ExperimentTracker')
    def test_pipeline_step_execution_order(self, mock_tracker, mock_mlflow, sample_config, sample_data):
        """Testa ordem de execução das etapas do pipeline."""
        config_path, config = sample_config
        
        # Mock experiment tracker
        mock_tracker_instance = MagicMock()
        mock_tracker.return_value = mock_tracker_instance
        
        executed_steps = []
        
        # Mock das funções de etapa para rastrear ordem
        def mock_step(step_name):
            def wrapper(config):
                executed_steps.append(step_name)
            return wrapper
        
        with patch('main.run_data_ingestion', mock_step('ingestion')), \
             patch('main.run_data_preprocessing', mock_step('preprocessing')), \
             patch('main.run_feature_engineering', mock_step('features')), \
             patch('main.run_model_training', mock_step('training')), \
             patch('main.run_model_validation', mock_step('validation')), \
             patch('main.run_prediction_generation', mock_step('prediction')), \
             patch('main.run_output_formatting', mock_step('output')):
            
            run_full_pipeline(config)
        
        # Verificar ordem correta
        expected_order = [
            'ingestion', 'preprocessing', 'features', 
            'training', 'validation', 'prediction', 'output'
        ]
        assert executed_steps == expected_order
    
    @patch('main.setup_mlflow_autolog')
    @patch('main.ExperimentTracker')
    def test_pipeline_error_handling(self, mock_tracker, mock_mlflow, sample_config, sample_data):
        """Testa tratamento de erros no pipeline."""
        config_path, config = sample_config
        
        # Mock experiment tracker
        mock_tracker_instance = MagicMock()
        mock_tracker.return_value = mock_tracker_instance
        
        # Mock função que falha
        def failing_step(config):
            raise ValueError("Erro simulado")
        
        with patch('main.run_data_ingestion', failing_step):
            with pytest.raises(ValueError, match="Erro simulado"):
                run_full_pipeline(config)
        
        # Verificar se métricas de erro foram logadas
        mock_tracker_instance.log_metrics.assert_called()
        mock_tracker_instance.log_params.assert_called()
    
    def test_pipeline_summary_generation(self, temp_workspace, sample_config):
        """Testa geração de resumo de execução."""
        config_path, config = sample_config
        
        # Executar com mocks para evitar dependências
        with patch('main.setup_mlflow_autolog'), \
             patch('main.ExperimentTracker', return_value=None), \
             patch('main.run_data_ingestion'), \
             patch('main.run_data_preprocessing'), \
             patch('main.run_feature_engineering'), \
             patch('main.run_model_training'), \
             patch('main.run_model_validation'), \
             patch('main.run_prediction_generation'), \
             patch('main.run_output_formatting'):
            
            # Mudar para diretório temporário
            original_cwd = os.getcwd()
            os.chdir(temp_workspace)
            
            try:
                run_full_pipeline(config)
                
                # Verificar se arquivo de resumo foi criado
                summary_files = list(Path('.').glob('pipeline_execution_summary_*.json'))
                assert len(summary_files) > 0
                
                # Verificar conteúdo do resumo
                with open(summary_files[0], 'r') as f:
                    summary = json.load(f)
                
                assert 'start_time' in summary
                assert 'end_time' in summary
                assert 'total_duration_seconds' in summary
                assert 'total_steps' in summary
                assert 'successful_steps' in summary
                assert 'steps' in summary
                
            finally:
                os.chdir(original_cwd)


class TestPipelineReproducibility:
    """Testes específicos para reprodutibilidade do pipeline."""
    
    def test_multiple_runs_same_results(self):
        """Testa se múltiplas execuções produzem resultados idênticos."""
        config = {
            'general': {'random_seed': 42, 'n_jobs': 1},
            'experiment_tracking': {'enabled': False}
        }
        
        results = []
        
        for _ in range(3):
            # Configurar ambiente
            setup_environment(config)
            
            # Gerar dados aleatórios
            import numpy as np
            data = np.random.random(100)
            results.append(data.copy())
        
        # Todos os resultados devem ser idênticos
        for i in range(1, len(results)):
            assert np.array_equal(results[0], results[i])
    
    def test_seed_isolation(self):
        """Testa isolamento entre diferentes seeds."""
        config1 = {
            'general': {'random_seed': 42, 'n_jobs': 1},
            'experiment_tracking': {'enabled': False}
        }
        config2 = {
            'general': {'random_seed': 123, 'n_jobs': 1},
            'experiment_tracking': {'enabled': False}
        }
        
        # Executar com seed 42
        setup_environment(config1)
        import numpy as np
        result1 = np.random.random(100)
        
        # Executar com seed 123
        setup_environment(config2)
        result2 = np.random.random(100)
        
        # Resultados devem ser diferentes
        assert not np.array_equal(result1, result2)
        
        # Executar novamente com seed 42
        setup_environment(config1)
        result3 = np.random.random(100)
        
        # Deve ser igual ao primeiro resultado
        assert np.array_equal(result1, result3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])