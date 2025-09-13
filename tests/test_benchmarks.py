"""
Testes de benchmark e performance para o pipeline de ML.
"""
import pytest
import time
import psutil
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.data.ingestion import DataIngestion
from src.data.preprocessing import DataPreprocessor
from src.features.engineering import FeatureEngineer
from src.models.training import XGBoostModel, LightGBMModel
from src.models.prediction import PredictionGenerator

class TestBenchmarks:
    """Testes de benchmark para componentes críticos."""
    
    @pytest.fixture
    def large_dataset(self):
        """Dataset grande para testes de performance."""
        np.random.seed(42)
        n_records = 100000
        
        dates = pd.date_range('2022-01-01', '2022-12-31', freq='D')
        data = []
        
        for i in range(n_records):
            date = np.random.choice(dates)
            pdv = np.random.randint(1, 101)  # 100 PDVs
            produto = np.random.randint(1, 1001)  # 1000 produtos
            quantidade = np.random.poisson(10)
            faturamento = quantidade * np.random.uniform(5, 50)
            
            data.append({
                'data': date,
                'pdv': pdv,
                'produto': produto,
                'quantidade': quantidade,
                'faturamento': faturamento
            })
        
        return pd.DataFrame(data)
    
    @pytest.mark.performance
    def test_data_ingestion_performance(self, large_dataset, tmp_path):
        """Testa performance de ingestão de dados."""
        # Salvar dataset como parquet
        parquet_file = tmp_path / "test_data.parquet"
        large_dataset.to_parquet(parquet_file)
        
        ingestion = DataIngestion()
        
        # Medir tempo de carregamento
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        df = ingestion.load_transactions(str(parquet_file))
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        duration = end_time - start_time
        memory_used = end_memory - start_memory
        
        # Assertions de performance
        assert duration < 5.0, f"Carregamento muito lento: {duration:.2f}s"
        assert memory_used < 500, f"Uso excessivo de memória: {memory_used:.2f}MB"
        assert len(df) == len(large_dataset), "Dados perdidos durante carregamento"
        
        print(f"📊 Ingestão - Tempo: {duration:.2f}s, Memória: {memory_used:.2f}MB")
    
    @pytest.mark.performance
    def test_preprocessing_performance(self, large_dataset):
        """Testa performance de pré-processamento."""
        preprocessor = DataPreprocessor()
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Executar pré-processamento
        cleaned_df = preprocessor.clean_transactions(large_dataset)
        weekly_df = preprocessor.aggregate_weekly_sales(cleaned_df)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        duration = end_time - start_time
        memory_used = end_memory - start_memory
        
        # Assertions de performance
        assert duration < 10.0, f"Pré-processamento muito lento: {duration:.2f}s"
        assert memory_used < 200, f"Uso excessivo de memória: {memory_used:.2f}MB"
        assert len(weekly_df) > 0, "Nenhum dado após agregação"
        
        print(f"📊 Pré-processamento - Tempo: {duration:.2f}s, Memória: {memory_used:.2f}MB")
    
    @pytest.mark.performance
    def test_feature_engineering_performance(self, large_dataset):
        """Testa performance de engenharia de features."""
        preprocessor = DataPreprocessor()
        engineer = FeatureEngineer()
        
        # Preparar dados
        cleaned_df = preprocessor.clean_transactions(large_dataset)
        weekly_df = preprocessor.aggregate_weekly_sales(cleaned_df)
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Executar feature engineering
        features_df = engineer.create_temporal_features(weekly_df)
        features_df = engineer.create_lag_features(features_df)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        duration = end_time - start_time
        memory_used = end_memory - start_memory
        
        # Assertions de performance
        assert duration < 15.0, f"Feature engineering muito lento: {duration:.2f}s"
        assert memory_used < 300, f"Uso excessivo de memória: {memory_used:.2f}MB"
        assert len(features_df.columns) > len(weekly_df.columns), "Features não criadas"
        
        print(f"📊 Feature Engineering - Tempo: {duration:.2f}s, Memória: {memory_used:.2f}MB")
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_model_training_performance(self, sample_transactions, sample_products, sample_stores):
        """Testa performance de treinamento de modelo."""
        # Preparar dados menores para treinamento rápido
        preprocessor = DataPreprocessor()
        engineer = FeatureEngineer()
        trainer = ModelTrainer()
        
        # Pipeline de preparação
        cleaned_df = preprocessor.clean_transactions(sample_transactions)
        weekly_df = preprocessor.aggregate_weekly_sales(cleaned_df)
        merged_df = preprocessor.merge_master_data(weekly_df, sample_products, sample_stores)
        features_df = engineer.create_all_features(merged_df)
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Treinamento
        feature_cols = [col for col in features_df.columns if col not in ['quantidade', 'pdv', 'produto', 'semana']]
        X = features_df[feature_cols].fillna(0)
        y = features_df['quantidade'].fillna(0)
        
        model = xgb_model.train(X, y)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        duration = end_time - start_time
        memory_used = end_memory - start_memory
        
        # Assertions de performance
        assert duration < 30.0, f"Treinamento muito lento: {duration:.2f}s"
        assert memory_used < 500, f"Uso excessivo de memória: {memory_used:.2f}MB"
        assert model is not None, "Modelo não foi treinado"
        
        print(f"📊 Treinamento - Tempo: {duration:.2f}s, Memória: {memory_used:.2f}MB")
    
    @pytest.mark.performance
    def test_prediction_performance(self, sample_transactions, sample_products, sample_stores):
        """Testa performance de geração de previsões."""
        # Preparar modelo treinado
        preprocessor = DataPreprocessor()
        engineer = FeatureEngineer()
        xgb_model = XGBoostModel()
        predictor = PredictionGenerator()
        
        # Pipeline completo
        cleaned_df = preprocessor.clean_transactions(sample_transactions)
        weekly_df = preprocessor.aggregate_weekly_sales(cleaned_df)
        merged_df = preprocessor.merge_master_data(weekly_df, sample_products, sample_stores)
        features_df = engineer.create_all_features(merged_df)
        
        feature_cols = [col for col in features_df.columns if col not in ['quantidade', 'pdv', 'produto', 'semana']]
        X = features_df[feature_cols].fillna(0)
        y = features_df['quantidade'].fillna(0)
        
        model = xgb_model.train(X, y)
        
        # Preparar dados para predição
        prediction_data = features_df.head(1000)  # Subset para teste
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Gerar previsões
        predictions = predictor.predict(model, prediction_data)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        duration = end_time - start_time
        memory_used = end_memory - start_memory
        
        # Assertions de performance
        assert duration < 5.0, f"Predição muito lenta: {duration:.2f}s"
        assert memory_used < 100, f"Uso excessivo de memória: {memory_used:.2f}MB"
        assert len(predictions) == len(prediction_data), "Predições incompletas"
        
        print(f"📊 Predição - Tempo: {duration:.2f}s, Memória: {memory_used:.2f}MB")
    
    @pytest.mark.performance
    def test_memory_leak_detection(self, sample_transactions):
        """Testa vazamentos de memória em operações repetidas."""
        preprocessor = DataPreprocessor()
        
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_readings = []
        
        # Executar operação múltiplas vezes
        for i in range(10):
            cleaned_df = preprocessor.clean_transactions(sample_transactions)
            weekly_df = preprocessor.aggregate_weekly_sales(cleaned_df)
            
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_readings.append(current_memory)
            
            # Forçar garbage collection
            import gc
            gc.collect()
        
        final_memory = memory_readings[-1]
        memory_growth = final_memory - initial_memory
        
        # Verificar crescimento excessivo de memória
        assert memory_growth < 100, f"Possível vazamento de memória: {memory_growth:.2f}MB"
        
        print(f"📊 Teste de vazamento - Crescimento: {memory_growth:.2f}MB")
    
    @pytest.mark.performance
    def test_concurrent_processing(self, sample_transactions):
        """Testa processamento concorrente."""
        import concurrent.futures
        from threading import Lock
        
        preprocessor = DataPreprocessor()
        results = []
        lock = Lock()
        
        def process_chunk(chunk_data):
            """Processa um chunk de dados."""
            cleaned = preprocessor.clean_transactions(chunk_data)
            weekly = preprocessor.aggregate_weekly_sales(cleaned)
            
            with lock:
                results.append(len(weekly))
            
            return len(weekly)
        
        # Dividir dados em chunks
        chunk_size = len(sample_transactions) // 4
        chunks = [
            sample_transactions[i:i+chunk_size] 
            for i in range(0, len(sample_transactions), chunk_size)
        ]
        
        start_time = time.time()
        
        # Processamento concorrente
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
            concurrent.futures.wait(futures)
        
        duration = time.time() - start_time
        
        # Verificar resultados
        assert len(results) == len(chunks), "Nem todos os chunks foram processados"
        assert all(r > 0 for r in results), "Chunks vazios após processamento"
        
        print(f"📊 Processamento concorrente - Tempo: {duration:.2f}s, Chunks: {len(chunks)}")

class TestQualityMetrics:
    """Testes de métricas de qualidade de código."""
    
    def test_code_complexity(self):
        """Verifica complexidade do código."""
        # Este teste seria implementado com ferramentas como radon
        # Por simplicidade, apenas verificamos estrutura básica
        src_path = Path(__file__).parent.parent / "src"
        
        python_files = list(src_path.rglob("*.py"))
        assert len(python_files) > 0, "Nenhum arquivo Python encontrado"
        
        # Verificar que arquivos não são muito grandes
        for py_file in python_files:
            if py_file.name != "__init__.py":
                lines = py_file.read_text().count('\n')
                assert lines < 1000, f"Arquivo muito grande: {py_file} ({lines} linhas)"
    
    def test_import_structure(self):
        """Verifica estrutura de imports."""
        src_path = Path(__file__).parent.parent / "src"
        
        for py_file in src_path.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue
                
            content = py_file.read_text()
            lines = content.split('\n')
            
            # Verificar que imports estão no topo
            import_section_ended = False
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                if line.startswith(('import ', 'from ')):
                    assert not import_section_ended, f"Import fora de ordem em {py_file}"
                elif line and not line.startswith(('"""', "'''")):
                    import_section_ended = True
    
    def test_docstring_coverage(self):
        """Verifica cobertura de docstrings."""
        src_path = Path(__file__).parent.parent / "src"
        
        for py_file in src_path.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue
                
            content = py_file.read_text()
            
            # Verificar docstring no módulo
            if 'class ' in content or 'def ' in content:
                assert '"""' in content or "'''" in content, f"Sem docstrings em {py_file}"