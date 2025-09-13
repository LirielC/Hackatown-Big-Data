"""
Tests for performance optimization modules.

This module contains unit tests for the optimized data ingestion,
feature engineering, and prediction components.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch

import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.ingestion_optimized import OptimizedDataIngestion
from features.engineering_optimized import OptimizedFeatureEngineer, FeatureCache
from models.prediction_optimized import OptimizedPredictionGenerator, BatchPredictionOptimizer
from utils.performance_utils import PerformanceMonitor, MemoryManager, OptimizationCoordinator


class TestOptimizedDataIngestion:
    """Test optimized data ingestion functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample transaction data."""
        return pd.DataFrame({
            'data': pd.date_range('2022-01-01', periods=100, freq='D'),
            'pdv': np.random.randint(1, 10, 100),
            'produto': np.random.randint(1, 50, 100),
            'quantidade': np.random.randint(1, 100, 100)
        })
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_optimized_ingestion_initialization(self):
        """Test OptimizedDataIngestion initialization."""
        ingestion = OptimizedDataIngestion(
            use_lazy_loading=True,
            chunk_size=1000,
            max_workers=2
        )
        
        assert ingestion.use_lazy_loading is True
        assert ingestion.chunk_size == 1000
        assert ingestion.max_workers == 2
    
    def test_load_single_file(self, sample_data, temp_dir):
        """Test loading single Parquet file."""
        # Save sample data
        file_path = temp_dir / "test_data.parquet"
        sample_data.to_parquet(file_path, index=False)
        
        # Load with optimized ingestion
        ingestion = OptimizedDataIngestion()
        loaded_data = ingestion.load_transactions_optimized(file_path)
        
        assert len(loaded_data) == len(sample_data)
        assert list(loaded_data.columns) == list(sample_data.columns)
    
    def test_load_directory(self, sample_data, temp_dir):
        """Test loading multiple Parquet files from directory."""
        # Save multiple files
        for i in range(3):
            file_path = temp_dir / f"test_data_{i}.parquet"
            sample_data.to_parquet(file_path, index=False)
        
        # Load with optimized ingestion
        ingestion = OptimizedDataIngestion()
        loaded_data = ingestion.load_transactions_optimized(temp_dir)
        
        assert len(loaded_data) == len(sample_data) * 3
    
    def test_file_info_extraction(self, sample_data, temp_dir):
        """Test file information extraction."""
        file_path = temp_dir / "test_data.parquet"
        sample_data.to_parquet(file_path, index=False)
        
        ingestion = OptimizedDataIngestion()
        file_info = ingestion.get_file_info(file_path)
        
        assert 'file_size_mb' in file_info
        assert 'num_rows' in file_info
        assert 'num_columns' in file_info
        assert file_info['num_rows'] == len(sample_data)


class TestFeatureCache:
    """Test feature caching functionality."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for caching tests."""
        return pd.DataFrame({
            'pdv': [1, 2, 3],
            'produto': [10, 20, 30],
            'quantidade': [5, 10, 15]
        })
    
    def test_cache_initialization(self, temp_cache_dir):
        """Test FeatureCache initialization."""
        cache = FeatureCache(str(temp_cache_dir))
        assert cache.cache_dir == temp_cache_dir
        assert temp_cache_dir.exists()
    
    def test_cache_save_and_load(self, temp_cache_dir, sample_data):
        """Test saving and loading cached features."""
        cache = FeatureCache(str(temp_cache_dir))
        
        # Create features
        features_df = sample_data.copy()
        features_df['new_feature'] = features_df['quantidade'] * 2
        
        # Save to cache
        params = {'test_param': 'value'}
        cache.save_features(sample_data, features_df, 'test_features', params)
        
        # Load from cache
        loaded_features = cache.get_cached_features(sample_data, 'test_features', params)
        
        assert loaded_features is not None
        pd.testing.assert_frame_equal(loaded_features, features_df)
    
    def test_cache_miss(self, temp_cache_dir, sample_data):
        """Test cache miss scenario."""
        cache = FeatureCache(str(temp_cache_dir))
        
        # Try to load non-existent cache
        params = {'nonexistent': 'param'}
        loaded_features = cache.get_cached_features(sample_data, 'nonexistent', params)
        
        assert loaded_features is None


class TestOptimizedFeatureEngineer:
    """Test optimized feature engineering functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data with temporal information."""
        return pd.DataFrame({
            'data_semana': pd.date_range('2022-01-01', periods=52, freq='W'),
            'pdv': np.tile([1, 2, 3], 18)[:52],
            'produto': np.tile([10, 20], 26),
            'quantidade': np.random.randint(1, 100, 52),
            'semana': range(1, 53)
        })
    
    def test_optimized_engineer_initialization(self):
        """Test OptimizedFeatureEngineer initialization."""
        engineer = OptimizedFeatureEngineer(
            use_cache=True,
            use_polars=True,
            max_workers=2
        )
        
        assert engineer.use_cache is True
        assert engineer.use_polars is True
        assert engineer.max_workers == 2
    
    def test_temporal_features_creation(self, sample_data):
        """Test optimized temporal feature creation."""
        engineer = OptimizedFeatureEngineer(use_cache=False)
        
        result_df = engineer.create_temporal_features_optimized(sample_data)
        
        # Check that temporal features were added
        temporal_features = ['semana_ano', 'mes', 'trimestre', 'semana_sin', 'semana_cos']
        for feature in temporal_features:
            assert feature in result_df.columns
        
        assert len(result_df) == len(sample_data)
    
    def test_lag_features_creation(self, sample_data):
        """Test optimized lag feature creation."""
        engineer = OptimizedFeatureEngineer(use_cache=False)
        
        result_df = engineer.create_lag_features_optimized(
            sample_data,
            target_column='quantidade',
            lag_periods=[1, 2]
        )
        
        # Check that lag features were added
        assert 'lag_1_quantidade' in result_df.columns
        assert 'lag_2_quantidade' in result_df.columns
        assert len(result_df) == len(sample_data)
    
    def test_parallel_feature_creation(self, sample_data):
        """Test parallel feature creation."""
        engineer = OptimizedFeatureEngineer(use_cache=False, max_workers=2)
        
        result_df = engineer.create_all_features_parallel(
            sample_data,
            feature_types=['temporal', 'lag']
        )
        
        # Check that features from both types were created
        assert 'semana_ano' in result_df.columns  # temporal
        assert 'lag_1_quantidade' in result_df.columns  # lag
        assert len(result_df) == len(sample_data)


class TestBatchPredictionOptimizer:
    """Test batch prediction optimization functionality."""
    
    @pytest.fixture
    def sample_features(self):
        """Create sample features for prediction."""
        return pd.DataFrame({
            'pdv': np.random.randint(1, 10, 1000),
            'produto': np.random.randint(1, 50, 1000),
            'semana': np.random.randint(1, 6, 1000),
            'feature_1': np.random.randn(1000),
            'feature_2': np.random.randn(1000)
        })
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model for testing."""
        model = Mock()
        model.predict.return_value = np.random.randint(0, 100, 100)
        return model
    
    def test_batch_optimizer_initialization(self):
        """Test BatchPredictionOptimizer initialization."""
        optimizer = BatchPredictionOptimizer(
            batch_size=500,
            max_workers=2,
            memory_limit_gb=2.0
        )
        
        assert optimizer.batch_size == 500
        assert optimizer.max_workers == 2
        assert optimizer.memory_limit_gb == 2.0
    
    def test_batch_creation(self, sample_features):
        """Test batch creation from DataFrame."""
        optimizer = BatchPredictionOptimizer(batch_size=300)
        
        batches = list(optimizer.create_batches(sample_features))
        
        assert len(batches) == 4  # 1000 / 300 = 3.33, so 4 batches
        assert len(batches[0]) == 300
        assert len(batches[-1]) <= 300  # Last batch may be smaller
    
    def test_single_batch_prediction(self, sample_features, mock_model):
        """Test prediction on single batch."""
        optimizer = BatchPredictionOptimizer()
        
        batch = sample_features.head(100)
        feature_columns = ['feature_1', 'feature_2']
        
        predictions = optimizer.predict_batch(mock_model, batch, feature_columns)
        
        assert len(predictions) == 100
        mock_model.predict.assert_called_once()


class TestOptimizedPredictionGenerator:
    """Test optimized prediction generation functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return {
            'prediction': {
                'target_weeks': [1, 2, 3, 4, 5],
                'post_processing': {
                    'ensure_positive': True,
                    'apply_bounds': True
                }
            },
            'optimization': {
                'batch_size': 1000,
                'use_parallel': True,
                'memory_limit_gb': 2.0
            }
        }
    
    @pytest.fixture
    def sample_features(self):
        """Create sample features for prediction."""
        return pd.DataFrame({
            'pdv': np.random.randint(1, 10, 500),
            'produto': np.random.randint(1, 50, 500),
            'semana': np.random.randint(1, 6, 500),
            'feature_1': np.random.randn(500),
            'feature_2': np.random.randn(500)
        })
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model for testing."""
        model = Mock()
        model.is_fitted = True
        model.predict.return_value = np.random.randint(0, 100, 500)
        return model
    
    def test_optimized_generator_initialization(self, config):
        """Test OptimizedPredictionGenerator initialization."""
        generator = OptimizedPredictionGenerator(config)
        
        assert generator.batch_size == 1000
        assert generator.use_parallel is True
        assert generator.target_weeks == [1, 2, 3, 4, 5]
    
    def test_feature_preparation(self, config, sample_features):
        """Test feature preparation for prediction."""
        generator = OptimizedPredictionGenerator(config)
        
        prepared_features, feature_columns = generator._prepare_features_optimized(sample_features)
        
        assert 'feature_1' in feature_columns
        assert 'feature_2' in feature_columns
        assert 'pdv' not in feature_columns  # Should be excluded
        assert len(prepared_features) == len(sample_features)
    
    def test_prediction_generation(self, config, sample_features, mock_model):
        """Test optimized prediction generation."""
        generator = OptimizedPredictionGenerator(config)
        
        predictions_df = generator.generate_predictions_optimized(
            mock_model, sample_features
        )
        
        # Check output format
        required_columns = ['semana', 'pdv', 'produto', 'quantidade']
        for col in required_columns:
            assert col in predictions_df.columns
        
        # Check that only target weeks are included
        assert set(predictions_df['semana'].unique()).issubset(set([1, 2, 3, 4, 5]))


class TestPerformanceUtils:
    """Test performance utility functions."""
    
    def test_performance_monitor(self):
        """Test PerformanceMonitor functionality."""
        monitor = PerformanceMonitor()
        
        # Test timing
        monitor.start_timer('test_operation')
        import time
        time.sleep(0.1)
        duration = monitor.end_timer('test_operation')
        
        assert duration >= 0.1
        assert 'test_operation' in monitor.metrics
    
    def test_memory_manager(self):
        """Test MemoryManager functionality."""
        manager = MemoryManager(memory_limit_gb=1.0)
        
        # Test memory check
        within_limits = manager.check_memory_usage()
        assert isinstance(within_limits, bool)
        
        # Test garbage collection
        gc_stats = manager.force_garbage_collection()
        assert 'objects_collected' in gc_stats
    
    def test_optimization_coordinator(self):
        """Test OptimizationCoordinator functionality."""
        config = {
            'optimization': {
                'use_polars': True,
                'use_caching': True,
                'batch_size': 1000
            }
        }
        
        coordinator = OptimizationCoordinator(config)
        
        assert coordinator.use_polars is True
        assert coordinator.use_caching is True
        assert coordinator.batch_size == 1000
        
        # Test component creation
        data_loader = coordinator.get_optimized_data_loader()
        assert isinstance(data_loader, OptimizedDataIngestion)
        
        feature_engineer = coordinator.get_optimized_feature_engineer()
        assert isinstance(feature_engineer, OptimizedFeatureEngineer)


if __name__ == "__main__":
    pytest.main([__file__])