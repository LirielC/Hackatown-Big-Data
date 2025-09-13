"""
Unit tests for data ingestion module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
from unittest.mock import patch, MagicMock

from src.data.ingestion import DataIngestion, DataIngestionError


class TestDataIngestion:
    """Test cases for DataIngestion class."""
    
    @pytest.fixture
    def data_ingestion(self):
        """Create DataIngestion instance for testing."""
        return DataIngestion()
    
    @pytest.fixture
    def sample_transaction_data(self):
        """Create sample transaction data for testing."""
        return pd.DataFrame({
            'data': pd.date_range('2022-01-01', periods=100, freq='D'),
            'pdv': np.random.randint(1, 10, 100),
            'produto': np.random.randint(1, 50, 100),
            'quantidade': np.random.randint(1, 100, 100),
            'faturamento': np.random.uniform(10, 1000, 100)
        })
    
    @pytest.fixture
    def sample_product_data(self):
        """Create sample product data for testing."""
        return pd.DataFrame({
            'produto': range(1, 51),
            'categoria': ['categoria_' + str(i % 5) for i in range(50)],
            'descricao': ['produto_' + str(i) for i in range(1, 51)]
        })
    
    @pytest.fixture
    def sample_store_data(self):
        """Create sample store data for testing."""
        return pd.DataFrame({
            'pdv': range(1, 11),
            'tipo': ['c-store'] * 5 + ['g-store'] * 5,
            'zipcode': [f'0000{i}' for i in range(1, 11)]
        })
    
    @pytest.fixture
    def temp_parquet_file(self, sample_transaction_data):
        """Create temporary Parquet file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
            sample_transaction_data.to_parquet(tmp.name)
            yield tmp.name
        os.unlink(tmp.name)
    
    def test_init_default(self):
        """Test DataIngestion initialization with default parameters."""
        ingestion = DataIngestion()
        assert ingestion.use_polars is False
        assert ingestion.logger is not None
    
    def test_init_with_polars(self):
        """Test DataIngestion initialization with Polars enabled."""
        ingestion = DataIngestion(use_polars=True)
        assert ingestion.use_polars is True
    
    def test_load_transactions_success(self, data_ingestion, temp_parquet_file):
        """Test successful transaction data loading."""
        df = data_ingestion.load_transactions(temp_parquet_file)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'data' in df.columns
        assert 'pdv' in df.columns
        assert 'produto' in df.columns
        assert 'quantidade' in df.columns
    
    def test_load_transactions_file_not_found(self, data_ingestion):
        """Test transaction loading with non-existent file."""
        with pytest.raises(DataIngestionError):
            data_ingestion.load_transactions('non_existent_file.parquet')
    
    @patch('src.data.ingestion.pd.read_parquet')
    def test_load_transactions_validation_failure(self, mock_read_parquet, data_ingestion):
        """Test transaction loading with validation failure."""
        # Create invalid data (missing required columns)
        invalid_data = pd.DataFrame({'invalid_column': [1, 2, 3]})
        mock_read_parquet.return_value = invalid_data
        
        with pytest.raises(DataIngestionError):
            data_ingestion.load_transactions('dummy_path.parquet')
    
    def test_validate_transaction_data_valid(self, data_ingestion, sample_transaction_data):
        """Test validation of valid transaction data."""
        result = data_ingestion.validate_transaction_data(sample_transaction_data)
        assert result['is_valid'] is True
        assert len(result['errors']) == 0
        assert result['row_count'] == len(sample_transaction_data)
    
    def test_validate_transaction_data_missing_columns(self, data_ingestion):
        """Test validation with missing required columns."""
        invalid_data = pd.DataFrame({'invalid_column': [1, 2, 3]})
        result = data_ingestion.validate_transaction_data(invalid_data)
        assert result['is_valid'] is False
        assert len(result['errors']) > 0
        assert 'Missing required columns' in result['errors'][0]
    
    def test_validate_transaction_data_negative_quantities(self, data_ingestion):
        """Test validation with negative quantities."""
        data_with_negatives = pd.DataFrame({
            'data': pd.date_range('2022-01-01', periods=5),
            'pdv': [1, 2, 3, 4, 5],
            'produto': [1, 2, 3, 4, 5],
            'quantidade': [10, -5, 20, -3, 15]
        })
        result = data_ingestion.validate_transaction_data(data_with_negatives)
        assert result['is_valid'] is True  # Warnings don't make it invalid
        assert len(result['warnings']) > 0
        assert any('negative quantities' in warning for warning in result['warnings'])
    
    def test_validate_transaction_data_null_values(self, data_ingestion):
        """Test validation with null values."""
        data_with_nulls = pd.DataFrame({
            'data': pd.date_range('2022-01-01', periods=5),
            'pdv': [1, 2, None, 4, 5],
            'produto': [1, 2, 3, 4, 5],
            'quantidade': [10, 5, 20, 3, 15]
        })
        result = data_ingestion.validate_transaction_data(data_with_nulls)
        assert result['is_valid'] is True  # Warnings don't make it invalid
        assert len(result['warnings']) > 0
        assert any('null values' in warning for warning in result['warnings'])
    
    def test_validate_product_data_valid(self, data_ingestion, sample_product_data):
        """Test validation of valid product data."""
        result = data_ingestion.validate_product_data(sample_product_data)
        assert result['is_valid'] is True
        assert len(result['errors']) == 0
    
    def test_validate_product_data_duplicate_ids(self, data_ingestion):
        """Test validation with duplicate product IDs."""
        duplicate_data = pd.DataFrame({
            'produto': [1, 2, 2, 3, 4],
            'categoria': ['A', 'B', 'B', 'C', 'D']
        })
        result = data_ingestion.validate_product_data(duplicate_data)
        assert result['is_valid'] is False
        assert len(result['errors']) > 0
        assert 'duplicate product IDs' in result['errors'][0]
    
    def test_validate_store_data_valid(self, data_ingestion, sample_store_data):
        """Test validation of valid store data."""
        result = data_ingestion.validate_store_data(sample_store_data)
        assert result['is_valid'] is True
        assert len(result['errors']) == 0
    
    def test_validate_store_data_duplicate_ids(self, data_ingestion):
        """Test validation with duplicate store IDs."""
        duplicate_data = pd.DataFrame({
            'pdv': [1, 2, 2, 3, 4],
            'tipo': ['c-store', 'g-store', 'g-store', 'liquor', 'c-store']
        })
        result = data_ingestion.validate_store_data(duplicate_data)
        assert result['is_valid'] is False
        assert len(result['errors']) > 0
        assert 'duplicate store IDs' in result['errors'][0]
    
    def test_validate_data_quality_generic(self, data_ingestion, sample_transaction_data):
        """Test generic data quality validation."""
        result = data_ingestion.validate_data_quality(sample_transaction_data, "generic")
        assert result['is_valid'] is True
        assert 'row_count' in result
        assert 'column_count' in result
    
    def test_validate_data_quality_transactions(self, data_ingestion, sample_transaction_data):
        """Test data quality validation for transactions."""
        result = data_ingestion.validate_data_quality(sample_transaction_data, "transactions")
        assert result['is_valid'] is True
    
    def test_get_data_summary(self, data_ingestion, sample_transaction_data):
        """Test data summary generation."""
        summary = data_ingestion.get_data_summary(sample_transaction_data)
        assert 'shape' in summary
        assert 'columns' in summary
        assert 'data_types' in summary
        assert 'null_counts' in summary
        assert 'memory_usage_mb' in summary
        assert 'numeric_stats' in summary
        assert summary['shape'] == sample_transaction_data.shape
    
    def test_get_data_summary_with_categorical(self, data_ingestion, sample_product_data):
        """Test data summary with categorical columns."""
        summary = data_ingestion.get_data_summary(sample_product_data)
        assert 'categorical_stats' in summary
        assert 'categoria' in summary['categorical_stats']
        assert 'unique_count' in summary['categorical_stats']['categoria']
    
    @patch('src.data.ingestion.Path.exists')
    def test_load_parquet_data_file_not_exists(self, mock_exists, data_ingestion):
        """Test loading Parquet data when file doesn't exist."""
        mock_exists.return_value = False
        with pytest.raises(FileNotFoundError):
            data_ingestion._load_parquet_data('non_existent.parquet')
    
    def test_load_parquet_data_directory_no_files(self, data_ingestion):
        """Test loading from directory with no Parquet files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(FileNotFoundError):
                data_ingestion._load_parquet_data(temp_dir)
    
    def test_load_parquet_data_multiple_files(self, data_ingestion, sample_transaction_data):
        """Test loading multiple Parquet files from directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create multiple Parquet files
            for i in range(3):
                file_path = Path(temp_dir) / f'data_{i}.parquet'
                sample_data = sample_transaction_data.iloc[i*10:(i+1)*10].copy()
                sample_data.to_parquet(file_path)
            
            df = data_ingestion._load_parquet_data(temp_dir)
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 30  # 3 files * 10 rows each


class TestDataIngestionIntegration:
    """Integration tests for DataIngestion class."""
    
    def test_full_workflow_with_real_data_structure(self):
        """Test complete workflow with realistic data structure."""
        # Create realistic test data
        transaction_data = pd.DataFrame({
            'data': pd.date_range('2022-01-01', periods=1000, freq='D'),
            'pdv': np.random.randint(1, 100, 1000),
            'produto': np.random.randint(1, 500, 1000),
            'quantidade': np.random.randint(1, 50, 1000),
            'faturamento': np.random.uniform(10, 500, 1000)
        })
        
        product_data = pd.DataFrame({
            'produto': range(1, 501),
            'categoria': [f'categoria_{i % 10}' for i in range(500)],
            'descricao': [f'produto_{i}' for i in range(1, 501)]
        })
        
        store_data = pd.DataFrame({
            'pdv': range(1, 101),
            'tipo': ['c-store'] * 50 + ['g-store'] * 30 + ['liquor'] * 20,
            'zipcode': [f'{i:05d}' for i in range(1, 101)]
        })
        
        ingestion = DataIngestion()
        
        # Test validation of all data types
        trans_validation = ingestion.validate_transaction_data(transaction_data)
        prod_validation = ingestion.validate_product_data(product_data)
        store_validation = ingestion.validate_store_data(store_data)
        
        assert trans_validation['is_valid']
        assert prod_validation['is_valid']
        assert store_validation['is_valid']
        
        # Test data summaries
        trans_summary = ingestion.get_data_summary(transaction_data)
        prod_summary = ingestion.get_data_summary(product_data)
        store_summary = ingestion.get_data_summary(store_data)
        
        assert trans_summary['shape'][0] == 1000
        assert prod_summary['shape'][0] == 500
        assert store_summary['shape'][0] == 100