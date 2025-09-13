"""
Tests for data preprocessing module.

This module contains unit tests for the DataPreprocessor class,
testing data cleaning, aggregation, and merging functionality.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.preprocessing import DataPreprocessor, DataPreprocessingError


class TestDataPreprocessor:
    """Test cases for DataPreprocessor class."""
    
    @pytest.fixture
    def preprocessor(self):
        """Create a DataPreprocessor instance for testing."""
        return DataPreprocessor(use_polars=False)
    
    @pytest.fixture
    def sample_transaction_data(self):
        """Create sample transaction data for testing."""
        dates = pd.date_range('2022-01-01', '2022-01-31', freq='D')
        data = []
        
        for i, date in enumerate(dates):
            data.extend([
                {
                    'internal_store_id': f'store_{i % 3}',
                    'internal_product_id': f'product_{i % 5}',
                    'transaction_date': date,
                    'quantity': np.random.randint(1, 10),
                    'gross_value': np.random.uniform(10, 100),
                    'net_value': np.random.uniform(8, 90),
                    'distributor_id': str(i % 3)
                }
                for _ in range(np.random.randint(1, 5))
            ])
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_store_data(self):
        """Create sample store master data for testing."""
        return pd.DataFrame([
            {'pdv': 'store_0', 'premise': 'On Premise', 'categoria_pdv': 'Restaurant', 'zipcode': 12345},
            {'pdv': 'store_1', 'premise': 'Off Premise', 'categoria_pdv': 'Convenience', 'zipcode': 23456},
            {'pdv': 'store_2', 'premise': 'On Premise', 'categoria_pdv': 'Bar', 'zipcode': 34567}
        ])
    
    @pytest.fixture
    def sample_product_data(self):
        """Create sample product master data for testing."""
        return pd.DataFrame([
            {'produto': 'product_0', 'categoria': 'Beverages', 'marca': 'Brand A'},
            {'produto': 'product_1', 'categoria': 'Food', 'marca': 'Brand B'},
            {'produto': 'product_2', 'categoria': 'Beverages', 'marca': 'Brand C'},
            {'produto': 'product_3', 'categoria': 'Snacks', 'marca': 'Brand A'},
            {'produto': 'product_4', 'categoria': 'Food', 'marca': 'Brand D'}
        ])
    
    def test_clean_transactions_basic(self, preprocessor, sample_transaction_data):
        """Test basic transaction cleaning functionality."""
        result = preprocessor.clean_transactions(sample_transaction_data)
        
        # Check that required columns exist
        required_columns = ['pdv', 'produto', 'data', 'quantidade']
        for col in required_columns:
            assert col in result.columns, f"Required column {col} missing"
        
        # Check data types
        assert pd.api.types.is_datetime64_any_dtype(result['data']), "Date column should be datetime"
        assert pd.api.types.is_numeric_dtype(result['quantidade']), "Quantity should be numeric"
        
        # Check no negative quantities
        assert (result['quantidade'] > 0).all(), "All quantities should be positive"
        
        # Check date range (should be 2022)
        assert result['data'].dt.year.eq(2022).all(), "All dates should be in 2022"
    
    def test_clean_transactions_with_missing_values(self, preprocessor):
        """Test transaction cleaning with missing values."""
        data_with_nulls = pd.DataFrame([
            {'internal_store_id': 'store_1', 'internal_product_id': 'product_1', 
             'transaction_date': '2022-01-01', 'quantity': 5},
            {'internal_store_id': 'store_2', 'internal_product_id': None, 
             'transaction_date': '2022-01-02', 'quantity': 3},
            {'internal_store_id': 'store_3', 'internal_product_id': 'product_3', 
             'transaction_date': '2022-01-03', 'quantity': None},
            {'internal_store_id': None, 'internal_product_id': 'product_4', 
             'transaction_date': '2022-01-04', 'quantity': 7}
        ])
        
        result = preprocessor.clean_transactions(data_with_nulls)
        
        # Should remove records with missing quantities (record 3), but keep others
        assert len(result) == 3, "Should remove records with missing quantities only"
        
        # Check that the record with missing quantity was removed
        assert not result['quantidade'].isnull().any(), "No null quantities should remain"
        
        # Check that records with missing IDs are kept (they can be handled in merging)
        assert 'store_1' in result['pdv'].values, "Should keep valid record"
    
    def test_clean_transactions_with_invalid_data(self, preprocessor):
        """Test transaction cleaning with invalid data."""
        invalid_data = pd.DataFrame([
            {'internal_store_id': 'store_1', 'internal_product_id': 'product_1', 
             'transaction_date': '2022-01-01', 'quantity': 5},
            {'internal_store_id': 'store_2', 'internal_product_id': 'product_2', 
             'transaction_date': '2022-01-02', 'quantity': -3},  # Negative quantity
            {'internal_store_id': 'store_3', 'internal_product_id': 'product_3', 
             'transaction_date': '2021-01-03', 'quantity': 4},  # Wrong year
            {'internal_store_id': 'store_4', 'internal_product_id': 'product_4', 
             'transaction_date': '2022-01-04', 'quantity': 0}   # Zero quantity
        ])
        
        result = preprocessor.clean_transactions(invalid_data)
        
        # Should only keep the first record
        assert len(result) == 1, "Should remove invalid records"
        assert result.iloc[0]['quantidade'] == 5, "Should keep valid quantity"
    
    def test_aggregate_weekly_sales(self, preprocessor, sample_transaction_data):
        """Test weekly sales aggregation."""
        # First clean the data
        clean_data = preprocessor.clean_transactions(sample_transaction_data)
        
        # Then aggregate
        result = preprocessor.aggregate_weekly_sales(clean_data)
        
        # Check required columns
        assert 'semana' in result.columns, "Week column should exist"
        assert 'ano' in result.columns, "Year column should exist"
        assert 'quantidade' in result.columns, "Quantity should be aggregated"
        
        # Check that quantities are summed (should be >= original individual quantities)
        assert result['quantidade'].min() >= 1, "Aggregated quantities should be positive"
        
        # Check that we have fewer records (aggregated)
        assert len(result) <= len(clean_data), "Aggregated data should have fewer records"
        
        # Check week numbers are valid (1-53)
        assert result['semana'].min() >= 1, "Week numbers should be >= 1"
        assert result['semana'].max() <= 53, "Week numbers should be <= 53"
    
    def test_aggregate_weekly_sales_consistency(self, preprocessor):
        """Test that weekly aggregation preserves total quantities."""
        # Create test data with known quantities
        test_data = pd.DataFrame([
            {'pdv': 'store_1', 'produto': 'product_1', 'data': '2022-01-01', 'quantidade': 10},
            {'pdv': 'store_1', 'produto': 'product_1', 'data': '2022-01-02', 'quantidade': 5},
            {'pdv': 'store_1', 'produto': 'product_1', 'data': '2022-01-03', 'quantidade': 3},
            {'pdv': 'store_1', 'produto': 'product_2', 'data': '2022-01-01', 'quantidade': 7},
        ])
        
        result = preprocessor.aggregate_weekly_sales(test_data)
        
        # Check total quantities are preserved
        original_total = test_data['quantidade'].sum()
        aggregated_total = result['quantidade'].sum()
        assert original_total == aggregated_total, "Total quantities should be preserved"
        
        # Check that same product/store combinations are aggregated
        product_1_total = result[
            (result['pdv'] == 'store_1') & (result['produto'] == 'product_1')
        ]['quantidade'].sum()
        assert product_1_total == 18, "Product 1 quantities should be summed (10+5+3)"
    
    def test_merge_master_data_stores(self, preprocessor, sample_store_data):
        """Test merging with store master data."""
        # Create transaction data
        transactions = pd.DataFrame([
            {'pdv': 'store_0', 'produto': 'product_1', 'quantidade': 5},
            {'pdv': 'store_1', 'produto': 'product_2', 'quantidade': 3},
            {'pdv': 'store_unknown', 'produto': 'product_3', 'quantidade': 2}
        ])
        
        result = preprocessor.merge_master_data(transactions, stores=sample_store_data)
        
        # Check that store information is merged
        assert 'premise' in result.columns, "Store premise should be merged"
        assert 'categoria_pdv' in result.columns, "Store category should be merged"
        
        # Check that known stores have correct information
        store_0_row = result[result['pdv'] == 'store_0'].iloc[0]
        assert store_0_row['premise'] == 'On Premise', "Store 0 should have correct premise"
        assert store_0_row['categoria_pdv'] == 'Restaurant', "Store 0 should have correct category"
        
        # Check that unknown stores have 'Unknown' filled
        unknown_row = result[result['pdv'] == 'store_unknown'].iloc[0]
        assert unknown_row['premise'] == 'Unknown', "Unknown store should have 'Unknown' premise"
    
    def test_merge_master_data_products(self, preprocessor, sample_product_data):
        """Test merging with product master data."""
        # Create transaction data
        transactions = pd.DataFrame([
            {'pdv': 'store_1', 'produto': 'product_0', 'quantidade': 5},
            {'pdv': 'store_2', 'produto': 'product_1', 'quantidade': 3},
            {'pdv': 'store_3', 'produto': 'product_unknown', 'quantidade': 2}
        ])
        
        result = preprocessor.merge_master_data(transactions, products=sample_product_data)
        
        # Check that product information is merged
        assert 'categoria' in result.columns, "Product category should be merged"
        assert 'marca' in result.columns, "Product brand should be merged"
        
        # Check that known products have correct information
        product_0_row = result[result['produto'] == 'product_0'].iloc[0]
        assert product_0_row['categoria'] == 'Beverages', "Product 0 should have correct category"
        assert product_0_row['marca'] == 'Brand A', "Product 0 should have correct brand"
        
        # Check that unknown products have 'Unknown' filled
        unknown_row = result[result['produto'] == 'product_unknown'].iloc[0]
        assert unknown_row['categoria'] == 'Unknown', "Unknown product should have 'Unknown' category"
    
    def test_merge_master_data_both(self, preprocessor, sample_store_data, sample_product_data):
        """Test merging with both store and product master data."""
        transactions = pd.DataFrame([
            {'pdv': 'store_0', 'produto': 'product_0', 'quantidade': 5},
            {'pdv': 'store_1', 'produto': 'product_1', 'quantidade': 3}
        ])
        
        result = preprocessor.merge_master_data(
            transactions, 
            products=sample_product_data, 
            stores=sample_store_data
        )
        
        # Check that both store and product information is merged
        assert 'premise' in result.columns, "Store information should be present"
        assert 'categoria' in result.columns, "Product information should be present"
        
        # Check specific values
        first_row = result.iloc[0]
        assert first_row['premise'] == 'On Premise', "Store info should be correct"
        assert first_row['categoria'] == 'Beverages', "Product info should be correct"
    
    def test_create_time_features(self, preprocessor):
        """Test time feature creation."""
        test_data = pd.DataFrame([
            {'pdv': 'store_1', 'data': '2022-01-15', 'quantidade': 5},  # Saturday
            {'pdv': 'store_2', 'data': '2022-02-01', 'quantidade': 3},  # Tuesday
            {'pdv': 'store_3', 'data': '2022-12-31', 'quantidade': 2}   # Saturday
        ])
        
        result = preprocessor.create_time_features(test_data)
        
        # Check that time features are created
        time_features = ['mes', 'trimestre', 'dia_semana', 'dia_mes', 'semana_mes']
        for feature in time_features:
            assert feature in result.columns, f"Time feature {feature} should be created"
        
        # Check specific values
        jan_row = result[result['data'] == '2022-01-15'].iloc[0]
        assert jan_row['mes'] == 1, "January should be month 1"
        assert jan_row['trimestre'] == 1, "January should be Q1"
        assert jan_row['dia_mes'] == 15, "Should extract day correctly"
        
        # Check seasonal indicators
        assert 'is_inicio_mes' in result.columns, "Should have start of month indicator"
        assert 'is_fim_mes' in result.columns, "Should have end of month indicator"
        assert 'is_weekend' in result.columns, "Should have weekend indicator"
    
    def test_validate_processed_data(self, preprocessor):
        """Test data validation functionality."""
        # Valid data
        valid_data = pd.DataFrame([
            {'pdv': 'store_1', 'produto': 'product_1', 'quantidade': 5},
            {'pdv': 'store_2', 'produto': 'product_2', 'quantidade': 3}
        ])
        
        result = preprocessor.validate_processed_data(valid_data)
        
        assert result['is_valid'] == True, "Valid data should pass validation"
        assert len(result['errors']) == 0, "Valid data should have no errors"
        assert result['summary']['total_records'] == 2, "Should count records correctly"
        
        # Invalid data (missing required columns)
        invalid_data = pd.DataFrame([
            {'store': 'store_1', 'product': 'product_1', 'qty': 5}
        ])
        
        result_invalid = preprocessor.validate_processed_data(invalid_data)
        
        assert result_invalid['is_valid'] == False, "Invalid data should fail validation"
        assert len(result_invalid['errors']) > 0, "Invalid data should have errors"
    
    def test_get_preprocessing_summary(self, preprocessor):
        """Test preprocessing summary generation."""
        original_data = pd.DataFrame([
            {'internal_store_id': 'store_1', 'internal_product_id': 'product_1', 'quantity': 5},
            {'internal_store_id': 'store_2', 'internal_product_id': 'product_2', 'quantity': -3},
            {'internal_store_id': 'store_3', 'internal_product_id': 'product_3', 'quantity': 4}
        ])
        
        processed_data = pd.DataFrame([
            {'pdv': 'store_1', 'produto': 'product_1', 'quantidade': 5, 'mes': 1},
            {'pdv': 'store_3', 'produto': 'product_3', 'quantidade': 4, 'mes': 1}
        ])
        
        summary = preprocessor.get_preprocessing_summary(original_data, processed_data)
        
        assert summary['original_records'] == 3, "Should count original records"
        assert summary['processed_records'] == 2, "Should count processed records"
        assert summary['records_removed'] == 1, "Should count removed records"
        assert summary['columns_added'] == 1, "Should count added columns"
    
    def test_outlier_handling_iqr(self, preprocessor):
        """Test outlier handling using IQR method."""
        # Create data with outliers
        data_with_outliers = pd.DataFrame([
            {'pdv': 'store_1', 'produto': 'product_1', 'data': '2022-01-01', 'quantidade': 5},
            {'pdv': 'store_1', 'produto': 'product_1', 'data': '2022-01-02', 'quantidade': 6},
            {'pdv': 'store_1', 'produto': 'product_1', 'data': '2022-01-03', 'quantidade': 7},
            {'pdv': 'store_1', 'produto': 'product_1', 'data': '2022-01-04', 'quantidade': 1000}  # Outlier
        ])
        
        result = preprocessor._handle_outliers(data_with_outliers, method='iqr')
        
        # Check that outlier is capped, not removed
        assert len(result) == len(data_with_outliers), "Should not remove records"
        assert result['quantidade'].max() < 1000, "Outlier should be capped"
        assert result['quantidade'].min() > 0, "Should maintain positive values"
    
    def test_missing_date_column_error(self, preprocessor):
        """Test error handling when date column is missing for aggregation."""
        data_no_date = pd.DataFrame([
            {'pdv': 'store_1', 'produto': 'product_1', 'quantidade': 5}
        ])
        
        with pytest.raises(DataPreprocessingError, match="Date column 'data' not found"):
            preprocessor.aggregate_weekly_sales(data_no_date)
    
    def test_empty_dataframe_handling(self, preprocessor):
        """Test handling of empty DataFrames."""
        empty_df = pd.DataFrame()
        
        # Should handle empty DataFrame gracefully
        result = preprocessor.validate_processed_data(empty_df)
        assert result['is_valid'] == False, "Empty DataFrame should be invalid"
        assert "DataFrame is empty" in result['errors'], "Should report empty DataFrame error"


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])