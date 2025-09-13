"""
Tests for EDA utilities and analysis functions.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from utils.eda_utils import EDAAnalyzer


class TestEDAAnalyzer:
    """Test cases for EDA analyzer."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        
        # Create sample transaction data
        n_records = 1000
        data = {
            'data': pd.date_range('2022-01-01', periods=n_records, freq='D'),
            'pdv': np.random.choice(['PDV_001', 'PDV_002', 'PDV_003'], n_records),
            'produto': np.random.choice(['PROD_A', 'PROD_B', 'PROD_C', 'PROD_D'], n_records),
            'quantidade': np.random.poisson(10, n_records),
            'categoria': np.random.choice(['Cat1', 'Cat2', 'Cat3'], n_records),
            'tipo_pdv': np.random.choice(['c-store', 'g-store', 'liquor'], n_records),
            'valor': np.random.normal(100, 20, n_records)
        }
        
        df = pd.DataFrame(data)
        
        # Add some missing values
        df.loc[np.random.choice(df.index, 50), 'valor'] = np.nan
        
        # Add some outliers
        df.loc[np.random.choice(df.index, 20), 'quantidade'] = np.random.randint(100, 200, 20)
        
        return df
    
    @pytest.fixture
    def eda_analyzer(self):
        """Create EDA analyzer instance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            analyzer = EDAAnalyzer(save_plots=False, plot_dir=temp_dir)
            yield analyzer
    
    def test_analyze_data_quality(self, eda_analyzer, sample_data):
        """Test data quality analysis."""
        quality_report = eda_analyzer.analyze_data_quality(sample_data)
        
        # Check basic structure
        assert 'basic_info' in quality_report
        assert 'missing_values' in quality_report
        assert 'duplicates' in quality_report
        assert 'data_types' in quality_report
        
        # Check basic info
        basic_info = quality_report['basic_info']
        assert basic_info['total_records'] == len(sample_data)
        assert basic_info['total_columns'] == len(sample_data.columns)
        assert basic_info['memory_usage_mb'] > 0
        
        # Check missing values detection
        missing_info = quality_report['missing_values']
        assert missing_info['total_missing'] > 0  # We added missing values
        assert 'valor' in missing_info['missing_by_column']
        
        # Check data types classification
        data_types = quality_report['data_types']
        assert 'quantidade' in data_types['numeric_columns']
        assert 'categoria' in data_types['categorical_columns']
    
    def test_analyze_temporal_patterns(self, eda_analyzer, sample_data):
        """Test temporal pattern analysis."""
        temporal_report = eda_analyzer.analyze_temporal_patterns(
            sample_data, date_col='data', quantity_col='quantidade'
        )
        
        # Check structure
        assert 'date_column' in temporal_report
        assert 'quantity_column' in temporal_report
        assert 'temporal_features' in temporal_report
        assert 'seasonality' in temporal_report
        
        # Check date column detection
        assert temporal_report['date_column'] == 'data'
        assert temporal_report['quantity_column'] == 'quantidade'
        
        # Check temporal features
        features = temporal_report['temporal_features']
        assert 'date_range' in features
        assert 'total_weeks' in features
        assert 'total_months' in features
        
        # Check seasonality analysis
        seasonality = temporal_report['seasonality']
        assert 'monthly_variation_cv' in seasonality
        assert 'weekly_variation_cv' in seasonality
    
    def test_analyze_categorical_distributions(self, eda_analyzer, sample_data):
        """Test categorical distribution analysis."""
        categorical_report = eda_analyzer.analyze_categorical_distributions(sample_data)
        
        # Check structure
        assert 'product_analysis' in categorical_report
        assert 'store_analysis' in categorical_report
        
        # Check product analysis
        product_analysis = categorical_report['product_analysis']
        assert len(product_analysis) > 0  # Should find product-related columns
        
        # Check store analysis
        store_analysis = categorical_report['store_analysis']
        assert len(store_analysis) > 0  # Should find store-related columns
    
    def test_detect_outliers(self, eda_analyzer, sample_data):
        """Test outlier detection."""
        outlier_report = eda_analyzer.detect_outliers(sample_data)
        
        # Check that numeric columns are analyzed
        assert 'quantidade' in outlier_report
        assert 'valor' in outlier_report
        
        # Check outlier metrics for quantidade
        qty_outliers = outlier_report['quantidade']
        assert 'total_outliers' in qty_outliers
        assert 'outlier_percentage' in qty_outliers
        assert 'Q1' in qty_outliers
        assert 'Q3' in qty_outliers
        assert 'IQR' in qty_outliers
        
        # Should detect the outliers we added
        assert qty_outliers['total_outliers'] > 0
    
    def test_generate_correlation_analysis(self, eda_analyzer, sample_data):
        """Test correlation analysis."""
        correlation_report = eda_analyzer.generate_correlation_analysis(sample_data)
        
        # Check structure
        assert 'correlation_matrix' in correlation_report
        assert 'high_correlations' in correlation_report
        assert 'multicollinearity_risk' in correlation_report
        
        # Check correlation matrix
        corr_matrix = correlation_report['correlation_matrix']
        assert len(corr_matrix) > 0  # Should have correlations
        
        # High correlations and multicollinearity should be lists
        assert isinstance(correlation_report['high_correlations'], list)
        assert isinstance(correlation_report['multicollinearity_risk'], list)
    
    def test_generate_insights_summary(self, eda_analyzer, sample_data):
        """Test comprehensive insights generation."""
        insights_summary = eda_analyzer.generate_insights_summary(sample_data)
        
        # Check main structure
        assert 'dataset_overview' in insights_summary
        assert 'insights' in insights_summary
        assert 'recommendations' in insights_summary
        assert 'detailed_reports' in insights_summary
        
        # Check dataset overview
        overview = insights_summary['dataset_overview']
        assert overview['records'] == len(sample_data)
        assert overview['columns'] == len(sample_data.columns)
        assert overview['memory_mb'] > 0
        
        # Check insights and recommendations are lists
        assert isinstance(insights_summary['insights'], list)
        assert isinstance(insights_summary['recommendations'], list)
        assert len(insights_summary['insights']) > 0
        assert len(insights_summary['recommendations']) > 0
        
        # Check detailed reports
        detailed = insights_summary['detailed_reports']
        assert 'quality' in detailed
        assert 'temporal' in detailed
        assert 'categorical' in detailed
        assert 'outliers' in detailed
        assert 'correlations' in detailed
    
    def test_auto_column_detection(self, eda_analyzer):
        """Test automatic column detection."""
        # Create data with various column naming patterns
        data = {
            'transaction_date': pd.date_range('2022-01-01', periods=100),
            'internal_store_id': np.random.choice(['S1', 'S2'], 100),
            'internal_product_id': np.random.choice(['P1', 'P2'], 100),
            'quantity': np.random.poisson(5, 100),
            'product_category': np.random.choice(['A', 'B'], 100),
            'store_type': np.random.choice(['Type1', 'Type2'], 100)
        }
        
        df = pd.DataFrame(data)
        
        # Test temporal analysis with auto-detection
        temporal_report = eda_analyzer.analyze_temporal_patterns(df)
        assert temporal_report['date_column'] == 'transaction_date'
        assert temporal_report['quantity_column'] == 'quantity'
        
        # Test categorical analysis
        categorical_report = eda_analyzer.analyze_categorical_distributions(df)
        assert len(categorical_report['product_analysis']) > 0
        assert len(categorical_report['store_analysis']) > 0
    
    def test_empty_dataframe(self, eda_analyzer):
        """Test handling of empty dataframe."""
        empty_df = pd.DataFrame()
        
        quality_report = eda_analyzer.analyze_data_quality(empty_df)
        assert quality_report['basic_info']['total_records'] == 0
        assert quality_report['basic_info']['total_columns'] == 0
    
    def test_no_numeric_columns(self, eda_analyzer):
        """Test handling of dataframe with no numeric columns."""
        data = {
            'col1': ['A', 'B', 'C'] * 10,
            'col2': ['X', 'Y', 'Z'] * 10
        }
        df = pd.DataFrame(data)
        
        outlier_report = eda_analyzer.detect_outliers(df)
        assert len(outlier_report) == 0
        
        correlation_report = eda_analyzer.generate_correlation_analysis(df)
        assert len(correlation_report['correlation_matrix']) == 0