"""
Tests for feature engineering module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from features.engineering import FeatureEngineer, FeatureEngineeringError


class TestFeatureEngineer:
    """Test cases for FeatureEngineer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        dates = pd.date_range('2022-01-01', '2022-12-31', freq='W')
        data = {
            'data_semana': dates,
            'pdv': ['PDV001'] * len(dates),
            'produto': ['PROD001'] * len(dates),
            'quantidade': np.random.randint(1, 100, len(dates))
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def feature_engineer(self):
        """Create FeatureEngineer instance."""
        return FeatureEngineer(country_code='BR')
    
    def test_initialization(self, feature_engineer):
        """Test FeatureEngineer initialization."""
        assert feature_engineer.country_code == 'BR'
        assert hasattr(feature_engineer, 'holiday_calendar')
        assert hasattr(feature_engineer, 'logger')
    
    def test_create_temporal_features_success(self, feature_engineer, sample_data):
        """Test successful temporal feature creation."""
        result = feature_engineer.create_temporal_features(sample_data)
        
        # Check that original columns are preserved
        for col in sample_data.columns:
            assert col in result.columns
        
        # Check basic temporal features
        expected_temporal_features = [
            'semana_ano', 'mes', 'trimestre', 'ano',
            'dia_semana', 'dia_mes', 'dia_ano', 'semana_mes',
            'is_inicio_mes', 'is_meio_mes', 'is_fim_mes', 'is_weekend'
        ]
        
        for feature in expected_temporal_features:
            assert feature in result.columns, f"Missing feature: {feature}"
        
        # Check seasonality features
        expected_seasonal_features = [
            'semana_sin', 'semana_cos', 'mes_sin', 'mes_cos',
            'dia_semana_sin', 'dia_semana_cos', 'trimestre_sin', 'trimestre_cos',
            'is_q1', 'is_q2', 'is_q3', 'is_q4',
            'is_janeiro', 'is_dezembro', 'is_junho_julho'
        ]
        
        for feature in expected_seasonal_features:
            assert feature in result.columns, f"Missing seasonal feature: {feature}"
        
        # Check holiday features
        expected_holiday_features = [
            'is_feriado', 'dias_ate_feriado', 'dias_pos_feriado',
            'is_pre_feriado', 'is_pos_feriado',
            'is_black_friday', 'is_natal', 'is_volta_aulas', 'is_dia_maes'
        ]
        
        for feature in expected_holiday_features:
            assert feature in result.columns, f"Missing holiday feature: {feature}"
        
        # Check trend features
        expected_trend_features = [
            'dias_desde_inicio', 'semanas_desde_inicio',
            'trend_normalizado', 'trend_quadratico', 'trend_anual'
        ]
        
        for feature in expected_trend_features:
            assert feature in result.columns, f"Missing trend feature: {feature}"
    
    def test_create_temporal_features_missing_date_column(self, feature_engineer, sample_data):
        """Test temporal feature creation with missing date column."""
        sample_data_no_date = sample_data.drop('data_semana', axis=1)
        
        with pytest.raises(FeatureEngineeringError):
            feature_engineer.create_temporal_features(sample_data_no_date)
    
    def test_basic_temporal_features_values(self, feature_engineer, sample_data):
        """Test that basic temporal features have correct values."""
        result = feature_engineer.create_temporal_features(sample_data)
        
        # Test a specific date
        test_date = pd.Timestamp('2022-06-15')  # Wednesday, June 15, 2022
        test_row = result[result['data_semana'].dt.date == test_date.date()]
        
        if not test_row.empty:
            row = test_row.iloc[0]
            assert row['mes'] == 6
            assert row['trimestre'] == 2
            assert row['ano'] == 2022
            assert row['dia_semana'] == 2  # Wednesday (0=Monday)
            assert row['is_q2'] == 1
            assert row['is_junho_julho'] == 1
    
    def test_seasonality_features_cyclical_encoding(self, feature_engineer, sample_data):
        """Test that cyclical encoding produces values in correct range."""
        result = feature_engineer.create_temporal_features(sample_data)
        
        # Cyclical features should be between -1 and 1
        cyclical_features = ['semana_sin', 'semana_cos', 'mes_sin', 'mes_cos',
                           'dia_semana_sin', 'dia_semana_cos', 'trimestre_sin', 'trimestre_cos']
        
        for feature in cyclical_features:
            assert result[feature].min() >= -1.0, f"{feature} has values below -1"
            assert result[feature].max() <= 1.0, f"{feature} has values above 1"
    
    def test_holiday_features_binary(self, feature_engineer, sample_data):
        """Test that holiday features are binary (0 or 1)."""
        result = feature_engineer.create_temporal_features(sample_data)
        
        binary_features = ['is_feriado', 'is_pre_feriado', 'is_pos_feriado',
                          'is_black_friday', 'is_natal', 'is_volta_aulas', 'is_dia_maes']
        
        for feature in binary_features:
            unique_values = result[feature].unique()
            assert set(unique_values).issubset({0, 1}), f"{feature} has non-binary values: {unique_values}"
    
    def test_trend_features_monotonic(self, feature_engineer, sample_data):
        """Test that trend features are monotonic."""
        result = feature_engineer.create_temporal_features(sample_data)
        
        # Sort by date to check monotonicity
        result_sorted = result.sort_values('data_semana')
        
        # dias_desde_inicio should be monotonically increasing
        dias_diff = result_sorted['dias_desde_inicio'].diff().dropna()
        assert (dias_diff >= 0).all(), "dias_desde_inicio is not monotonically increasing"
        
        # trend_normalizado should be between 0 and 1
        assert result['trend_normalizado'].min() >= 0, "trend_normalizado has values below 0"
        assert result['trend_normalizado'].max() <= 1, "trend_normalizado has values above 1"
    
    def test_special_retail_periods(self, feature_engineer):
        """Test identification of special retail periods."""
        # Create specific dates for testing
        test_dates = pd.DataFrame({
            'data_semana': [
                pd.Timestamp('2022-12-15'),  # Christmas period
                pd.Timestamp('2022-01-15'),  # Back to school
                pd.Timestamp('2022-05-15'),  # Mother's Day period
                pd.Timestamp('2022-11-25'),  # Black Friday period
                pd.Timestamp('2022-07-15'),  # Regular period
            ],
            'pdv': ['PDV001'] * 5,
            'produto': ['PROD001'] * 5,
            'quantidade': [10] * 5
        })
        
        result = feature_engineer.create_temporal_features(test_dates)
        
        # Check Christmas period (December)
        christmas_row = result[result['data_semana'].dt.month == 12].iloc[0]
        assert christmas_row['is_natal'] == 1
        
        # Check back to school (January)
        school_row = result[result['data_semana'].dt.month == 1].iloc[0]
        assert school_row['is_volta_aulas'] == 1
        
        # Check Mother's Day (May)
        mothers_row = result[result['data_semana'].dt.month == 5].iloc[0]
        assert mothers_row['is_dia_maes'] == 1
    
    def test_empty_dataframe(self, feature_engineer):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame(columns=['data_semana', 'pdv', 'produto', 'quantidade'])
        
        result = feature_engineer.create_temporal_features(empty_df)
        
        # Should return DataFrame with all feature columns but no rows
        assert len(result) == 0
        assert 'semana_ano' in result.columns
        assert 'mes' in result.columns
    
    def test_single_row_dataframe(self, feature_engineer):
        """Test handling of single row DataFrame."""
        single_row = pd.DataFrame({
            'data_semana': [pd.Timestamp('2022-06-15')],
            'pdv': ['PDV001'],
            'produto': ['PROD001'],
            'quantidade': [50]
        })
        
        result = feature_engineer.create_temporal_features(single_row)
        
        assert len(result) == 1
        assert result.iloc[0]['mes'] == 6
        assert result.iloc[0]['trimestre'] == 2
        assert result.iloc[0]['trend_normalizado'] == 0  # Only one point, so normalized trend is 0
    
    def test_date_conversion(self, feature_engineer):
        """Test automatic date conversion."""
        # Test with string dates
        string_dates = pd.DataFrame({
            'data_semana': ['2022-01-01', '2022-01-08', '2022-01-15'],
            'pdv': ['PDV001'] * 3,
            'produto': ['PROD001'] * 3,
            'quantidade': [10, 20, 30]
        })
        
        result = feature_engineer.create_temporal_features(string_dates)
        
        # Should successfully convert and create features
        assert len(result) == 3
        assert 'mes' in result.columns
        assert result['mes'].iloc[0] == 1  # January
    
    def test_different_date_column_name(self, feature_engineer, sample_data):
        """Test with different date column name."""
        # Rename the date column
        sample_data_renamed = sample_data.rename(columns={'data_semana': 'custom_date'})
        
        result = feature_engineer.create_temporal_features(sample_data_renamed, date_column='custom_date')
        
        # Should work with custom column name
        assert len(result) == len(sample_data)
        assert 'mes' in result.columns
        assert 'custom_date' in result.columns
    
    def test_create_product_features_success(self, feature_engineer):
        """Test successful product feature creation."""
        # Create sample data with product information
        sample_data = pd.DataFrame({
            'produto': ['PROD001', 'PROD002', 'PROD001', 'PROD003'] * 10,
            'categoria': ['Bebidas', 'Alimentos', 'Bebidas', 'Limpeza'] * 10,
            'marca': ['MarcaA', 'MarcaB', 'MarcaA', 'MarcaC'] * 10,
            'preco_unitario': [10.5, 5.0, 10.5, 15.0] * 10,
            'quantidade': np.random.randint(1, 50, 40),
            'pdv': ['PDV001'] * 40,
            'data_semana': pd.date_range('2022-01-01', periods=40, freq='W')
        })
        
        result = feature_engineer.create_product_features(sample_data)
        
        # Check that original columns are preserved
        for col in sample_data.columns:
            assert col in result.columns
        
        # Check product category features
        assert 'categoria_bebidas' in result.columns
        assert 'categoria_alimentos' in result.columns
        assert 'categoria_outros' in result.columns
        
        # Check brand features
        assert 'marca_num_produtos' in result.columns
        assert 'is_marca_top' in result.columns
        
        # Check price-based features
        assert 'is_produto_barato' in result.columns
        assert 'is_produto_medio' in result.columns
        assert 'is_produto_caro' in result.columns
        
        # Check performance features
        assert 'produto_qty_media' in result.columns
        assert 'produto_qty_std' in result.columns
        assert 'produto_qty_cv' in result.columns
        
        # Check ranking features
        assert 'produto_rank_geral' in result.columns
        assert 'produto_percentil_geral' in result.columns
        assert 'is_produto_top10' in result.columns
    
    def test_create_store_features_success(self, feature_engineer):
        """Test successful store feature creation."""
        # Create sample data with store information
        sample_data = pd.DataFrame({
            'pdv': ['PDV001', 'PDV002', 'PDV003'] * 15,
            'premise': ['c-store', 'g-store', 'liquor'] * 15,
            'categoria_pdv': ['Conveniencia', 'Supermercado', 'Especializada'] * 15,
            'zipcode': [12345, 54321, 98765] * 15,
            'quantidade': np.random.randint(1, 100, 45),
            'produto': ['PROD001'] * 45,
            'data_semana': pd.date_range('2022-01-01', periods=45, freq='W')
        })
        
        result = feature_engineer.create_store_features(sample_data)
        
        # Check that original columns are preserved
        for col in sample_data.columns:
            assert col in result.columns
        
        # Check store type features
        assert 'store_type_c_store' in result.columns
        assert 'store_type_g_store' in result.columns
        assert 'pdv_categoria_conveniencia' in result.columns
        
        # Check location features
        assert 'regiao_zipcode' in result.columns
        assert 'zipcode_densidade' in result.columns
        assert 'is_area_urbana' in result.columns
        
        # Check performance features
        assert 'pdv_qty_media' in result.columns
        assert 'pdv_qty_std' in result.columns
        assert 'pdv_num_produtos' in result.columns
        
        # Check ranking features
        assert 'pdv_rank_geral' in result.columns
        assert 'pdv_percentil_geral' in result.columns
        assert 'is_pdv_top10' in result.columns
    
    def test_product_features_without_optional_columns(self, feature_engineer):
        """Test product features creation without optional columns."""
        # Minimal data without category, brand, price
        minimal_data = pd.DataFrame({
            'produto': ['PROD001', 'PROD002'] * 10,
            'quantidade': np.random.randint(1, 50, 20),
            'pdv': ['PDV001'] * 20
        })
        
        result = feature_engineer.create_product_features(minimal_data)
        
        # Should still create performance and ranking features
        assert 'produto_qty_media' in result.columns
        assert 'produto_rank_geral' in result.columns
        
        # Should not have category-specific features
        assert 'categoria_bebidas' not in result.columns
    
    def test_store_features_without_optional_columns(self, feature_engineer):
        """Test store features creation without optional columns."""
        # Minimal data without premise, category, zipcode
        minimal_data = pd.DataFrame({
            'pdv': ['PDV001', 'PDV002'] * 10,
            'quantidade': np.random.randint(1, 50, 20),
            'produto': ['PROD001'] * 20
        })
        
        result = feature_engineer.create_store_features(minimal_data)
        
        # Should still create performance and ranking features
        assert 'pdv_qty_media' in result.columns
        assert 'pdv_rank_geral' in result.columns
        
        # Should not have type-specific features
        assert 'store_type_c_store' not in result.columns
    
    def test_product_ranking_values(self, feature_engineer):
        """Test that product ranking features have correct values."""
        # Create data with known ranking
        sample_data = pd.DataFrame({
            'produto': ['PROD001', 'PROD002', 'PROD003'],
            'quantidade': [100, 50, 25],  # Clear ranking
            'pdv': ['PDV001'] * 3
        })
        
        result = feature_engineer.create_product_features(sample_data)
        
        # Check rankings
        prod001_rank = result[result['produto'] == 'PROD001']['produto_rank_geral'].iloc[0]
        prod002_rank = result[result['produto'] == 'PROD002']['produto_rank_geral'].iloc[0]
        prod003_rank = result[result['produto'] == 'PROD003']['produto_rank_geral'].iloc[0]
        
        assert prod001_rank == 1  # Highest quantity
        assert prod002_rank == 2  # Middle quantity
        assert prod003_rank == 3  # Lowest quantity
    
    def test_store_ranking_values(self, feature_engineer):
        """Test that store ranking features have correct values."""
        # Create data with known ranking
        sample_data = pd.DataFrame({
            'pdv': ['PDV001', 'PDV002', 'PDV003'],
            'quantidade': [200, 100, 50],  # Clear ranking
            'produto': ['PROD001'] * 3
        })
        
        result = feature_engineer.create_store_features(sample_data)
        
        # Check rankings
        pdv001_rank = result[result['pdv'] == 'PDV001']['pdv_rank_geral'].iloc[0]
        pdv002_rank = result[result['pdv'] == 'PDV002']['pdv_rank_geral'].iloc[0]
        pdv003_rank = result[result['pdv'] == 'PDV003']['pdv_rank_geral'].iloc[0]
        
        assert pdv001_rank == 1  # Highest quantity
        assert pdv002_rank == 2  # Middle quantity
        assert pdv003_rank == 3  # Lowest quantity
    
    def test_create_lag_features_success(self, feature_engineer):
        """Test successful lag feature creation."""
        # Create time series data
        dates = pd.date_range('2022-01-01', periods=20, freq='W')
        sample_data = pd.DataFrame({
            'data_semana': dates,
            'pdv': ['PDV001'] * 20,
            'produto': ['PROD001'] * 20,
            'quantidade': range(1, 21)  # Sequential values for easy testing
        })
        
        result = feature_engineer.create_lag_features(sample_data)
        
        # Check that original columns are preserved
        for col in sample_data.columns:
            assert col in result.columns
        
        # Check lag features
        expected_lag_features = [
            'quantidade_lag_1', 'quantidade_lag_2', 'quantidade_lag_4', 'quantidade_lag_8'
        ]
        
        for feature in expected_lag_features:
            assert feature in result.columns, f"Missing lag feature: {feature}"
        
        # Check lag ratios and differences
        assert 'quantidade_ratio_lag1' in result.columns
        assert 'quantidade_diff_lag1' in result.columns
        
        # Test lag values
        # Row 5 (index 4) should have lag_1 = value from row 4 (index 3)
        if len(result) > 4:
            assert result.iloc[4]['quantidade_lag_1'] == result.iloc[3]['quantidade']
    
    def test_create_rolling_features_success(self, feature_engineer):
        """Test successful rolling feature creation."""
        # Create time series data
        dates = pd.date_range('2022-01-01', periods=20, freq='W')
        sample_data = pd.DataFrame({
            'data_semana': dates,
            'pdv': ['PDV001'] * 20,
            'produto': ['PROD001'] * 20,
            'quantidade': [10] * 20  # Constant values for easy testing
        })
        
        result = feature_engineer.create_rolling_features(sample_data)
        
        # Check that original columns are preserved
        for col in sample_data.columns:
            assert col in result.columns
        
        # Check rolling features
        expected_rolling_features = [
            'quantidade_rolling_mean_4', 'quantidade_rolling_std_4',
            'quantidade_rolling_min_4', 'quantidade_rolling_max_4',
            'quantidade_rolling_median_4', 'quantidade_rolling_cv_4',
            'quantidade_rolling_trend_4'
        ]
        
        for feature in expected_rolling_features:
            assert feature in result.columns, f"Missing rolling feature: {feature}"
        
        # Test rolling mean with constant values
        # All rolling means should be 10 (the constant value)
        assert all(result['quantidade_rolling_mean_4'] == 10)
        
        # Rolling std should be 0 for constant values
        assert all(result['quantidade_rolling_std_4'] == 0)
    
    def test_create_growth_features_success(self, feature_engineer):
        """Test successful growth feature creation."""
        # Create time series data with growth pattern
        dates = pd.date_range('2022-01-01', periods=20, freq='W')
        sample_data = pd.DataFrame({
            'data_semana': dates,
            'pdv': ['PDV001'] * 20,
            'produto': ['PROD001'] * 20,
            'quantidade': [10 * (1.1 ** i) for i in range(20)]  # 10% growth each week
        })
        
        result = feature_engineer.create_growth_features(sample_data)
        
        # Check that original columns are preserved
        for col in sample_data.columns:
            assert col in result.columns
        
        # Check growth features
        expected_growth_features = [
            'quantidade_pct_change_1w', 'quantidade_pct_change_4w',
            'quantidade_cumulative_growth', 'quantidade_growth_acceleration',
            'quantidade_growth_volatility'
        ]
        
        for feature in expected_growth_features:
            assert feature in result.columns, f"Missing growth feature: {feature}"
        
        # Test growth values (should be approximately 0.1 for 10% growth)
        # Skip first value (NaN due to pct_change)
        growth_values = result['quantidade_pct_change_1w'].dropna()
        if len(growth_values) > 0:
            # Growth should be approximately 10% (0.1)
            assert abs(growth_values.mean() - 0.1) < 0.01
    
    def test_lag_features_with_multiple_groups(self, feature_engineer):
        """Test lag features with multiple PDV/product combinations."""
        # Create data for multiple groups
        dates = pd.date_range('2022-01-01', periods=10, freq='W')
        sample_data = pd.DataFrame({
            'data_semana': list(dates) * 2,
            'pdv': ['PDV001'] * 10 + ['PDV002'] * 10,
            'produto': ['PROD001'] * 20,
            'quantidade': list(range(1, 11)) + list(range(11, 21))
        })
        
        result = feature_engineer.create_lag_features(sample_data)
        
        # Check that lag features are calculated separately for each group
        pdv001_data = result[result['pdv'] == 'PDV001'].sort_values('data_semana')
        pdv002_data = result[result['pdv'] == 'PDV002'].sort_values('data_semana')
        
        # First row of each group should have NaN for lag_1
        assert pd.isna(pdv001_data.iloc[0]['quantidade_lag_1'])
        assert pd.isna(pdv002_data.iloc[0]['quantidade_lag_1'])
        
        # Second row should have lag_1 equal to first row's quantidade
        if len(pdv001_data) > 1:
            assert pdv001_data.iloc[1]['quantidade_lag_1'] == pdv001_data.iloc[0]['quantidade']
        if len(pdv002_data) > 1:
            assert pdv002_data.iloc[1]['quantidade_lag_1'] == pdv002_data.iloc[0]['quantidade']
    
    def test_create_all_lag_and_rolling_features(self, feature_engineer):
        """Test creation of all lag and rolling features together."""
        # Create time series data
        dates = pd.date_range('2022-01-01', periods=15, freq='W')
        sample_data = pd.DataFrame({
            'data_semana': dates,
            'pdv': ['PDV001'] * 15,
            'produto': ['PROD001'] * 15,
            'quantidade': range(1, 16)
        })
        
        result = feature_engineer.create_all_lag_and_rolling_features(sample_data)
        
        # Check that all feature types are present
        feature_types = ['lag', 'rolling', 'pct_change', 'growth']
        
        for feature_type in feature_types:
            matching_columns = [col for col in result.columns if feature_type in col]
            assert len(matching_columns) > 0, f"No {feature_type} features found"
    
    def test_lag_features_missing_columns(self, feature_engineer):
        """Test lag features with missing required columns."""
        # Data without quantidade column
        sample_data = pd.DataFrame({
            'data_semana': pd.date_range('2022-01-01', periods=5, freq='W'),
            'pdv': ['PDV001'] * 5,
            'produto': ['PROD001'] * 5
        })
        
        with pytest.raises(FeatureEngineeringError):
            feature_engineer.create_lag_features(sample_data)
    
    def test_rolling_features_edge_cases(self, feature_engineer):
        """Test rolling features with edge cases."""
        # Single row data
        single_row = pd.DataFrame({
            'data_semana': [pd.Timestamp('2022-01-01')],
            'pdv': ['PDV001'],
            'produto': ['PROD001'],
            'quantidade': [10]
        })
        
        result = feature_engineer.create_rolling_features(single_row)
        
        # Should work with single row
        assert len(result) == 1
        assert 'quantidade_rolling_mean_4' in result.columns
        assert result.iloc[0]['quantidade_rolling_mean_4'] == 10


if __name__ == '__main__':
    pytest.main([__file__])