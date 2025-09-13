"""
Tests for feature selection module.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock

from src.features.selection import FeatureSelector, FeatureSelectionError


class TestFeatureSelector:
    """Test cases for FeatureSelector class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 1000
        
        # Create correlated features
        x1 = np.random.normal(0, 1, n_samples)
        x2 = x1 + np.random.normal(0, 0.1, n_samples)  # Highly correlated with x1
        x3 = np.random.normal(0, 1, n_samples)
        x4 = x3 * 2 + np.random.normal(0, 0.5, n_samples)  # Correlated with x3
        x5 = np.random.normal(0, 1, n_samples)  # Independent
        
        # Create target with some relationship to features
        target = 2 * x1 + 1.5 * x3 + 0.5 * x5 + np.random.normal(0, 0.5, n_samples)
        
        # Add some noise features
        noise_features = np.random.normal(0, 1, (n_samples, 5))
        
        df = pd.DataFrame({
            'feature_1': x1,
            'feature_2': x2,  # Highly correlated with feature_1
            'feature_3': x3,
            'feature_4': x4,  # Correlated with feature_3
            'feature_5': x5,
            'noise_1': noise_features[:, 0],
            'noise_2': noise_features[:, 1],
            'noise_3': noise_features[:, 2],
            'noise_4': noise_features[:, 3],
            'noise_5': noise_features[:, 4],
            'target': target
        })
        
        return df
    
    @pytest.fixture
    def feature_selector(self):
        """Create FeatureSelector instance."""
        return FeatureSelector(random_state=42)
    
    def test_initialization(self, feature_selector):
        """Test FeatureSelector initialization."""
        assert feature_selector.random_state == 42
        assert feature_selector.correlation_results_ is None
        assert feature_selector.importance_results_ is None
        assert feature_selector.selected_features_ is None
        assert feature_selector.multicollinearity_results_ is None
    
    def test_analyze_correlations_pearson(self, feature_selector, sample_data):
        """Test correlation analysis with Pearson method."""
        results = feature_selector.analyze_correlations(
            sample_data, 'target', method='pearson', threshold=0.8
        )
        
        assert 'correlation_matrix' in results
        assert 'target_correlations' in results
        assert 'high_correlation_pairs' in results
        assert results['method'] == 'pearson'
        assert results['threshold'] == 0.8
        
        # Check that highly correlated features are detected
        high_corr_pairs = results['high_correlation_pairs']
        assert len(high_corr_pairs) > 0
        
        # feature_1 and feature_2 should be highly correlated
        pair_features = [(pair['feature1'], pair['feature2']) for pair in high_corr_pairs]
        assert any(('feature_1' in pair and 'feature_2' in pair) for pair in pair_features)
    
    def test_analyze_correlations_spearman(self, feature_selector, sample_data):
        """Test correlation analysis with Spearman method."""
        results = feature_selector.analyze_correlations(
            sample_data, 'target', method='spearman', threshold=0.7
        )
        
        assert results['method'] == 'spearman'
        assert 'correlation_matrix' in results
    
    def test_analyze_correlations_invalid_method(self, feature_selector, sample_data):
        """Test correlation analysis with invalid method."""
        with pytest.raises(FeatureSelectionError, match="Unsupported correlation method"):
            feature_selector.analyze_correlations(sample_data, 'target', method='invalid')
    
    def test_analyze_correlations_missing_target(self, feature_selector, sample_data):
        """Test correlation analysis with missing target column."""
        with pytest.raises(FeatureSelectionError, match="Target column 'missing' not found"):
            feature_selector.analyze_correlations(sample_data, 'missing')
    
    def test_calculate_feature_importance_random_forest(self, feature_selector, sample_data):
        """Test feature importance calculation with Random Forest."""
        results = feature_selector.calculate_feature_importance(
            sample_data, 'target', method='random_forest'
        )
        
        assert 'importance_scores' in results
        assert 'method' in results
        assert results['method'] == 'random_forest'
        assert 'top_10_features' in results
        
        importance_df = results['importance_scores']
        assert len(importance_df) == 10  # All features except target
        assert 'importance_normalized' in importance_df.columns
        
        # Check that important features (feature_1, feature_3, feature_5) have high importance
        top_features = results['top_10_features'][:3]
        assert any(f in ['feature_1', 'feature_3', 'feature_5'] for f in top_features)
    
    def test_calculate_feature_importance_mutual_info(self, feature_selector, sample_data):
        """Test feature importance calculation with mutual information."""
        results = feature_selector.calculate_feature_importance(
            sample_data, 'target', method='mutual_info'
        )
        
        assert results['method'] == 'mutual_info'
        assert 'importance_scores' in results
    
    def test_calculate_feature_importance_f_score(self, feature_selector, sample_data):
        """Test feature importance calculation with F-score."""
        results = feature_selector.calculate_feature_importance(
            sample_data, 'target', method='f_score'
        )
        
        assert results['method'] == 'f_score'
        assert 'importance_scores' in results
    
    def test_calculate_feature_importance_invalid_method(self, feature_selector, sample_data):
        """Test feature importance calculation with invalid method."""
        with pytest.raises(FeatureSelectionError, match="Unsupported importance method"):
            feature_selector.calculate_feature_importance(sample_data, 'target', method='invalid')
    
    def test_select_features_rfe(self, feature_selector, sample_data):
        """Test RFE feature selection."""
        results = feature_selector.select_features_rfe(
            sample_data, 'target', n_features=5, estimator='random_forest'
        )
        
        assert 'selected_features' in results
        assert 'feature_rankings' in results
        assert 'n_features_selected' in results
        assert results['estimator'] == 'random_forest'
        
        selected_features = results['selected_features']
        assert len(selected_features) == 5
        assert all(isinstance(f, str) for f in selected_features)
        
        # Check that important features are likely selected
        important_features = ['feature_1', 'feature_3', 'feature_5']
        selected_important = [f for f in selected_features if f in important_features]
        assert len(selected_important) >= 2  # At least 2 important features should be selected
    
    def test_select_features_rfe_linear(self, feature_selector, sample_data):
        """Test RFE feature selection with linear regression."""
        results = feature_selector.select_features_rfe(
            sample_data, 'target', n_features=3, estimator='linear_regression'
        )
        
        assert results['estimator'] == 'linear_regression'
        assert len(results['selected_features']) == 3
    
    def test_select_features_rfe_invalid_estimator(self, feature_selector, sample_data):
        """Test RFE with invalid estimator."""
        with pytest.raises(FeatureSelectionError, match="Unsupported estimator"):
            feature_selector.select_features_rfe(sample_data, 'target', estimator='invalid')
    
    def test_select_features_kbest(self, feature_selector, sample_data):
        """Test SelectKBest feature selection."""
        results = feature_selector.select_features_kbest(
            sample_data, 'target', k=5, score_func='f_regression'
        )
        
        assert 'selected_features' in results
        assert 'feature_scores' in results
        assert 'n_features_selected' in results
        assert results['score_func'] == 'f_regression'
        
        selected_features = results['selected_features']
        assert len(selected_features) == 5
        
        # Check feature scores
        feature_scores = results['feature_scores']
        assert len(feature_scores) == 10
        assert 'score' in feature_scores.columns
        assert 'selected' in feature_scores.columns
    
    def test_select_features_kbest_mutual_info(self, feature_selector, sample_data):
        """Test SelectKBest with mutual information."""
        results = feature_selector.select_features_kbest(
            sample_data, 'target', k=3, score_func='mutual_info_regression'
        )
        
        assert results['score_func'] == 'mutual_info_regression'
        assert len(results['selected_features']) == 3
    
    def test_select_features_kbest_invalid_score_func(self, feature_selector, sample_data):
        """Test SelectKBest with invalid scoring function."""
        with pytest.raises(FeatureSelectionError, match="Unsupported scoring function"):
            feature_selector.select_features_kbest(sample_data, 'target', score_func='invalid')
    
    def test_validate_multicollinearity(self, feature_selector, sample_data):
        """Test multicollinearity validation."""
        features = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
        results = feature_selector.validate_multicollinearity(
            sample_data, features=features, vif_threshold=5.0
        )
        
        assert 'vif_results' in results
        assert 'high_vif_features' in results
        assert 'num_high_vif' in results
        assert results['vif_threshold'] == 5.0
        
        vif_df = results['vif_results']
        assert len(vif_df) == len(features)
        assert 'vif' in vif_df.columns
        assert 'high_multicollinearity' in vif_df.columns
        
        # Should detect some multicollinearity between correlated features
        high_vif_features = results['high_vif_features']
        assert isinstance(high_vif_features, list)
    
    def test_validate_multicollinearity_constant_features(self, feature_selector):
        """Test multicollinearity validation with constant features."""
        # Create data with constant feature
        df = pd.DataFrame({
            'feature_1': [1, 2, 3, 4, 5],
            'feature_2': [1, 1, 1, 1, 1],  # Constant feature
            'feature_3': [2, 4, 6, 8, 10],
            'target': [1, 2, 3, 4, 5]
        })
        
        results = feature_selector.validate_multicollinearity(df, vif_threshold=5.0)
        
        # Constant feature should be removed
        assert 'feature_2' in results['constant_features_removed']
    
    def test_create_feature_selection_pipeline(self, feature_selector, sample_data):
        """Test comprehensive feature selection pipeline."""
        results = feature_selector.create_feature_selection_pipeline(
            sample_data, 
            'target',
            correlation_threshold=0.8,
            vif_threshold=10.0,
            importance_method='random_forest',
            selection_method='rfe',
            n_features=5
        )
        
        assert 'correlation_analysis' in results
        assert 'importance_analysis' in results
        assert 'feature_selection' in results
        assert 'multicollinearity_validation' in results
        assert 'final_selected_features' in results
        assert 'pipeline_summary' in results
        
        # Check pipeline summary
        summary = results['pipeline_summary']
        assert 'initial_features' in summary
        assert 'final_features' in summary
        assert summary['final_features'] <= 5
        
        # Check that final features are stored
        final_features = results['final_selected_features']
        assert feature_selector.get_selected_features() == final_features
    
    def test_create_feature_selection_pipeline_kbest(self, feature_selector, sample_data):
        """Test pipeline with SelectKBest."""
        results = feature_selector.create_feature_selection_pipeline(
            sample_data, 
            'target',
            selection_method='kbest',
            n_features=3
        )
        
        assert 'feature_selection' in results
        assert len(results['final_selected_features']) <= 3
    
    def test_create_feature_selection_pipeline_invalid_method(self, feature_selector, sample_data):
        """Test pipeline with invalid selection method."""
        with pytest.raises(FeatureSelectionError, match="Unsupported selection method"):
            feature_selector.create_feature_selection_pipeline(
                sample_data, 'target', selection_method='invalid'
            )
    
    def test_get_selected_features_none(self, feature_selector):
        """Test get_selected_features when no selection has been run."""
        assert feature_selector.get_selected_features() is None
    
    def test_save_and_load_selection_results(self, feature_selector, sample_data):
        """Test saving and loading selection results."""
        # Run pipeline to generate results
        feature_selector.create_feature_selection_pipeline(
            sample_data, 'target', n_features=3
        )
        
        # Save results
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            feature_selector.save_selection_results(tmp_path)
            
            # Create new selector and load results
            new_selector = FeatureSelector()
            new_selector.load_selection_results(tmp_path)
            
            # Check that results were loaded
            assert new_selector.get_selected_features() == feature_selector.get_selected_features()
            assert new_selector.correlation_results_ is not None
            assert new_selector.importance_results_ is not None
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    @patch('matplotlib.pyplot.show')
    def test_plot_correlation_heatmap(self, mock_show, feature_selector, sample_data):
        """Test correlation heatmap plotting."""
        # First run correlation analysis
        results = feature_selector.analyze_correlations(sample_data, 'target')
        correlation_matrix = results['correlation_matrix']
        
        # Test plotting (mocked to avoid display)
        feature_selector.plot_correlation_heatmap(correlation_matrix)
        mock_show.assert_called_once()
    
    @patch('matplotlib.pyplot.show')
    def test_plot_feature_importance(self, mock_show, feature_selector, sample_data):
        """Test feature importance plotting."""
        # First run importance analysis
        results = feature_selector.calculate_feature_importance(sample_data, 'target')
        importance_df = results['importance_scores']
        
        # Test plotting (mocked to avoid display)
        feature_selector.plot_feature_importance(importance_df, top_n=5)
        mock_show.assert_called_once()
    
    def test_error_handling_empty_dataframe(self, feature_selector):
        """Test error handling with empty DataFrame."""
        empty_df = pd.DataFrame()
        
        with pytest.raises(FeatureSelectionError):
            feature_selector.analyze_correlations(empty_df, 'target')
    
    def test_error_handling_non_numeric_target(self, feature_selector):
        """Test error handling with non-numeric target."""
        df = pd.DataFrame({
            'feature_1': [1, 2, 3, 4, 5],
            'target': ['a', 'b', 'c', 'd', 'e']  # Non-numeric target
        })
        
        with pytest.raises(FeatureSelectionError, match="Target column 'target' is not numeric"):
            feature_selector.analyze_correlations(df, 'target')
    
    def test_feature_selection_with_missing_values(self, feature_selector):
        """Test feature selection with missing values."""
        # Create data with missing values
        df = pd.DataFrame({
            'feature_1': [1, 2, np.nan, 4, 5],
            'feature_2': [2, np.nan, 6, 8, 10],
            'feature_3': [1, 1, 1, 1, 1],
            'target': [1, 2, 3, 4, 5]
        })
        
        # Should handle missing values gracefully
        results = feature_selector.calculate_feature_importance(df, 'target')
        assert 'importance_scores' in results
        
        results = feature_selector.select_features_rfe(df, 'target', n_features=2)
        assert len(results['selected_features']) == 2
    
    def test_feature_selection_edge_cases(self, feature_selector):
        """Test edge cases in feature selection."""
        # Test with more features requested than available
        small_df = pd.DataFrame({
            'feature_1': [1, 2, 3, 4, 5],
            'feature_2': [2, 4, 6, 8, 10],
            'target': [1, 2, 3, 4, 5]
        })
        
        # Request more features than available
        results = feature_selector.select_features_rfe(small_df, 'target', n_features=10)
        assert len(results['selected_features']) == 2  # Only 2 features available
        
        results = feature_selector.select_features_kbest(small_df, 'target', k=10)
        assert len(results['selected_features']) == 2  # Only 2 features available


if __name__ == '__main__':
    pytest.main([__file__])