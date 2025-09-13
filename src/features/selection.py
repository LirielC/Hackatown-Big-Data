"""
Feature selection module for Hackathon Forecast Model 2025.

This module handles feature selection using correlation analysis, feature importance,
automatic selection techniques, and multicollinearity validation.
"""

import logging
import warnings
from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE, SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns


class FeatureSelectionError(Exception):
    """Custom exception for feature selection errors."""
    pass


class FeatureSelector:
    """
    Handles feature selection for sales forecasting model.
    
    This class provides methods for correlation analysis, feature importance calculation,
    automatic feature selection, and multicollinearity validation.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize FeatureSelector instance.

        Args:
            random_state: Random state for reproducible results
        """
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
        self._setup_logging()

        # Store selection results
        self.correlation_results_ = None
        self.importance_results_ = None

    def select_features(self, df: pd.DataFrame, method: str = 'rfe',
                       max_features: int = 50, target_column: str = 'quantidade') -> pd.DataFrame:
        """
        Wrapper method for feature selection that returns filtered DataFrame.

        Args:
            df: DataFrame with features and target
            method: Selection method ('rfe', 'importance', 'kbest')
            max_features: Maximum number of features to select
            target_column: Name of target column

        Returns:
            DataFrame with selected features
        """
        if method == 'rfe':
            # Use correlation-based selection instead of RFE to avoid memory issues
            self.logger.info(f"Using correlation-based selection instead of RFE for stability")
            result = self._select_by_correlation(df, target_column, max_features)
            selected_features = result.get('selected_features', [])
        elif method == 'kbest':
            result = self.select_features_kbest(df, target_column, max_features, 'f_regression')
            selected_features = result.get('selected_features', [])
        elif method == 'importance':
            # For importance-based selection, use RFE with Random Forest
            self.logger.info(f"Using RFE with Random Forest for importance-based selection")
            result = self.select_features_rfe(df, target_column, max_features, 'random_forest')
            selected_features = result.get('selected_features', [])
        else:
            self.logger.warning(f"Unknown method '{method}', using RFE")
            result = self.select_features_rfe(df, target_column, max_features, 'random_forest')
            selected_features = result.get('selected_features', [])

        # Always include target column and essential columns
        essential_columns = [target_column, 'pdv', 'produto', 'semana', 'data_semana']
        final_features = essential_columns + selected_features

        # Ensure we don't exceed max_features (excluding essential columns)
        if len(final_features) > max_features + len(essential_columns):
            final_features = essential_columns + selected_features[:max_features]

        # Filter DataFrame
        available_features = [col for col in final_features if col in df.columns]
        filtered_df = df[available_features]

        self.logger.info(f"Selected {len(available_features)} features using {method}")
        return filtered_df

    def _select_by_correlation(self, df: pd.DataFrame, target_column: str,
                              max_features: int) -> Dict[str, Any]:
        """
        Select features based on correlation with target.

        Args:
            df: DataFrame with features and target
            target_column: Name of target column
            max_features: Maximum number of features to select

        Returns:
            Dictionary with selection results
        """
        if target_column not in df.columns:
            raise FeatureSelectionError(f"Target column '{target_column}' not found")

        # Prepare data
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_columns = [col for col in numeric_columns if col != target_column]

        # Calculate correlations
        correlations = []
        y = df[target_column].fillna(0)

        for col in feature_columns:
            try:
                X_col = df[col].fillna(0)
                if X_col.std() > 0:  # Avoid constant features
                    corr = abs(np.corrcoef(X_col, y)[0, 1])
                    if not np.isnan(corr):
                        correlations.append((col, corr))
            except Exception as e:
                self.logger.debug(f"Could not calculate correlation for {col}: {e}")
                continue

        # Sort by correlation (highest first)
        correlations.sort(key=lambda x: x[1], reverse=True)

        # Select top features
        n_select = min(max_features, len(correlations))
        selected_features = [col for col, _ in correlations[:n_select]]

        # Create rankings DataFrame
        feature_rankings = pd.DataFrame({
            'feature': [col for col, _ in correlations],
            'correlation': [corr for _, corr in correlations],
            'selected': [i < n_select for i in range(len(correlations))]
        })

        results = {
            'selected_features': selected_features,
            'n_selected': len(selected_features),
            'method': 'correlation',
            'feature_rankings': feature_rankings
        }

        self.logger.info(f"Selected {len(selected_features)} features by correlation")
        return results

    def _setup_logging(self) -> None:
        """Configure logging for feature selection operations."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def analyze_correlations(self, 
                           df: pd.DataFrame, 
                           target_column: str,
                           method: str = 'pearson',
                           threshold: float = 0.8) -> Dict[str, Any]:
        """
        Analyze correlations between features and with target variable.
        
        Args:
            df: DataFrame with features and target
            target_column: Name of target column
            method: Correlation method ('pearson', 'spearman')
            threshold: Threshold for high correlation detection
            
        Returns:
            Dictionary with correlation analysis results
            
        Raises:
            FeatureSelectionError: If correlation analysis fails
        """
        self.logger.info(f"Analyzing correlations using {method} method")
        
        try:
            if target_column not in df.columns:
                raise FeatureSelectionError(f"Target column '{target_column}' not found")
            
            # Select only numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_column not in numeric_columns:
                raise FeatureSelectionError(f"Target column '{target_column}' is not numeric")
            
            df_numeric = df[numeric_columns].copy()
            
            # Calculate correlation matrix
            if method == 'pearson':
                corr_matrix = df_numeric.corr(method='pearson')
            elif method == 'spearman':
                corr_matrix = df_numeric.corr(method='spearman')
            else:
                raise FeatureSelectionError(f"Unsupported correlation method: {method}")
            
            # Target correlations
            target_correlations = corr_matrix[target_column].drop(target_column).abs().sort_values(ascending=False)
            
            # High correlation pairs (excluding target)
            feature_columns = [col for col in numeric_columns if col != target_column]
            high_corr_pairs = []
            
            for i, col1 in enumerate(feature_columns):
                for col2 in feature_columns[i+1:]:
                    corr_value = abs(corr_matrix.loc[col1, col2])
                    if corr_value >= threshold:
                        high_corr_pairs.append({
                            'feature1': col1,
                            'feature2': col2,
                            'correlation': corr_value
                        })
            
            # Sort high correlation pairs by correlation value
            high_corr_pairs = sorted(high_corr_pairs, key=lambda x: x['correlation'], reverse=True)
            
            results = {
                'correlation_matrix': corr_matrix,
                'target_correlations': target_correlations,
                'high_correlation_pairs': high_corr_pairs,
                'method': method,
                'threshold': threshold,
                'num_features': len(feature_columns),
                'num_high_corr_pairs': len(high_corr_pairs)
            }
            
            self.correlation_results_ = results
            self.logger.info(f"Correlation analysis completed. Found {len(high_corr_pairs)} high correlation pairs")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Correlation analysis failed: {str(e)}")
            raise FeatureSelectionError(f"Correlation analysis failed: {str(e)}")
    
    def calculate_feature_importance(self, 
                                   df: pd.DataFrame, 
                                   target_column: str,
                                   method: str = 'random_forest',
                                   **kwargs) -> Dict[str, Any]:
        """
        Calculate feature importance using various methods.
        
        Args:
            df: DataFrame with features and target
            target_column: Name of target column
            method: Importance method ('random_forest', 'mutual_info', 'f_score')
            **kwargs: Additional parameters for the method
            
        Returns:
            Dictionary with feature importance results
            
        Raises:
            FeatureSelectionError: If importance calculation fails
        """
        self.logger.info(f"Calculating feature importance using {method} method")
        
        try:
            if target_column not in df.columns:
                raise FeatureSelectionError(f"Target column '{target_column}' not found")
            
            # Prepare data
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_columns = [col for col in numeric_columns if col != target_column]
            
            X = df[feature_columns].fillna(0)  # Handle missing values
            y = df[target_column].fillna(0)
            
            if method == 'random_forest':
                # Random Forest feature importance
                n_estimators = kwargs.get('n_estimators', 100)
                max_depth = kwargs.get('max_depth', 10)
                
                rf = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=self.random_state,
                    n_jobs=-1
                )
                rf.fit(X, y)
                
                importance_scores = rf.feature_importances_
                
            elif method == 'mutual_info':
                # Mutual information
                importance_scores = mutual_info_regression(
                    X, y, random_state=self.random_state
                )
                
            elif method == 'f_score':
                # F-score (univariate statistical test)
                f_scores, _ = f_regression(X, y)
                importance_scores = f_scores
                
            else:
                raise FeatureSelectionError(f"Unsupported importance method: {method}")
            
            # Create importance DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_columns,
                'importance': importance_scores
            }).sort_values('importance', ascending=False)
            
            # Normalize importance scores to 0-1 range
            importance_df['importance_normalized'] = (
                importance_df['importance'] / importance_df['importance'].max()
            )
            
            results = {
                'importance_scores': importance_df,
                'method': method,
                'num_features': len(feature_columns),
                'top_10_features': importance_df.head(10)['feature'].tolist(),
                'parameters': kwargs
            }
            
            self.importance_results_ = results
            self.logger.info(f"Feature importance calculated for {len(feature_columns)} features")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Feature importance calculation failed: {str(e)}")
            raise FeatureSelectionError(f"Feature importance calculation failed: {str(e)}")
    
    def select_features_rfe(self, 
                           df: pd.DataFrame, 
                           target_column: str,
                           n_features: int = 50,
                           estimator: str = 'random_forest') -> Dict[str, Any]:
        """
        Select features using Recursive Feature Elimination (RFE).
        
        Args:
            df: DataFrame with features and target
            target_column: Name of target column
            n_features: Number of features to select
            estimator: Base estimator ('random_forest', 'linear_regression')
            
        Returns:
            Dictionary with RFE selection results
            
        Raises:
            FeatureSelectionError: If RFE selection fails
        """
        self.logger.info(f"Selecting {n_features} features using RFE with {estimator}")
        
        try:
            if target_column not in df.columns:
                raise FeatureSelectionError(f"Target column '{target_column}' not found")
            
            # Prepare data
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_columns = [col for col in numeric_columns if col != target_column]
            
            X = df[feature_columns].fillna(0)
            y = df[target_column].fillna(0)
            
            # Choose estimator
            if estimator == 'random_forest':
                base_estimator = RandomForestRegressor(
                    n_estimators=20,  # Reduced for stability
                    max_depth=6,      # Reduced for stability
                    random_state=self.random_state,
                    n_jobs=1          # Single thread to avoid memory issues
                )
            elif estimator == 'linear_regression':
                base_estimator = LinearRegression()
            else:
                raise FeatureSelectionError(f"Unsupported estimator: {estimator}")
            
            # Apply RFE with error handling
            try:
                rfe = RFE(
                    estimator=base_estimator,
                    n_features_to_select=min(n_features, len(feature_columns))
                )
                rfe.fit(X, y)
            except Exception as e:
                self.logger.warning(f"RFE failed with {estimator}: {e}")
                # Fallback: select top features by correlation
                if len(feature_columns) > n_features:
                    correlations = []
                    for col in feature_columns:
                        try:
                            corr = abs(np.corrcoef(X[col], y)[0, 1])
                            correlations.append((col, corr))
                        except:
                            correlations.append((col, 0.0))

                    correlations.sort(key=lambda x: x[1], reverse=True)
                    selected_features = [col for col, _ in correlations[:n_features]]
                else:
                    selected_features = feature_columns

                # Return fallback results
                results = {
                    'selected_features': selected_features,
                    'n_selected': len(selected_features),
                    'method': 'correlation_fallback',
                    'feature_rankings': pd.DataFrame({
                        'feature': selected_features,
                        'selected': [True] * len(selected_features),
                        'ranking': list(range(1, len(selected_features) + 1))
                    })
                }

                self.logger.info(f"RFE fallback: selected {len(selected_features)} features by correlation")
                return results
            
            # Get selected features
            selected_features = [feature_columns[i] for i, selected in enumerate(rfe.support_) if selected]
            feature_rankings = pd.DataFrame({
                'feature': feature_columns,
                'selected': rfe.support_,
                'ranking': rfe.ranking_
            }).sort_values('ranking')
            
            results = {
                'selected_features': selected_features,
                'feature_rankings': feature_rankings,
                'n_features_selected': len(selected_features),
                'estimator': estimator,
                'rfe_object': rfe
            }
            
            self.logger.info(f"RFE completed. Selected {len(selected_features)} features")
            
            return results
            
        except Exception as e:
            self.logger.error(f"RFE feature selection failed: {str(e)}")
            raise FeatureSelectionError(f"RFE feature selection failed: {str(e)}")
    
    def select_features_kbest(self, 
                             df: pd.DataFrame, 
                             target_column: str,
                             k: int = 50,
                             score_func: str = 'f_regression') -> Dict[str, Any]:
        """
        Select features using SelectKBest.
        
        Args:
            df: DataFrame with features and target
            target_column: Name of target column
            k: Number of features to select
            score_func: Scoring function ('f_regression', 'mutual_info_regression')
            
        Returns:
            Dictionary with SelectKBest results
            
        Raises:
            FeatureSelectionError: If SelectKBest fails
        """
        self.logger.info(f"Selecting {k} best features using {score_func}")
        
        try:
            if target_column not in df.columns:
                raise FeatureSelectionError(f"Target column '{target_column}' not found")
            
            # Prepare data
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_columns = [col for col in numeric_columns if col != target_column]
            
            X = df[feature_columns].fillna(0)
            y = df[target_column].fillna(0)
            
            # Choose scoring function
            if score_func == 'f_regression':
                scoring_function = f_regression
            elif score_func == 'mutual_info_regression':
                scoring_function = mutual_info_regression
            else:
                raise FeatureSelectionError(f"Unsupported scoring function: {score_func}")
            
            # Apply SelectKBest
            selector = SelectKBest(
                score_func=scoring_function,
                k=min(k, len(feature_columns))
            )
            selector.fit(X, y)
            
            # Get selected features
            selected_features = [feature_columns[i] for i, selected in enumerate(selector.get_support()) if selected]
            
            # Create scores DataFrame
            feature_scores = pd.DataFrame({
                'feature': feature_columns,
                'score': selector.scores_,
                'selected': selector.get_support()
            }).sort_values('score', ascending=False)
            
            results = {
                'selected_features': selected_features,
                'feature_scores': feature_scores,
                'n_features_selected': len(selected_features),
                'score_func': score_func,
                'selector_object': selector
            }
            
            self.logger.info(f"SelectKBest completed. Selected {len(selected_features)} features")
            
            return results
            
        except Exception as e:
            self.logger.error(f"SelectKBest feature selection failed: {str(e)}")
            raise FeatureSelectionError(f"SelectKBest feature selection failed: {str(e)}")
    
    def validate_multicollinearity(self, 
                                  df: pd.DataFrame, 
                                  features: List[str] = None,
                                  vif_threshold: float = 10.0) -> Dict[str, Any]:
        """
        Validate multicollinearity using Variance Inflation Factor (VIF).
        
        Args:
            df: DataFrame with features
            features: List of features to check (if None, use all numeric)
            vif_threshold: VIF threshold for multicollinearity detection
            
        Returns:
            Dictionary with multicollinearity validation results
            
        Raises:
            FeatureSelectionError: If multicollinearity validation fails
        """
        self.logger.info(f"Validating multicollinearity with VIF threshold {vif_threshold}")
        
        try:
            # Select features
            if features is None:
                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                features = numeric_columns
            
            # Prepare data
            X = df[features].fillna(0)
            
            # Remove constant features
            constant_features = []
            for col in X.columns:
                if X[col].nunique() <= 1:
                    constant_features.append(col)
            
            if constant_features:
                self.logger.warning(f"Removing constant features: {constant_features}")
                X = X.drop(columns=constant_features)
                features = [f for f in features if f not in constant_features]
            
            # Calculate VIF for each feature
            vif_data = []
            for i, feature in enumerate(features):
                try:
                    vif_value = variance_inflation_factor(X.values, i)
                    vif_data.append({
                        'feature': feature,
                        'vif': vif_value,
                        'high_multicollinearity': vif_value > vif_threshold
                    })
                except Exception as e:
                    self.logger.warning(f"Could not calculate VIF for {feature}: {e}")
                    vif_data.append({
                        'feature': feature,
                        'vif': np.nan,
                        'high_multicollinearity': False
                    })
            
            # Create VIF DataFrame
            vif_df = pd.DataFrame(vif_data).sort_values('vif', ascending=False)
            
            # Identify problematic features
            high_vif_features = vif_df[vif_df['high_multicollinearity']]['feature'].tolist()
            
            results = {
                'vif_results': vif_df,
                'high_vif_features': high_vif_features,
                'num_high_vif': len(high_vif_features),
                'vif_threshold': vif_threshold,
                'constant_features_removed': constant_features
            }
            
            self.multicollinearity_results_ = results
            self.logger.info(f"Multicollinearity validation completed. Found {len(high_vif_features)} high VIF features")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Multicollinearity validation failed: {str(e)}")
            raise FeatureSelectionError(f"Multicollinearity validation failed: {str(e)}")
    
    def create_feature_selection_pipeline(self, 
                                        df: pd.DataFrame, 
                                        target_column: str,
                                        correlation_threshold: float = 0.95,
                                        vif_threshold: float = 10.0,
                                        importance_method: str = 'random_forest',
                                        selection_method: str = 'rfe',
                                        n_features: int = 50) -> Dict[str, Any]:
        """
        Create comprehensive feature selection pipeline.
        
        Args:
            df: DataFrame with features and target
            target_column: Name of target column
            correlation_threshold: Threshold for correlation-based removal
            vif_threshold: VIF threshold for multicollinearity
            importance_method: Method for importance calculation
            selection_method: Method for feature selection ('rfe', 'kbest')
            n_features: Number of features to select
            
        Returns:
            Dictionary with complete pipeline results
            
        Raises:
            FeatureSelectionError: If pipeline fails
        """
        self.logger.info("Running comprehensive feature selection pipeline")
        
        try:
            pipeline_results = {}
            
            # Step 1: Correlation analysis
            self.logger.info("Step 1: Analyzing correlations")
            corr_results = self.analyze_correlations(
                df, target_column, threshold=correlation_threshold
            )
            pipeline_results['correlation_analysis'] = corr_results
            
            # Step 2: Remove highly correlated features
            high_corr_pairs = corr_results['high_correlation_pairs']
            features_to_remove = set()
            
            for pair in high_corr_pairs:
                # Remove feature with lower target correlation
                target_corr1 = abs(corr_results['target_correlations'].get(pair['feature1'], 0))
                target_corr2 = abs(corr_results['target_correlations'].get(pair['feature2'], 0))
                
                if target_corr1 < target_corr2:
                    features_to_remove.add(pair['feature1'])
                else:
                    features_to_remove.add(pair['feature2'])
            
            # Create filtered dataset
            df_filtered = df.drop(columns=list(features_to_remove))
            pipeline_results['removed_correlated_features'] = list(features_to_remove)
            
            # Step 3: Feature importance
            self.logger.info("Step 2: Calculating feature importance")
            importance_results = self.calculate_feature_importance(
                df_filtered, target_column, method=importance_method
            )
            pipeline_results['importance_analysis'] = importance_results
            
            # Step 4: Feature selection
            self.logger.info("Step 3: Selecting features")
            if selection_method == 'rfe':
                selection_results = self.select_features_rfe(
                    df_filtered, target_column, n_features=n_features
                )
            elif selection_method == 'kbest':
                selection_results = self.select_features_kbest(
                    df_filtered, target_column, k=n_features
                )
            else:
                raise FeatureSelectionError(f"Unsupported selection method: {selection_method}")
            
            pipeline_results['feature_selection'] = selection_results
            
            # Step 5: Multicollinearity validation on selected features
            self.logger.info("Step 4: Validating multicollinearity")
            selected_features = selection_results['selected_features']
            multicollinearity_results = self.validate_multicollinearity(
                df_filtered, features=selected_features, vif_threshold=vif_threshold
            )
            pipeline_results['multicollinearity_validation'] = multicollinearity_results
            
            # Final feature list (remove high VIF features if any)
            high_vif_features = multicollinearity_results['high_vif_features']
            final_features = [f for f in selected_features if f not in high_vif_features]
            
            pipeline_results['final_selected_features'] = final_features
            pipeline_results['pipeline_summary'] = {
                'initial_features': len(df.select_dtypes(include=[np.number]).columns) - 1,
                'after_correlation_filter': len(df_filtered.select_dtypes(include=[np.number]).columns) - 1,
                'after_feature_selection': len(selected_features),
                'final_features': len(final_features),
                'removed_correlated': len(features_to_remove),
                'removed_high_vif': len(high_vif_features)
            }
            
            # Store final results
            self.selected_features_ = final_features
            
            self.logger.info(f"Feature selection pipeline completed. Final features: {len(final_features)}")
            
            return pipeline_results
            
        except Exception as e:
            self.logger.error(f"Feature selection pipeline failed: {str(e)}")
            raise FeatureSelectionError(f"Feature selection pipeline failed: {str(e)}")
    
    def plot_correlation_heatmap(self, 
                                correlation_matrix: pd.DataFrame,
                                figsize: Tuple[int, int] = (12, 10),
                                save_path: Optional[str] = None) -> None:
        """
        Plot correlation heatmap.
        
        Args:
            correlation_matrix: Correlation matrix to plot
            figsize: Figure size
            save_path: Path to save the plot (optional)
        """
        plt.figure(figsize=figsize)
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        # Create heatmap
        sns.heatmap(
            correlation_matrix,
            mask=mask,
            annot=False,
            cmap='coolwarm',
            center=0,
            square=True,
            fmt='.2f'
        )
        
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Correlation heatmap saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, 
                               importance_df: pd.DataFrame,
                               top_n: int = 20,
                               figsize: Tuple[int, int] = (10, 8),
                               save_path: Optional[str] = None) -> None:
        """
        Plot feature importance.
        
        Args:
            importance_df: DataFrame with feature importance scores
            top_n: Number of top features to plot
            figsize: Figure size
            save_path: Path to save the plot (optional)
        """
        plt.figure(figsize=figsize)
        
        # Get top N features
        top_features = importance_df.head(top_n)
        
        # Create horizontal bar plot
        plt.barh(range(len(top_features)), top_features['importance_normalized'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Normalized Importance')
        plt.title(f'Top {top_n} Feature Importance')
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def get_selected_features(self) -> Optional[List[str]]:
        """
        Get the final selected features from the last pipeline run.
        
        Returns:
            List of selected feature names or None if no selection has been run
        """
        return self.selected_features_
    
    def save_selection_results(self, filepath: str) -> None:
        """
        Save feature selection results to file.
        
        Args:
            filepath: Path to save the results
        """
        import pickle
        
        results = {
            'correlation_results': self.correlation_results_,
            'importance_results': self.importance_results_,
            'multicollinearity_results': self.multicollinearity_results_,
            'selected_features': self.selected_features_
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        
        self.logger.info(f"Feature selection results saved to {filepath}")
    
    def load_selection_results(self, filepath: str) -> None:
        """
        Load feature selection results from file.
        
        Args:
            filepath: Path to load the results from
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
        
        self.correlation_results_ = results.get('correlation_results')
        self.importance_results_ = results.get('importance_results')
        self.multicollinearity_results_ = results.get('multicollinearity_results')
        self.selected_features_ = results.get('selected_features')
        
        self.logger.info(f"Feature selection results loaded from {filepath}")