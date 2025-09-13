"""
Utilities for Exploratory Data Analysis (EDA).

This module provides functions to perform automated EDA on the hackathon dataset,
generating insights about data quality, temporal patterns, and distributions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging


class EDAAnalyzer:
    """
    Automated Exploratory Data Analysis for hackathon forecast data.
    
    This class provides methods to analyze data quality, temporal patterns,
    categorical distributions, and generate insights for feature engineering.
    """
    
    def __init__(self, save_plots: bool = True, plot_dir: str = "plots"):
        """
        Initialize EDA analyzer.
        
        Args:
            save_plots: Whether to save plots to disk
            plot_dir: Directory to save plots
        """
        self.save_plots = save_plots
        self.plot_dir = Path(plot_dir)
        self.plot_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # Configure plotting
        plt.style.use('seaborn-v0_8')
        sns.set_palette('husl')
        warnings.filterwarnings('ignore')
    
    def _setup_logging(self) -> None:
        """Configure logging for EDA operations."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def analyze_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze data quality including missing values, duplicates, and data types.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with data quality metrics
        """
        self.logger.info("Analyzing data quality...")
        
        quality_report = {
            'basic_info': {
                'total_records': len(df),
                'total_columns': len(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
            },
            'missing_values': {},
            'duplicates': {},
            'data_types': {},
            'outliers': {}
        }
        
        # Missing values analysis
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        
        quality_report['missing_values'] = {
            'total_missing': missing_data.sum(),
            'columns_with_missing': (missing_data > 0).sum(),
            'missing_by_column': missing_data[missing_data > 0].to_dict(),
            'missing_percent_by_column': missing_percent[missing_percent > 0].to_dict()
        }
        
        # Duplicates analysis
        total_duplicates = df.duplicated().sum()
        quality_report['duplicates'] = {
            'total_duplicates': total_duplicates,
            'duplicate_percentage': (total_duplicates / len(df)) * 100
        }
        
        # Data types analysis
        quality_report['data_types'] = {
            'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(df.select_dtypes(include=['object']).columns),
            'datetime_columns': list(df.select_dtypes(include=['datetime64']).columns)
        }
        
        return quality_report
    
    def analyze_temporal_patterns(self, df: pd.DataFrame, 
                                date_col: Optional[str] = None,
                                quantity_col: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze temporal patterns and seasonality.
        
        Args:
            df: DataFrame to analyze
            date_col: Name of date column (auto-detected if None)
            quantity_col: Name of quantity column (auto-detected if None)
            
        Returns:
            Dictionary with temporal analysis results
        """
        self.logger.info("Analyzing temporal patterns...")
        
        # Auto-detect date column
        if date_col is None:
            date_candidates = [col for col in df.columns if any(keyword in col.lower() 
                             for keyword in ['date', 'data', 'time', 'timestamp'])]
            if date_candidates:
                date_col = date_candidates[0]
        
        # Auto-detect quantity column
        if quantity_col is None:
            qty_candidates = [col for col in df.columns if any(keyword in col.lower() 
                            for keyword in ['quantidade', 'quantity', 'sales', 'value'])]
            for col in qty_candidates:
                if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    quantity_col = col
                    break
        
        temporal_report = {
            'date_column': date_col,
            'quantity_column': quantity_col,
            'temporal_features': {},
            'seasonality': {},
            'trends': {}
        }
        
        if date_col and date_col in df.columns:
            df_temp = df.copy()
            df_temp[date_col] = pd.to_datetime(df_temp[date_col])
            
            # Filter for 2022 data
            df_temp = df_temp[df_temp[date_col].dt.year == 2022]
            
            # Create temporal features
            df_temp['year'] = df_temp[date_col].dt.year
            df_temp['month'] = df_temp[date_col].dt.month
            df_temp['week'] = df_temp[date_col].dt.isocalendar().week
            df_temp['dayofweek'] = df_temp[date_col].dt.dayofweek
            df_temp['quarter'] = df_temp[date_col].dt.quarter
            
            temporal_report['temporal_features'] = {
                'date_range': f"{df_temp[date_col].min()} to {df_temp[date_col].max()}",
                'total_weeks': df_temp['week'].nunique(),
                'total_months': df_temp['month'].nunique()
            }
            
            if quantity_col and quantity_col in df_temp.columns:
                # Monthly analysis
                monthly_stats = df_temp.groupby('month')[quantity_col].agg([
                    'sum', 'mean', 'count', 'std'
                ]).round(2)
                
                # Weekly analysis
                weekly_stats = df_temp.groupby('week')[quantity_col].agg([
                    'sum', 'mean', 'count'
                ]).round(2)
                
                # Day of week analysis
                dow_stats = df_temp.groupby('dayofweek')[quantity_col].sum()
                
                temporal_report['seasonality'] = {
                    'monthly_variation_cv': (monthly_stats['sum'].std() / monthly_stats['sum'].mean()),
                    'weekly_variation_cv': (weekly_stats['sum'].std() / weekly_stats['sum'].mean()),
                    'strongest_month': monthly_stats['sum'].idxmax(),
                    'weakest_month': monthly_stats['sum'].idxmin(),
                    'strongest_dow': dow_stats.idxmax(),
                    'weakest_dow': dow_stats.idxmin()
                }
                
                # Generate temporal plots
                if self.save_plots:
                    self._plot_temporal_patterns(df_temp, date_col, quantity_col)
        
        return temporal_report
    
    def analyze_categorical_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze distributions of categorical variables.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with categorical analysis results
        """
        self.logger.info("Analyzing categorical distributions...")
        
        categorical_report = {
            'product_analysis': {},
            'store_analysis': {},
            'general_categorical': {}
        }
        
        # Identify product-related columns
        product_cols = [col for col in df.columns if any(keyword in col.lower() 
                       for keyword in ['categoria', 'category', 'produto', 'product'])]
        
        # Identify store-related columns
        store_cols = [col for col in df.columns if any(keyword in col.lower() 
                     for keyword in ['pdv', 'store', 'loja', 'premise', 'tipo'])]
        
        # Analyze product columns
        for col in product_cols:
            if df[col].dtype == 'object' and df[col].nunique() < 100:
                value_counts = df[col].value_counts()
                categorical_report['product_analysis'][col] = {
                    'unique_count': df[col].nunique(),
                    'top_categories': value_counts.head(10).to_dict(),
                    'category_distribution': value_counts.describe().to_dict()
                }
        
        # Analyze store columns
        for col in store_cols:
            if df[col].dtype == 'object' and df[col].nunique() < 50:
                value_counts = df[col].value_counts()
                categorical_report['store_analysis'][col] = {
                    'unique_count': df[col].nunique(),
                    'distribution': value_counts.to_dict()
                }
        
        return categorical_report
    
    def detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect outliers in numeric columns using IQR method.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with outlier analysis results
        """
        self.logger.info("Detecting outliers...")
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        outlier_report = {}
        
        for col in numeric_columns:
            if df[col].nunique() > 10:  # Skip columns with few unique values
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_low = (df[col] < lower_bound).sum()
                outliers_high = (df[col] > upper_bound).sum()
                total_outliers = outliers_low + outliers_high
                
                outlier_report[col] = {
                    'total_outliers': total_outliers,
                    'outlier_percentage': (total_outliers / len(df)) * 100,
                    'outliers_low': outliers_low,
                    'outliers_high': outliers_high,
                    'Q1': Q1,
                    'Q3': Q3,
                    'IQR': IQR,
                    'bounds': [lower_bound, upper_bound]
                }
        
        return outlier_report
    
    def generate_correlation_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate correlation analysis for numeric variables.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with correlation analysis results
        """
        self.logger.info("Analyzing correlations...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if df[col].nunique() > 1 and df[col].std() > 0]
        
        correlation_report = {
            'correlation_matrix': {},
            'high_correlations': [],
            'multicollinearity_risk': []
        }
        
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            correlation_report['correlation_matrix'] = corr_matrix.to_dict()
            
            # Find high correlations
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    var1 = corr_matrix.columns[i]
                    var2 = corr_matrix.columns[j]
                    corr_value = corr_matrix.iloc[i, j]
                    
                    if abs(corr_value) > 0.7:
                        high_corr_pairs.append({
                            'variable_1': var1,
                            'variable_2': var2,
                            'correlation': corr_value
                        })
            
            correlation_report['high_correlations'] = high_corr_pairs
            correlation_report['multicollinearity_risk'] = [
                pair for pair in high_corr_pairs if abs(pair['correlation']) > 0.8
            ]
            
            # Generate correlation plot
            if self.save_plots:
                self._plot_correlation_matrix(corr_matrix)
        
        return correlation_report
    
    def generate_insights_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive insights summary for feature engineering.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with insights and recommendations
        """
        self.logger.info("Generating insights summary...")
        
        # Run all analyses
        quality_report = self.analyze_data_quality(df)
        temporal_report = self.analyze_temporal_patterns(df)
        categorical_report = self.analyze_categorical_distributions(df)
        outlier_report = self.detect_outliers(df)
        correlation_report = self.generate_correlation_analysis(df)
        
        # Generate insights
        insights = []
        recommendations = []
        
        # Data quality insights
        missing_pct = (quality_report['missing_values']['total_missing'] / 
                      (len(df) * len(df.columns))) * 100
        if missing_pct > 10:
            insights.append(f"HIGH_MISSING: {missing_pct:.1f}% missing data - imputation strategy needed")
            recommendations.append("Implement robust missing value imputation strategy")
        elif missing_pct > 0:
            insights.append(f"LOW_MISSING: {missing_pct:.1f}% missing data - manageable")
        else:
            insights.append("NO_MISSING: No missing values detected")
        
        # Temporal insights
        if temporal_report['date_column']:
            insights.append("TEMPORAL_DATA: Date column identified - temporal features possible")
            recommendations.append("Create temporal features: week, month, quarter, seasonality")
            
            if 'seasonality' in temporal_report and temporal_report['seasonality']:
                weekly_cv = temporal_report['seasonality'].get('weekly_variation_cv', 0)
                if weekly_cv > 0.3:
                    insights.append(f"HIGH_SEASONALITY: Weekly CV={weekly_cv:.3f} - strong seasonal patterns")
                    recommendations.append("Implement lag features and rolling statistics")
        else:
            insights.append("NO_TEMPORAL: No date column identified")
        
        # Categorical insights
        product_count = len(categorical_report['product_analysis'])
        store_count = len(categorical_report['store_analysis'])
        
        if product_count > 0:
            insights.append(f"PRODUCT_CATEGORIES: {product_count} product columns identified")
            recommendations.append("Apply appropriate encoding for product categories")
        
        if store_count > 0:
            insights.append(f"STORE_CATEGORIES: {store_count} store columns identified")
            recommendations.append("Create store-type specific features")
        
        # Outlier insights
        high_outlier_cols = [col for col, data in outlier_report.items() 
                           if data['outlier_percentage'] > 5]
        if high_outlier_cols:
            insights.append(f"HIGH_OUTLIERS: {len(high_outlier_cols)} columns with >5% outliers")
            recommendations.append("Apply outlier treatment (winsorization or capping)")
        
        # Correlation insights
        multicollinearity_risk = len(correlation_report.get('multicollinearity_risk', []))
        if multicollinearity_risk > 0:
            insights.append(f"MULTICOLLINEARITY: {multicollinearity_risk} high correlation pairs")
            recommendations.append("Consider feature selection to reduce multicollinearity")
        
        summary = {
            'dataset_overview': {
                'records': len(df),
                'columns': len(df.columns),
                'memory_mb': quality_report['basic_info']['memory_usage_mb']
            },
            'insights': insights,
            'recommendations': recommendations,
            'detailed_reports': {
                'quality': quality_report,
                'temporal': temporal_report,
                'categorical': categorical_report,
                'outliers': outlier_report,
                'correlations': correlation_report
            }
        }
        
        return summary
    
    def _plot_temporal_patterns(self, df: pd.DataFrame, date_col: str, qty_col: str) -> None:
        """Generate and save temporal pattern plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Monthly sales
        monthly_sales = df.groupby('month')[qty_col].sum()
        axes[0,0].bar(monthly_sales.index, monthly_sales.values)
        axes[0,0].set_title('Monthly Sales Distribution')
        axes[0,0].set_xlabel('Month')
        axes[0,0].set_ylabel('Total Quantity')
        
        # Weekly trend
        weekly_sales = df.groupby('week')[qty_col].sum()
        axes[0,1].plot(weekly_sales.index, weekly_sales.values, marker='o')
        axes[0,1].set_title('Weekly Sales Trend')
        axes[0,1].set_xlabel('Week')
        axes[0,1].set_ylabel('Total Quantity')
        
        # Day of week pattern
        dow_sales = df.groupby('dayofweek')[qty_col].sum()
        dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        axes[1,0].bar(dow_labels, dow_sales.values)
        axes[1,0].set_title('Day of Week Pattern')
        axes[1,0].set_xlabel('Day of Week')
        axes[1,0].set_ylabel('Total Quantity')
        
        # Quantity distribution
        axes[1,1].hist(df[qty_col], bins=50, alpha=0.7, edgecolor='black')
        axes[1,1].set_title('Quantity Distribution')
        axes[1,1].set_xlabel('Quantity')
        axes[1,1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'temporal_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_matrix(self, corr_matrix: pd.DataFrame) -> None:
        """Generate and save correlation matrix heatmap."""
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Correlation Matrix - Numeric Variables')
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()