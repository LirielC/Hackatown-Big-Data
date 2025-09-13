"""
Example script demonstrating feature selection functionality.

This script shows how to use the FeatureSelector class for comprehensive
feature selection in the sales forecasting model.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.features.selection import FeatureSelector
from src.features.engineering import FeatureEngineer
from src.data.preprocessing import DataPreprocessor


def create_sample_sales_data():
    """Create sample sales data for demonstration."""
    np.random.seed(42)
    n_samples = 2000
    
    # Create time series data
    dates = pd.date_range('2022-01-01', periods=n_samples, freq='D')
    
    # Create store and product IDs
    n_stores = 50
    n_products = 100
    
    data = []
    for i in range(n_samples):
        date = dates[i]
        store_id = np.random.randint(1, n_stores + 1)
        product_id = np.random.randint(1, n_products + 1)
        
        # Create seasonal patterns
        day_of_year = date.dayofyear
        seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * day_of_year / 365)
        
        # Create store and product effects
        store_effect = np.random.normal(1, 0.2)
        product_effect = np.random.normal(1, 0.3)
        
        # Create base sales with some randomness
        base_sales = 50 * seasonal_factor * store_effect * product_effect
        quantity = max(0, int(base_sales + np.random.normal(0, 10)))
        
        data.append({
            'data': date,
            'pdv': store_id,
            'produto': product_id,
            'quantidade': quantity,
            'categoria': f'categoria_{product_id % 10}',
            'premise': ['c-store', 'g-store', 'liquor'][store_id % 3],
            'zipcode': 10000 + (store_id % 100)
        })
    
    return pd.DataFrame(data)


def demonstrate_feature_selection():
    """Demonstrate feature selection pipeline."""
    print("=== Feature Selection Demonstration ===\n")
    
    # Create sample data
    print("1. Creating sample sales data...")
    df = create_sample_sales_data()
    print(f"Created dataset with {len(df)} records")
    print(f"Columns: {list(df.columns)}")
    
    # Preprocess data
    print("\n2. Preprocessing data...")
    preprocessor = DataPreprocessor()
    
    # Aggregate to weekly data
    df['data_semana'] = df['data'] - pd.to_timedelta(df['data'].dt.dayofweek, unit='d')
    df_weekly = df.groupby(['data_semana', 'pdv', 'produto', 'categoria', 'premise', 'zipcode']).agg({
        'quantidade': 'sum'
    }).reset_index()
    
    print(f"Weekly aggregated data: {len(df_weekly)} records")
    
    # Create features
    print("\n3. Creating features...")
    feature_engineer = FeatureEngineer()
    
    # Create temporal features
    df_features = feature_engineer.create_temporal_features(df_weekly)
    
    # Create product and store features
    df_features = feature_engineer.create_product_features(df_features)
    df_features = feature_engineer.create_store_features(df_features)
    
    # Create lag features (simplified for demo)
    df_features = df_features.sort_values(['pdv', 'produto', 'data_semana'])
    df_features['lag_1'] = df_features.groupby(['pdv', 'produto'])['quantidade'].shift(1)
    df_features['lag_2'] = df_features.groupby(['pdv', 'produto'])['quantidade'].shift(2)
    df_features['rolling_mean_4'] = df_features.groupby(['pdv', 'produto'])['quantidade'].rolling(4).mean().values
    
    # Fill missing values
    df_features = df_features.fillna(0)
    
    print(f"Features created. Total columns: {len(df_features.columns)}")
    
    # Initialize feature selector
    print("\n4. Initializing feature selector...")
    selector = FeatureSelector(random_state=42)
    
    # Run correlation analysis
    print("\n5. Analyzing correlations...")
    corr_results = selector.analyze_correlations(
        df_features, 'quantidade', threshold=0.8
    )
    
    print(f"Found {corr_results['num_high_corr_pairs']} highly correlated feature pairs")
    if corr_results['high_correlation_pairs']:
        print("Top 3 highly correlated pairs:")
        for i, pair in enumerate(corr_results['high_correlation_pairs'][:3]):
            print(f"  {i+1}. {pair['feature1']} - {pair['feature2']}: {pair['correlation']:.3f}")
    
    # Calculate feature importance
    print("\n6. Calculating feature importance...")
    importance_results = selector.calculate_feature_importance(
        df_features, 'quantidade', method='random_forest'
    )
    
    print("Top 10 most important features:")
    top_features = importance_results['top_10_features']
    importance_scores = importance_results['importance_scores']
    
    for i, feature in enumerate(top_features):
        score = importance_scores[importance_scores['feature'] == feature]['importance_normalized'].iloc[0]
        print(f"  {i+1}. {feature}: {score:.3f}")
    
    # Feature selection with RFE
    print("\n7. Selecting features with RFE...")
    rfe_results = selector.select_features_rfe(
        df_features, 'quantidade', n_features=20, estimator='random_forest'
    )
    
    print(f"RFE selected {len(rfe_results['selected_features'])} features:")
    for i, feature in enumerate(rfe_results['selected_features'][:10]):
        print(f"  {i+1}. {feature}")
    if len(rfe_results['selected_features']) > 10:
        print(f"  ... and {len(rfe_results['selected_features']) - 10} more")
    
    # Feature selection with SelectKBest
    print("\n8. Selecting features with SelectKBest...")
    kbest_results = selector.select_features_kbest(
        df_features, 'quantidade', k=15, score_func='f_regression'
    )
    
    print(f"SelectKBest selected {len(kbest_results['selected_features'])} features:")
    for i, feature in enumerate(kbest_results['selected_features'][:10]):
        print(f"  {i+1}. {feature}")
    if len(kbest_results['selected_features']) > 10:
        print(f"  ... and {len(kbest_results['selected_features']) - 10} more")
    
    # Validate multicollinearity
    print("\n9. Validating multicollinearity...")
    selected_features = rfe_results['selected_features']
    multicollinearity_results = selector.validate_multicollinearity(
        df_features, features=selected_features, vif_threshold=10.0
    )
    
    print(f"Multicollinearity analysis on {len(selected_features)} features:")
    print(f"Features with high VIF (>10): {len(multicollinearity_results['high_vif_features'])}")
    
    if multicollinearity_results['high_vif_features']:
        print("High VIF features:")
        vif_results = multicollinearity_results['vif_results']
        high_vif = vif_results[vif_results['high_multicollinearity']]
        for _, row in high_vif.head(5).iterrows():
            print(f"  {row['feature']}: VIF = {row['vif']:.2f}")
    
    # Run comprehensive pipeline
    print("\n10. Running comprehensive feature selection pipeline...")
    pipeline_results = selector.create_feature_selection_pipeline(
        df_features,
        'quantidade',
        correlation_threshold=0.9,
        vif_threshold=10.0,
        importance_method='random_forest',
        selection_method='rfe',
        n_features=15
    )
    
    summary = pipeline_results['pipeline_summary']
    print("Pipeline Summary:")
    print(f"  Initial features: {summary['initial_features']}")
    print(f"  After correlation filter: {summary['after_correlation_filter']}")
    print(f"  After feature selection: {summary['after_feature_selection']}")
    print(f"  Final features: {summary['final_features']}")
    print(f"  Removed (correlation): {summary['removed_correlated']}")
    print(f"  Removed (high VIF): {summary['removed_high_vif']}")
    
    print(f"\nFinal selected features ({len(pipeline_results['final_selected_features'])}):")
    for i, feature in enumerate(pipeline_results['final_selected_features']):
        print(f"  {i+1}. {feature}")
    
    # Create visualizations
    print("\n11. Creating visualizations...")
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    importance_df = importance_results['importance_scores']
    top_20 = importance_df.head(20)
    
    plt.barh(range(len(top_20)), top_20['importance_normalized'])
    plt.yticks(range(len(top_20)), top_20['feature'])
    plt.xlabel('Normalized Importance')
    plt.title('Top 20 Feature Importance (Random Forest)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    # Save plot
    os.makedirs('plots/feature_selection', exist_ok=True)
    plt.savefig('plots/feature_selection/feature_importance.png', dpi=300, bbox_inches='tight')
    print("Feature importance plot saved to plots/feature_selection/feature_importance.png")
    plt.close()
    
    # Plot correlation heatmap for selected features
    final_features = pipeline_results['final_selected_features'] + ['quantidade']
    corr_matrix_final = df_features[final_features].corr()
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix_final, dtype=bool))
    
    import seaborn as sns
    sns.heatmap(
        corr_matrix_final,
        mask=mask,
        annot=True,
        cmap='coolwarm',
        center=0,
        square=True,
        fmt='.2f'
    )
    plt.title('Correlation Heatmap - Final Selected Features')
    plt.tight_layout()
    plt.savefig('plots/feature_selection/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("Correlation heatmap saved to plots/feature_selection/correlation_heatmap.png")
    plt.close()
    
    # Save selection results
    print("\n12. Saving selection results...")
    selector.save_selection_results('data/processed/feature_selection_results.pkl')
    print("Feature selection results saved to data/processed/feature_selection_results.pkl")
    
    print("\n=== Feature Selection Demonstration Complete ===")
    
    return pipeline_results


def compare_selection_methods():
    """Compare different feature selection methods."""
    print("\n=== Comparing Feature Selection Methods ===\n")
    
    # Create sample data
    df = create_sample_sales_data()
    
    # Simple feature engineering for comparison
    df['data_semana'] = df['data'] - pd.to_timedelta(df['data'].dt.dayofweek, unit='d')
    df_weekly = df.groupby(['data_semana', 'pdv', 'produto']).agg({
        'quantidade': 'sum'
    }).reset_index()
    
    # Add simple features
    df_weekly['semana_ano'] = df_weekly['data_semana'].dt.isocalendar().week
    df_weekly['mes'] = df_weekly['data_semana'].dt.month
    df_weekly['trimestre'] = df_weekly['data_semana'].dt.quarter
    
    # Add lag features
    df_weekly = df_weekly.sort_values(['pdv', 'produto', 'data_semana'])
    for lag in [1, 2, 4]:
        df_weekly[f'lag_{lag}'] = df_weekly.groupby(['pdv', 'produto'])['quantidade'].shift(lag)
    
    df_weekly = df_weekly.fillna(0)
    
    selector = FeatureSelector(random_state=42)
    
    methods = [
        ('RFE + Random Forest', 'rfe', 'random_forest'),
        ('SelectKBest + F-regression', 'kbest', 'f_regression'),
        ('SelectKBest + Mutual Info', 'kbest', 'mutual_info_regression')
    ]
    
    results_comparison = {}
    
    for method_name, selection_method, score_method in methods:
        print(f"Testing {method_name}...")
        
        if selection_method == 'rfe':
            results = selector.select_features_rfe(
                df_weekly, 'quantidade', n_features=5, estimator='random_forest'
            )
        else:
            results = selector.select_features_kbest(
                df_weekly, 'quantidade', k=5, score_func=score_method
            )
        
        selected_features = results['selected_features']
        results_comparison[method_name] = selected_features
        
        print(f"  Selected features: {selected_features}")
    
    # Find common features
    all_features = set()
    for features in results_comparison.values():
        all_features.update(features)
    
    print(f"\nFeature selection comparison:")
    print(f"Total unique features selected: {len(all_features)}")
    
    # Count how many methods selected each feature
    feature_counts = {}
    for feature in all_features:
        count = sum(1 for features in results_comparison.values() if feature in features)
        feature_counts[feature] = count
    
    print("\nFeatures selected by multiple methods:")
    for feature, count in sorted(feature_counts.items(), key=lambda x: x[1], reverse=True):
        if count > 1:
            print(f"  {feature}: selected by {count}/{len(methods)} methods")
    
    return results_comparison


if __name__ == '__main__':
    # Run demonstrations
    try:
        # Main demonstration
        pipeline_results = demonstrate_feature_selection()
        
        # Method comparison
        comparison_results = compare_selection_methods()
        
        print("\n✅ Feature selection demonstration completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {str(e)}")
        import traceback
        traceback.print_exc()