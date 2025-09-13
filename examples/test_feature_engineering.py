"""
Example script to test feature engineering functionality.

This script demonstrates how to use the FeatureEngineer class to create
comprehensive features for sales forecasting.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from features.engineering import FeatureEngineer


def create_sample_data():
    """Create sample sales data for testing."""
    print("Creating sample sales data...")
    
    # Create date range for 2022
    dates = pd.date_range('2022-01-01', '2022-12-31', freq='W')
    
    # Create sample PDVs and products
    pdvs = [f'PDV{str(i).zfill(3)}' for i in range(1, 11)]  # 10 stores
    produtos = [f'PROD{str(i).zfill(3)}' for i in range(1, 21)]  # 20 products
    
    # Generate all combinations
    data = []
    for date in dates:
        for pdv in pdvs:
            for produto in produtos:
                # Add some randomness and seasonality
                base_quantity = np.random.randint(5, 50)
                
                # Add seasonality (higher sales in December)
                if date.month == 12:
                    base_quantity *= 1.5
                elif date.month in [6, 7]:  # Winter season in Brazil
                    base_quantity *= 0.8
                
                # Add some trend
                week_of_year = date.isocalendar().week
                trend_factor = 1 + (week_of_year / 52) * 0.2
                
                quantity = int(base_quantity * trend_factor)
                
                data.append({
                    'data_semana': date,
                    'pdv': pdv,
                    'produto': produto,
                    'quantidade': quantity
                })
    
    df = pd.DataFrame(data)
    
    # Add some product information
    categories = ['Bebidas', 'Alimentos', 'Limpeza', 'Higiene', 'Outros']
    brands = ['MarcaA', 'MarcaB', 'MarcaC', 'MarcaD', 'MarcaE']
    
    product_info = []
    for produto in produtos:
        product_info.append({
            'produto': produto,
            'categoria': np.random.choice(categories),
            'marca': np.random.choice(brands),
            'preco_unitario': np.random.uniform(5.0, 50.0)
        })
    
    product_df = pd.DataFrame(product_info)
    df = df.merge(product_df, on='produto', how='left')
    
    # Add some store information
    store_types = ['c-store', 'g-store', 'liquor']
    store_categories = ['Conveniencia', 'Supermercado', 'Especializada']
    
    store_info = []
    for pdv in pdvs:
        store_info.append({
            'pdv': pdv,
            'premise': np.random.choice(store_types),
            'categoria_pdv': np.random.choice(store_categories),
            'zipcode': np.random.randint(10000, 99999)
        })
    
    store_df = pd.DataFrame(store_info)
    df = df.merge(store_df, on='pdv', how='left')
    
    print(f"Created sample data with {len(df)} records")
    print(f"Date range: {df['data_semana'].min()} to {df['data_semana'].max()}")
    print(f"Number of PDVs: {df['pdv'].nunique()}")
    print(f"Number of products: {df['produto'].nunique()}")
    
    return df


def test_temporal_features():
    """Test temporal feature creation."""
    print("\n" + "="*50)
    print("TESTING TEMPORAL FEATURES")
    print("="*50)
    
    # Create sample data
    sample_data = create_sample_data().head(1000)  # Use subset for faster testing
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer(country_code='BR')
    
    # Create temporal features
    print("Creating temporal features...")
    df_temporal = feature_engineer.create_temporal_features(sample_data)
    
    print(f"Original columns: {len(sample_data.columns)}")
    print(f"With temporal features: {len(df_temporal.columns)}")
    print(f"Added features: {len(df_temporal.columns) - len(sample_data.columns)}")
    
    # Show some temporal features
    temporal_features = [col for col in df_temporal.columns if col not in sample_data.columns]
    print(f"\nSample temporal features: {temporal_features[:10]}")
    
    # Show sample values
    print("\nSample temporal feature values:")
    sample_row = df_temporal.iloc[0]
    for feature in temporal_features[:5]:
        print(f"  {feature}: {sample_row[feature]}")
    
    return df_temporal


def test_product_features():
    """Test product feature creation."""
    print("\n" + "="*50)
    print("TESTING PRODUCT FEATURES")
    print("="*50)
    
    # Create sample data
    sample_data = create_sample_data().head(1000)
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer()
    
    # Create product features
    print("Creating product features...")
    df_product = feature_engineer.create_product_features(sample_data)
    
    print(f"Original columns: {len(sample_data.columns)}")
    print(f"With product features: {len(df_product.columns)}")
    print(f"Added features: {len(df_product.columns) - len(sample_data.columns)}")
    
    # Show some product features
    product_features = [col for col in df_product.columns if col not in sample_data.columns]
    print(f"\nSample product features: {product_features[:10]}")
    
    # Show sample values
    print("\nSample product feature values:")
    sample_row = df_product.iloc[0]
    for feature in product_features[:5]:
        print(f"  {feature}: {sample_row[feature]}")
    
    return df_product


def test_store_features():
    """Test store feature creation."""
    print("\n" + "="*50)
    print("TESTING STORE FEATURES")
    print("="*50)
    
    # Create sample data
    sample_data = create_sample_data().head(1000)
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer()
    
    # Create store features
    print("Creating store features...")
    df_store = feature_engineer.create_store_features(sample_data)
    
    print(f"Original columns: {len(sample_data.columns)}")
    print(f"With store features: {len(df_store.columns)}")
    print(f"Added features: {len(df_store.columns) - len(sample_data.columns)}")
    
    # Show some store features
    store_features = [col for col in df_store.columns if col not in sample_data.columns]
    print(f"\nSample store features: {store_features[:10]}")
    
    # Show sample values
    print("\nSample store feature values:")
    sample_row = df_store.iloc[0]
    for feature in store_features[:5]:
        print(f"  {feature}: {sample_row[feature]}")
    
    return df_store


def test_lag_and_rolling_features():
    """Test lag and rolling feature creation."""
    print("\n" + "="*50)
    print("TESTING LAG AND ROLLING FEATURES")
    print("="*50)
    
    # Create sample data (need more time periods for lag features)
    sample_data = create_sample_data()
    
    # Use subset of PDVs and products for faster processing
    sample_pdvs = sample_data['pdv'].unique()[:3]
    sample_produtos = sample_data['produto'].unique()[:5]
    
    sample_data = sample_data[
        (sample_data['pdv'].isin(sample_pdvs)) & 
        (sample_data['produto'].isin(sample_produtos))
    ]
    
    print(f"Using subset with {len(sample_data)} records for lag features")
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer()
    
    # Create lag and rolling features
    print("Creating lag and rolling features...")
    df_lag_rolling = feature_engineer.create_all_lag_and_rolling_features(sample_data)
    
    print(f"Original columns: {len(sample_data.columns)}")
    print(f"With lag/rolling features: {len(df_lag_rolling.columns)}")
    print(f"Added features: {len(df_lag_rolling.columns) - len(sample_data.columns)}")
    
    # Show some lag and rolling features
    lag_rolling_features = [col for col in df_lag_rolling.columns if col not in sample_data.columns]
    print(f"\nSample lag/rolling features: {lag_rolling_features[:15]}")
    
    # Show sample values for one PDV/product combination
    sample_subset = df_lag_rolling[
        (df_lag_rolling['pdv'] == sample_pdvs[0]) & 
        (df_lag_rolling['produto'] == sample_produtos[0])
    ].head(10)
    
    print(f"\nSample values for {sample_pdvs[0]}/{sample_produtos[0]}:")
    print("Week | Quantity | Lag1 | Lag2 | Rolling Mean 4")
    for _, row in sample_subset.iterrows():
        week = row['data_semana'].strftime('%Y-%m-%d')
        qty = row['quantidade']
        lag1 = row.get('quantidade_lag_1', 'N/A')
        lag2 = row.get('quantidade_lag_2', 'N/A')
        rolling_mean = row.get('quantidade_rolling_mean_4', 'N/A')
        print(f"{week} | {qty:8.0f} | {lag1:4} | {lag2:4} | {rolling_mean:12.2f}")
    
    return df_lag_rolling


def test_all_features():
    """Test creation of all features together."""
    print("\n" + "="*50)
    print("TESTING ALL FEATURES TOGETHER")
    print("="*50)
    
    # Create sample data
    sample_data = create_sample_data()
    
    # Use smaller subset for comprehensive feature creation
    sample_pdvs = sample_data['pdv'].unique()[:2]
    sample_produtos = sample_data['produto'].unique()[:3]
    
    sample_data = sample_data[
        (sample_data['pdv'].isin(sample_pdvs)) & 
        (sample_data['produto'].isin(sample_produtos))
    ]
    
    print(f"Using subset with {len(sample_data)} records for all features")
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer()
    
    # Create all features
    print("Creating all features...")
    df_all_features = feature_engineer.create_all_features(sample_data)
    
    print(f"Original columns: {len(sample_data.columns)}")
    print(f"With all features: {len(df_all_features.columns)}")
    print(f"Added features: {len(df_all_features.columns) - len(sample_data.columns)}")
    
    # Get feature summary
    feature_summary = feature_engineer.get_feature_summary(df_all_features)
    
    print("\nFeature Summary:")
    print(f"Total features: {feature_summary['total_features']}")
    print("\nFeatures by type:")
    for feature_type, count in feature_summary['feature_types'].items():
        print(f"  {feature_type}: {count}")
    
    print(f"\nMissing values: {len(feature_summary['missing_values'])} columns have missing values")
    if feature_summary['missing_values']:
        print("Columns with missing values:")
        for col, count in list(feature_summary['missing_values'].items())[:5]:
            print(f"  {col}: {count}")
    
    print("\nData types:")
    for dtype, count in feature_summary['data_types'].items():
        print(f"  {dtype}: {count}")
    
    return df_all_features


def main():
    """Run all feature engineering tests."""
    print("FEATURE ENGINEERING TEST SUITE")
    print("="*60)
    
    try:
        # Test individual feature types
        df_temporal = test_temporal_features()
        df_product = test_product_features()
        df_store = test_store_features()
        df_lag_rolling = test_lag_and_rolling_features()
        
        # Test all features together
        df_all = test_all_features()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print(f"\nFinal dataset shape: {df_all.shape}")
        print(f"Memory usage: {df_all.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()