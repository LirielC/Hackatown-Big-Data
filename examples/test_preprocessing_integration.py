"""
Integration test for data preprocessing module.

This script demonstrates the preprocessing module working with sample data
similar to the hackathon dataset structure.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.ingestion import DataIngestion
from data.preprocessing import DataPreprocessor


def create_sample_data():
    """Create sample data similar to hackathon format."""
    
    # Create sample transaction data
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', '2022-12-31', freq='D')
    
    transactions = []
    for date in dates:
        # Generate random transactions for each day
        n_transactions = np.random.randint(50, 200)
        for _ in range(n_transactions):
            transactions.append({
                'internal_store_id': f"store_{np.random.randint(1, 100)}",
                'internal_product_id': f"product_{np.random.randint(1, 500)}",
                'transaction_date': date,
                'quantity': np.random.randint(1, 20),
                'gross_value': np.random.uniform(5, 100),
                'net_value': np.random.uniform(4, 95),
                'distributor_id': str(np.random.randint(1, 10))
            })
    
    transaction_df = pd.DataFrame(transactions)
    
    # Create sample store data
    stores = []
    for i in range(1, 101):
        stores.append({
            'pdv': f"store_{i}",
            'premise': np.random.choice(['On Premise', 'Off Premise']),
            'categoria_pdv': np.random.choice(['Restaurant', 'Bar', 'Convenience', 'Liquor Store']),
            'zipcode': np.random.randint(10000, 99999)
        })
    
    store_df = pd.DataFrame(stores)
    
    # Create sample product data
    products = []
    categories = ['Beverages', 'Food', 'Snacks', 'Alcohol', 'Tobacco']
    brands = ['Brand A', 'Brand B', 'Brand C', 'Brand D', 'Brand E']
    
    for i in range(1, 501):
        products.append({
            'produto': f"product_{i}",
            'categoria': np.random.choice(categories),
            'marca': np.random.choice(brands),
            'preco_unitario': np.random.uniform(1, 50)
        })
    
    product_df = pd.DataFrame(products)
    
    return transaction_df, store_df, product_df


def main():
    """Run preprocessing integration test."""
    print("=== Data Preprocessing Integration Test ===\n")
    
    # Create sample data
    print("1. Creating sample data...")
    transactions, stores, products = create_sample_data()
    
    print(f"   - Transactions: {len(transactions)} records")
    print(f"   - Stores: {len(stores)} records")
    print(f"   - Products: {len(products)} records")
    
    # Initialize preprocessor
    print("\n2. Initializing preprocessor...")
    preprocessor = DataPreprocessor(use_polars=False)
    
    # Clean transaction data
    print("\n3. Cleaning transaction data...")
    clean_transactions = preprocessor.clean_transactions(transactions)
    print(f"   - Original records: {len(transactions)}")
    print(f"   - Clean records: {len(clean_transactions)}")
    print(f"   - Records removed: {len(transactions) - len(clean_transactions)}")
    
    # Aggregate to weekly data
    print("\n4. Aggregating to weekly sales...")
    weekly_sales = preprocessor.aggregate_weekly_sales(clean_transactions)
    print(f"   - Daily records: {len(clean_transactions)}")
    print(f"   - Weekly records: {len(weekly_sales)}")
    print(f"   - Aggregation ratio: {len(clean_transactions) / len(weekly_sales):.1f}:1")
    
    # Show sample of weekly data
    print("\n   Sample weekly data:")
    print(weekly_sales.head())
    
    # Merge with master data
    print("\n5. Merging with master data...")
    merged_data = preprocessor.merge_master_data(
        weekly_sales, 
        products=products, 
        stores=stores
    )
    
    print(f"   - Records after merge: {len(merged_data)}")
    print(f"   - Columns after merge: {len(merged_data.columns)}")
    
    # Show sample of merged data
    print("\n   Sample merged data columns:")
    print(list(merged_data.columns))
    
    # Create time features
    print("\n6. Creating time features...")
    final_data = preprocessor.create_time_features(merged_data)
    
    print(f"   - Final columns: {len(final_data.columns)}")
    
    # Show time features
    time_features = [col for col in final_data.columns if col in 
                    ['mes', 'trimestre', 'dia_semana', 'is_weekend', 'is_inicio_mes']]
    print(f"   - Time features added: {time_features}")
    
    # Validate final data
    print("\n7. Validating processed data...")
    validation = preprocessor.validate_processed_data(final_data)
    
    print(f"   - Data is valid: {validation['is_valid']}")
    print(f"   - Total records: {validation['summary']['total_records']}")
    print(f"   - Total columns: {validation['summary']['total_columns']}")
    
    if validation['warnings']:
        print(f"   - Warnings: {len(validation['warnings'])}")
        for warning in validation['warnings'][:3]:  # Show first 3 warnings
            print(f"     * {warning}")
    
    # Generate preprocessing summary
    print("\n8. Preprocessing summary...")
    summary = preprocessor.get_preprocessing_summary(transactions, final_data)
    
    print(f"   - Original records: {summary['original_records']}")
    print(f"   - Final records: {summary['processed_records']}")
    print(f"   - Records removed: {summary['records_removed']} ({summary['removal_percentage']:.1f}%)")
    print(f"   - Columns added: {summary['columns_added']}")
    print(f"   - Memory usage: {summary['processed_memory_mb']:.1f} MB")
    
    # Show data quality metrics
    print("\n9. Data quality metrics...")
    if 'quantidade' in final_data.columns:
        print(f"   - Total quantity: {final_data['quantidade'].sum():,.0f}")
        print(f"   - Average quantity per transaction: {final_data['quantidade'].mean():.1f}")
        print(f"   - Quantity range: {final_data['quantidade'].min()} - {final_data['quantidade'].max()}")
    
    # Show temporal coverage
    if 'data_semana' in final_data.columns:
        print(f"   - Date range: {final_data['data_semana'].min()} to {final_data['data_semana'].max()}")
        print(f"   - Unique weeks: {final_data['semana'].nunique()}")
        print(f"   - Unique stores: {final_data['pdv'].nunique()}")
        print(f"   - Unique products: {final_data['produto'].nunique()}")
    
    print("\n=== Integration test completed successfully! ===")
    
    return final_data


if __name__ == "__main__":
    result_data = main()