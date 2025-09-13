"""
Example script demonstrating the data ingestion functionality.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.ingestion import DataIngestion
import logging

def main():
    """Demonstrate data ingestion with the hackathon data files."""
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize data ingestion
    print("Initializing DataIngestion...")
    ingestion = DataIngestion(use_polars=False)
    
    # Test loading the hackathon data files
    data_path = "hackathon_2025_templates"
    
    try:
        print(f"\nLoading data from: {data_path}")
        
        # Load just one file first to test
        import os
        files = [f for f in os.listdir(data_path) if f.endswith('.parquet')]
        first_file = os.path.join(data_path, files[0])
        
        print(f"Loading first file: {files[0]}")
        df = ingestion._load_parquet_data(first_file)
        
        print(f"Successfully loaded {len(df)} records")
        print(f"Columns: {list(df.columns)}")
        print(f"Data types:\n{df.dtypes}")
        
        # Generate data summary
        print("\nGenerating data summary...")
        summary = ingestion.get_data_summary(df)
        
        print(f"Shape: {summary['shape']}")
        print(f"Memory usage: {summary['memory_usage_mb']:.2f} MB")
        print(f"Null counts: {summary['null_counts']}")
        
        # Show first few rows
        print("\nFirst 5 rows:")
        print(df.head())
        
        # Validate data quality (this appears to be store/PDV data)
        print("\nValidating data quality...")
        validation = ingestion.validate_store_data(df)
        
        print(f"Validation passed: {validation['is_valid']}")
        if validation['errors']:
            print(f"Errors: {validation['errors']}")
        if validation['warnings']:
            print(f"Warnings: {validation['warnings']}")
            
        # Now try loading all files with different schemas (sample only for large files)
        print(f"\nLoading files with potentially different schemas (sample mode)...")
        schema_groups = ingestion.load_multiple_schemas(data_path, sample_only=True)
        
        print(f"Found {len(schema_groups)} different schemas:")
        for i, (schema, df) in enumerate(schema_groups.items()):
            print(f"  Schema {i+1}: {list(schema)} - {len(df)} records (sample)")
            print(f"    Sample data:\n{df.head(2)}")
            print(f"    Data types: {dict(df.dtypes)}")
            print()
            
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    print("\nData ingestion test completed successfully!")

if __name__ == "__main__":
    main()