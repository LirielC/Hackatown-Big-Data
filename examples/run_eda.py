#!/usr/bin/env python3
"""
Script to run Exploratory Data Analysis on hackathon data.

This script loads the data, performs comprehensive EDA analysis,
and generates insights for feature engineering.
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.ingestion import DataIngestion
from utils.eda_utils import EDAAnalyzer
import logging

def main():
    """Run EDA analysis on hackathon data."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Starting EDA analysis...")
    
    # Initialize components
    ingestion = DataIngestion(use_polars=True)
    eda_analyzer = EDAAnalyzer(save_plots=True, plot_dir="plots/eda")
    
    # Load data
    data_path = Path("hackathon_2025_templates")
    
    try:
        logger.info(f"Loading data from {data_path}")
        
        # Load multiple schemas if necessary
        data_schemas = ingestion.load_multiple_schemas(data_path, sample_only=False)
        
        logger.info(f"Found {len(data_schemas)} different schemas")
        
        # Use the largest dataset for analysis
        main_df = None
        max_records = 0
        
        for schema, df in data_schemas.items():
            logger.info(f"Schema with {len(df)} records: {list(schema)}")
            if len(df) > max_records:
                main_df = df
                max_records = len(df)
        
        if main_df is None:
            raise ValueError("No data loaded successfully")
        
        logger.info(f"Using dataset with {len(main_df)} records for EDA")
        
        # Run comprehensive EDA
        logger.info("Running comprehensive EDA analysis...")
        insights_summary = eda_analyzer.generate_insights_summary(main_df)
        
        # Save results
        results_path = Path("data/processed/eda_results.json")
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if hasattr(obj, 'item'):
                return obj.item()
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        insights_summary_serializable = convert_numpy_types(insights_summary)
        
        with open(results_path, 'w') as f:
            json.dump(insights_summary_serializable, f, indent=2, default=str)
        
        logger.info(f"EDA results saved to {results_path}")
        
        # Print summary
        print("\n" + "="*80)
        print("EDA ANALYSIS SUMMARY")
        print("="*80)
        
        overview = insights_summary['dataset_overview']
        print(f"📊 Dataset: {overview['records']:,} records, {overview['columns']} columns")
        print(f"💾 Memory: {overview['memory_mb']:.1f} MB")
        
        print(f"\n🔍 Key Insights:")
        for insight in insights_summary['insights']:
            print(f"   • {insight}")
        
        print(f"\n💡 Recommendations:")
        for rec in insights_summary['recommendations']:
            print(f"   • {rec}")
        
        print("\n" + "="*80)
        print("✅ EDA Analysis completed successfully!")
        print(f"📁 Plots saved to: plots/eda/")
        print(f"📄 Results saved to: {results_path}")
        print("="*80)
        
    except Exception as e:
        logger.error(f"EDA analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()