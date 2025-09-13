"""
Example script demonstrating performance optimizations.

This script shows how to use the optimized components for improved performance
in data loading, feature engineering, and prediction generation.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import logging
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.ingestion_optimized import OptimizedDataIngestion
from features.engineering_optimized import OptimizedFeatureEngineer
from models.prediction_optimized import OptimizedPredictionGenerator
from utils.performance_utils import (
    PerformanceMonitor, MemoryManager, OptimizationCoordinator,
    benchmark_function, compare_implementations, optimize_pandas_settings
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_data(num_rows: int = 100000) -> pd.DataFrame:
    """Create sample transaction data for testing."""
    logger.info(f"Creating sample data with {num_rows} rows")
    
    np.random.seed(42)
    
    # Generate sample data
    data = {
        'data_semana': pd.date_range('2022-01-01', periods=52, freq='W'),
        'pdv': np.random.randint(1, 1001, num_rows),
        'produto': np.random.randint(1, 5001, num_rows),
        'quantidade': np.random.poisson(10, num_rows),
        'semana': np.random.randint(1, 53, num_rows)
    }
    
    # Expand to full dataset
    df_base = pd.DataFrame({
        'data_semana': np.tile(data['data_semana'], num_rows // 52 + 1)[:num_rows],
        'pdv': data['pdv'],
        'produto': data['produto'],
        'quantidade': data['quantidade'],
        'semana': data['semana']
    })
    
    return df_base


def test_optimized_data_loading():
    """Test optimized data loading performance."""
    logger.info("Testing optimized data loading")
    
    # Create sample data and save as Parquet
    sample_data = create_sample_data(50000)
    
    # Create test directory
    test_dir = Path("data/test_performance")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Save sample files
    file_paths = []
    for i in range(3):
        file_path = test_dir / f"sample_data_{i}.parquet"
        sample_data.to_parquet(file_path, index=False)
        file_paths.append(file_path)
    
    # Test standard vs optimized loading
    from data.ingestion import DataIngestion
    
    standard_loader = DataIngestion(use_polars=False)
    optimized_loader = OptimizedDataIngestion(use_lazy_loading=True)
    
    def load_standard():
        return standard_loader.load_transactions(test_dir)
    
    def load_optimized():
        return optimized_loader.load_transactions_optimized(test_dir)
    
    # Compare implementations
    implementations = {
        'standard': load_standard,
        'optimized': load_optimized
    }
    
    results = compare_implementations(implementations)
    
    logger.info("Data loading comparison results:")
    for name, result in results.items():
        if 'error' not in result:
            logger.info(f"{name}: {result['avg_time']:.3f}s average")
        else:
            logger.error(f"{name}: {result['error']}")
    
    # Clean up
    for file_path in file_paths:
        file_path.unlink()
    
    return results


def test_optimized_feature_engineering():
    """Test optimized feature engineering performance."""
    logger.info("Testing optimized feature engineering")
    
    # Create sample data
    sample_data = create_sample_data(20000)
    
    # Test standard vs optimized feature engineering
    from features.engineering import FeatureEngineer
    
    standard_engineer = FeatureEngineer()
    optimized_engineer = OptimizedFeatureEngineer(use_cache=True, use_polars=True)
    
    def create_features_standard():
        return standard_engineer.create_temporal_features(sample_data)
    
    def create_features_optimized():
        return optimized_engineer.create_temporal_features_optimized(sample_data)
    
    def create_all_features_parallel():
        return optimized_engineer.create_all_features_parallel(
            sample_data, 
            feature_types=['temporal', 'lag']
        )
    
    # Compare implementations
    implementations = {
        'standard_temporal': create_features_standard,
        'optimized_temporal': create_features_optimized,
        'parallel_all': create_all_features_parallel
    }
    
    results = compare_implementations(implementations)
    
    logger.info("Feature engineering comparison results:")
    for name, result in results.items():
        if 'error' not in result:
            logger.info(f"{name}: {result['avg_time']:.3f}s average")
        else:
            logger.error(f"{name}: {result['error']}")
    
    return results


def test_optimized_prediction():
    """Test optimized prediction generation."""
    logger.info("Testing optimized prediction generation")
    
    # Create sample features data
    sample_data = create_sample_data(10000)
    
    # Add some dummy features
    np.random.seed(42)
    for i in range(20):
        sample_data[f'feature_{i}'] = np.random.randn(len(sample_data))
    
    # Create mock model
    class MockModel:
        def __init__(self):
            self.is_fitted = True
        
        def predict(self, X):
            return np.random.poisson(5, len(X))
    
    model = MockModel()
    
    # Create config
    config = {
        'prediction': {
            'target_weeks': [1, 2, 3, 4, 5],
            'post_processing': {
                'ensure_positive': True,
                'apply_bounds': True
            }
        },
        'optimization': {
            'batch_size': 5000,
            'use_parallel': True,
            'memory_limit_gb': 4.0
        }
    }
    
    # Test standard vs optimized prediction
    from models.prediction import PredictionGenerator
    
    standard_predictor = PredictionGenerator(config)
    optimized_predictor = OptimizedPredictionGenerator(config)
    
    def predict_standard():
        return standard_predictor.generate_predictions(model, sample_data)
    
    def predict_optimized():
        return optimized_predictor.generate_predictions_optimized(model, sample_data)
    
    # Compare implementations
    implementations = {
        'standard': predict_standard,
        'optimized': predict_optimized
    }
    
    results = compare_implementations(implementations)
    
    logger.info("Prediction generation comparison results:")
    for name, result in results.items():
        if 'error' not in result:
            logger.info(f"{name}: {result['avg_time']:.3f}s average")
        else:
            logger.error(f"{name}: {result['error']}")
    
    return results


def test_memory_optimization():
    """Test memory optimization utilities."""
    logger.info("Testing memory optimization")
    
    # Create large DataFrame
    large_data = create_sample_data(100000)
    
    # Add various data types
    large_data['category_col'] = np.random.choice(['A', 'B', 'C'], len(large_data))
    large_data['float_col'] = np.random.randn(len(large_data))
    large_data['int_col'] = np.random.randint(0, 1000, len(large_data))
    
    # Test memory optimization
    memory_manager = MemoryManager()
    
    original_memory = large_data.memory_usage(deep=True).sum()
    logger.info(f"Original memory usage: {original_memory / 1024 / 1024:.2f} MB")
    
    optimized_data = memory_manager.optimize_dataframe_memory(large_data)
    
    optimized_memory = optimized_data.memory_usage(deep=True).sum()
    logger.info(f"Optimized memory usage: {optimized_memory / 1024 / 1024:.2f} MB")
    
    reduction = (original_memory - optimized_memory) / original_memory * 100
    logger.info(f"Memory reduction: {reduction:.1f}%")
    
    return {
        'original_memory_mb': original_memory / 1024 / 1024,
        'optimized_memory_mb': optimized_memory / 1024 / 1024,
        'reduction_percent': reduction
    }


def test_full_optimization_pipeline():
    """Test full optimized pipeline execution."""
    logger.info("Testing full optimization pipeline")
    
    # Create configuration
    config = {
        'optimization': {
            'use_polars': True,
            'use_caching': True,
            'use_parallel': True,
            'batch_size': 10000,
            'memory_limit_gb': 4.0,
            'max_workers': 4
        },
        'prediction': {
            'target_weeks': [1, 2, 3, 4, 5]
        }
    }
    
    # Initialize optimization coordinator
    coordinator = OptimizationCoordinator(config)
    
    def optimized_pipeline():
        """Full optimized pipeline."""
        # Create sample data
        sample_data = create_sample_data(30000)
        
        # Get optimized components
        data_loader = coordinator.get_optimized_data_loader()
        feature_engineer = coordinator.get_optimized_feature_engineer()
        
        # Create features
        with coordinator.performance_monitor.time_operation("feature_engineering"):
            features_df = feature_engineer.create_all_features_parallel(
                sample_data,
                feature_types=['temporal', 'lag']
            )
        
        # Memory optimization
        with coordinator.performance_monitor.time_operation("memory_optimization"):
            features_df = coordinator.memory_manager.optimize_dataframe_memory(features_df)
        
        return features_df
    
    # Execute optimized pipeline
    result = coordinator.optimize_pipeline_execution(optimized_pipeline)
    
    # Generate optimization report
    report = coordinator.create_optimization_report()
    
    logger.info("Optimization pipeline completed")
    logger.info(f"Result shape: {result.shape}")
    logger.info(f"Total execution time: {report['performance_metrics']['total_time']:.2f}s")
    
    return report


def main():
    """Run all performance optimization tests."""
    logger.info("Starting performance optimization tests")
    
    # Apply pandas optimizations
    optimize_pandas_settings()
    
    # Initialize performance monitor
    monitor = PerformanceMonitor()
    
    results = {}
    
    # Test 1: Data loading optimization
    with monitor.time_operation("data_loading_test"):
        try:
            results['data_loading'] = test_optimized_data_loading()
        except Exception as e:
            logger.error(f"Data loading test failed: {e}")
            results['data_loading'] = {'error': str(e)}
    
    # Test 2: Feature engineering optimization
    with monitor.time_operation("feature_engineering_test"):
        try:
            results['feature_engineering'] = test_optimized_feature_engineering()
        except Exception as e:
            logger.error(f"Feature engineering test failed: {e}")
            results['feature_engineering'] = {'error': str(e)}
    
    # Test 3: Prediction optimization
    with monitor.time_operation("prediction_test"):
        try:
            results['prediction'] = test_optimized_prediction()
        except Exception as e:
            logger.error(f"Prediction test failed: {e}")
            results['prediction'] = {'error': str(e)}
    
    # Test 4: Memory optimization
    with monitor.time_operation("memory_optimization_test"):
        try:
            results['memory_optimization'] = test_memory_optimization()
        except Exception as e:
            logger.error(f"Memory optimization test failed: {e}")
            results['memory_optimization'] = {'error': str(e)}
    
    # Test 5: Full pipeline optimization
    with monitor.time_operation("full_pipeline_test"):
        try:
            results['full_pipeline'] = test_full_optimization_pipeline()
        except Exception as e:
            logger.error(f"Full pipeline test failed: {e}")
            results['full_pipeline'] = {'error': str(e)}
    
    # Generate final report
    final_report = {
        'test_results': results,
        'overall_performance': monitor.get_performance_summary(),
        'timestamp': datetime.now().isoformat()
    }
    
    # Save report
    report_path = Path("performance_optimization_report.yaml")
    with open(report_path, 'w') as f:
        yaml.dump(final_report, f, default_flow_style=False)
    
    logger.info(f"Performance optimization tests completed")
    logger.info(f"Report saved to: {report_path}")
    logger.info(f"Total test time: {monitor.get_performance_summary()['total_time']:.2f}s")
    
    return final_report


if __name__ == "__main__":
    main()