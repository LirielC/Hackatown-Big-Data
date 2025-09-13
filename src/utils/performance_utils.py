"""
Performance utilities for optimized pipeline execution.

This module provides utilities for monitoring performance, memory management,
and coordinating optimized components.
"""

import logging
import time
import psutil
import gc
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
import pandas as pd
import numpy as np
from functools import wraps
from contextlib import contextmanager
import warnings


class PerformanceMonitor:
    """Monitor and log performance metrics during pipeline execution."""
    
    def __init__(self):
        """Initialize performance monitor."""
        self.logger = logging.getLogger(__name__)
        self.metrics = {}
        self.start_times = {}
        
    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        self.start_times[operation] = time.time()
        self.logger.info(f"Started: {operation}")
    
    def end_timer(self, operation: str) -> float:
        """End timing an operation and return duration."""
        if operation not in self.start_times:
            self.logger.warning(f"No start time found for operation: {operation}")
            return 0.0
        
        duration = time.time() - self.start_times[operation]
        self.metrics[operation] = duration
        
        self.logger.info(f"Completed: {operation} ({duration:.2f}s)")
        
        return duration
    
    @contextmanager
    def time_operation(self, operation: str):
        """Context manager for timing operations."""
        self.start_timer(operation)
        try:
            yield
        finally:
            self.end_timer(operation)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024
        }
    
    def log_memory_usage(self, operation: str = "current") -> None:
        """Log current memory usage."""
        memory_stats = self.get_memory_usage()
        self.logger.info(f"Memory usage ({operation}): "
                        f"RSS={memory_stats['rss_mb']:.1f}MB, "
                        f"Percent={memory_stats['percent']:.1f}%, "
                        f"Available={memory_stats['available_mb']:.1f}MB")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of all performance metrics."""
        return {
            'timing_metrics': self.metrics.copy(),
            'total_time': sum(self.metrics.values()),
            'memory_usage': self.get_memory_usage(),
            'operations_count': len(self.metrics)
        }


def performance_monitor(operation_name: str = None):
    """Decorator to monitor performance of functions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            monitor = PerformanceMonitor()
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            monitor.log_memory_usage(f"before_{op_name}")
            
            with monitor.time_operation(op_name):
                result = func(*args, **kwargs)
            
            monitor.log_memory_usage(f"after_{op_name}")
            
            return result
        
        return wrapper
    return decorator


class MemoryManager:
    """Manage memory usage and optimization during pipeline execution."""
    
    def __init__(self, memory_limit_gb: float = 8.0):
        """
        Initialize memory manager.
        
        Args:
            memory_limit_gb: Memory limit in GB
        """
        self.memory_limit_gb = memory_limit_gb
        self.memory_limit_bytes = memory_limit_gb * 1024 * 1024 * 1024
        self.logger = logging.getLogger(__name__)
    
    def check_memory_usage(self) -> bool:
        """
        Check if memory usage is within limits.
        
        Returns:
            True if within limits, False otherwise
        """
        process = psutil.Process()
        current_usage = process.memory_info().rss
        
        if current_usage > self.memory_limit_bytes:
            self.logger.warning(f"Memory usage ({current_usage / 1024 / 1024 / 1024:.2f} GB) "
                              f"exceeds limit ({self.memory_limit_gb} GB)")
            return False
        
        return True
    
    def force_garbage_collection(self) -> Dict[str, int]:
        """Force garbage collection and return statistics."""
        before_objects = len(gc.get_objects())
        
        # Force garbage collection
        collected = gc.collect()
        
        after_objects = len(gc.get_objects())
        
        stats = {
            'objects_before': before_objects,
            'objects_after': after_objects,
            'objects_collected': collected,
            'objects_freed': before_objects - after_objects
        }
        
        self.logger.info(f"Garbage collection: freed {stats['objects_freed']} objects")
        
        return stats
    
    def optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage by downcasting numeric types.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Memory-optimized DataFrame
        """
        original_memory = df.memory_usage(deep=True).sum()
        
        optimized_df = df.copy()
        
        # Optimize numeric columns
        for col in optimized_df.columns:
            col_type = optimized_df[col].dtype
            
            if col_type != 'object':
                if str(col_type).startswith('int'):
                    optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='integer')
                elif str(col_type).startswith('float'):
                    optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')
        
        # Convert object columns to category if beneficial
        for col in optimized_df.select_dtypes(include=['object']).columns:
            if optimized_df[col].nunique() / len(optimized_df) < 0.5:  # Less than 50% unique
                optimized_df[col] = optimized_df[col].astype('category')
        
        optimized_memory = optimized_df.memory_usage(deep=True).sum()
        reduction = (original_memory - optimized_memory) / original_memory * 100
        
        self.logger.info(f"Memory optimization: {reduction:.1f}% reduction "
                        f"({original_memory / 1024 / 1024:.1f}MB -> "
                        f"{optimized_memory / 1024 / 1024:.1f}MB)")
        
        return optimized_df
    
    @contextmanager
    def memory_limit_context(self):
        """Context manager that monitors memory usage."""
        initial_usage = psutil.Process().memory_info().rss
        
        try:
            yield
        finally:
            final_usage = psutil.Process().memory_info().rss
            usage_change = (final_usage - initial_usage) / 1024 / 1024
            
            self.logger.info(f"Memory change: {usage_change:+.1f}MB")
            
            if not self.check_memory_usage():
                self.force_garbage_collection()


class OptimizationCoordinator:
    """Coordinate optimized components for maximum performance."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize optimization coordinator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.optimization_config = config.get('optimization', {})
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.performance_monitor = PerformanceMonitor()
        self.memory_manager = MemoryManager(
            memory_limit_gb=self.optimization_config.get('memory_limit_gb', 8.0)
        )
        
        # Optimization settings
        self.use_polars = self.optimization_config.get('use_polars', True)
        self.use_caching = self.optimization_config.get('use_caching', True)
        self.use_parallel = self.optimization_config.get('use_parallel', True)
        self.batch_size = self.optimization_config.get('batch_size', 10000)
        
    def optimize_pipeline_execution(self, 
                                  pipeline_func: Callable,
                                  *args, **kwargs) -> Any:
        """
        Execute pipeline with full optimization.
        
        Args:
            pipeline_func: Pipeline function to execute
            *args, **kwargs: Arguments for pipeline function
            
        Returns:
            Pipeline execution result
        """
        self.logger.info("Starting optimized pipeline execution")
        
        with self.memory_manager.memory_limit_context():
            with self.performance_monitor.time_operation("full_pipeline"):
                # Set optimization flags
                if 'config' in kwargs:
                    kwargs['config']['optimization'] = self.optimization_config
                
                # Execute pipeline
                result = pipeline_func(*args, **kwargs)
                
                # Log performance summary
                summary = self.performance_monitor.get_performance_summary()
                self.logger.info(f"Pipeline completed in {summary['total_time']:.2f}s")
                
                return result
    
    def get_optimized_data_loader(self):
        """Get optimized data ingestion instance."""
        try:
            from ..data.ingestion_optimized import OptimizedDataIngestion
        except ImportError:
            # Handle case when running from different context
            import sys
            from pathlib import Path
            sys.path.append(str(Path(__file__).parent.parent))
            from data.ingestion_optimized import OptimizedDataIngestion
        
        return OptimizedDataIngestion(
            use_lazy_loading=True,
            chunk_size=self.batch_size,
            max_workers=self.optimization_config.get('max_workers')
        )
    
    def get_optimized_feature_engineer(self):
        """Get optimized feature engineering instance."""
        try:
            from ..features.engineering_optimized import OptimizedFeatureEngineer
        except ImportError:
            # Handle case when running from different context
            import sys
            from pathlib import Path
            sys.path.append(str(Path(__file__).parent.parent))
            from features.engineering_optimized import OptimizedFeatureEngineer
        
        return OptimizedFeatureEngineer(
            use_cache=self.use_caching,
            max_workers=self.optimization_config.get('max_workers'),
            use_polars=self.use_polars
        )
    
    def get_optimized_prediction_generator(self):
        """Get optimized prediction generator instance."""
        try:
            from ..models.prediction_optimized import OptimizedPredictionGenerator
        except ImportError:
            # Handle case when running from different context
            import sys
            from pathlib import Path
            sys.path.append(str(Path(__file__).parent.parent))
            from models.prediction_optimized import OptimizedPredictionGenerator
        
        return OptimizedPredictionGenerator(self.config)
    
    def create_optimization_report(self) -> Dict[str, Any]:
        """Create comprehensive optimization report."""
        return {
            'configuration': self.optimization_config,
            'performance_metrics': self.performance_monitor.get_performance_summary(),
            'memory_statistics': self.performance_monitor.get_memory_usage(),
            'optimization_flags': {
                'use_polars': self.use_polars,
                'use_caching': self.use_caching,
                'use_parallel': self.use_parallel,
                'batch_size': self.batch_size
            }
        }


def benchmark_function(func: Callable, *args, iterations: int = 1, **kwargs) -> Dict[str, Any]:
    """
    Benchmark a function's performance.
    
    Args:
        func: Function to benchmark
        *args: Function arguments
        iterations: Number of iterations to run
        **kwargs: Function keyword arguments
        
    Returns:
        Benchmark results
    """
    logger = logging.getLogger(__name__)
    
    times = []
    memory_usage = []
    
    for i in range(iterations):
        # Measure memory before
        initial_memory = psutil.Process().memory_info().rss
        
        # Time execution
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Measure memory after
        final_memory = psutil.Process().memory_info().rss
        
        execution_time = end_time - start_time
        memory_change = final_memory - initial_memory
        
        times.append(execution_time)
        memory_usage.append(memory_change)
        
        logger.info(f"Iteration {i+1}/{iterations}: {execution_time:.3f}s, "
                   f"Memory: {memory_change / 1024 / 1024:+.1f}MB")
    
    return {
        'iterations': iterations,
        'times': times,
        'avg_time': np.mean(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'std_time': np.std(times),
        'memory_changes': memory_usage,
        'avg_memory_change': np.mean(memory_usage),
        'total_memory_change': np.sum(memory_usage)
    }


def compare_implementations(implementations: Dict[str, Callable], 
                          *args, **kwargs) -> Dict[str, Dict[str, Any]]:
    """
    Compare performance of different implementations.
    
    Args:
        implementations: Dictionary of implementation name -> function
        *args, **kwargs: Arguments for functions
        
    Returns:
        Comparison results
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Comparing {len(implementations)} implementations")
    
    results = {}
    
    for name, func in implementations.items():
        logger.info(f"Benchmarking: {name}")
        
        try:
            benchmark_result = benchmark_function(func, *args, **kwargs)
            results[name] = benchmark_result
        except Exception as e:
            logger.error(f"Failed to benchmark {name}: {e}")
            results[name] = {'error': str(e)}
    
    # Create comparison summary
    if len(results) > 1:
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if valid_results:
            fastest = min(valid_results.keys(), key=lambda k: valid_results[k]['avg_time'])
            logger.info(f"Fastest implementation: {fastest} "
                       f"({valid_results[fastest]['avg_time']:.3f}s)")
    
    return results


# Utility functions for common optimizations

def optimize_pandas_settings():
    """Apply optimal pandas settings for performance."""
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
    
    # Set optimal pandas options
    pd.set_option('mode.chained_assignment', None)
    pd.set_option('compute.use_bottleneck', True)
    pd.set_option('compute.use_numexpr', True)


def clear_memory_cache():
    """Clear various memory caches."""
    # Clear pandas cache
    if hasattr(pd, '_cache'):
        pd._cache.clear()
    
    # Force garbage collection
    gc.collect()
    
    # Clear numpy cache
    if hasattr(np, '_NoValue'):
        np._NoValue._cache.clear()


def get_system_info() -> Dict[str, Any]:
    """Get system information for optimization decisions."""
    return {
        'cpu_count': psutil.cpu_count(),
        'cpu_count_logical': psutil.cpu_count(logical=True),
        'memory_total_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
        'memory_available_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024,
        'disk_usage': {
            'total_gb': psutil.disk_usage('/').total / 1024 / 1024 / 1024,
            'free_gb': psutil.disk_usage('/').free / 1024 / 1024 / 1024
        }
    }


class PerformanceOptimizer:
    """Otimizador de performance para o pipeline de submissões."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Inicializar otimizador de performance."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Configurações de otimização
        self.use_polars = self.config.get('use_polars', True)
        self.use_parallel = self.config.get('use_parallel', True)
        self.use_caching = self.config.get('use_caching', True)
        self.memory_limit_gb = self.config.get('memory_limit_gb', 8.0)

        # Componentes de otimização
        self.performance_monitor = PerformanceMonitor()
        self.memory_manager = MemoryManager(self.memory_limit_gb)

    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Otimizar DataFrame para melhor performance."""
        return self.memory_manager.optimize_dataframe_memory(df)

    def monitor_operation(self, operation_name: str):
        """Decorator para monitorar operações."""
        return performance_monitor(operation_name)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Obter resumo de performance."""
        return self.performance_monitor.get_performance_summary()

    def check_memory_limits(self) -> bool:
        """Verificar limites de memória."""
        return self.memory_manager.check_memory_usage()

    def force_garbage_collection(self) -> Dict[str, int]:
        """Forçar coleta de lixo."""
        return self.memory_manager.force_garbage_collection()