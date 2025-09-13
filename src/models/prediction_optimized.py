"""
Optimized prediction module with batch processing and memory efficiency.

This module provides performance-optimized prediction generation using
batch processing, memory management, and parallel execution.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Iterator
import logging
from datetime import datetime, timedelta
import warnings
from pathlib import Path
import joblib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
import gc

from .training import BaseModel
from .ensemble import BaseEnsemble
from .output_formatter import SubmissionFormatter, OutputFormatterError

logger = logging.getLogger(__name__)


class BatchPredictionOptimizer:
    """
    Optimized batch prediction with memory management and parallel processing.
    """
    
    def __init__(self, 
                 batch_size: int = 10000,
                 max_workers: Optional[int] = None,
                 memory_limit_gb: float = 4.0):
        """
        Initialize batch prediction optimizer.
        
        Args:
            batch_size: Size of prediction batches
            max_workers: Maximum number of worker processes
            memory_limit_gb: Memory limit in GB for batch processing
        """
        self.batch_size = batch_size
        self.max_workers = max_workers or min(mp.cpu_count(), 8)
        self.memory_limit_gb = memory_limit_gb
        self.logger = logging.getLogger(__name__)
        
    def create_batches(self, df: pd.DataFrame) -> Iterator[pd.DataFrame]:
        """
        Create batches from DataFrame for processing.
        
        Args:
            df: Input DataFrame
            
        Yields:
            DataFrame batches
        """
        total_rows = len(df)
        num_batches = (total_rows + self.batch_size - 1) // self.batch_size
        
        self.logger.info(f"Creating {num_batches} batches of size {self.batch_size}")
        
        for i in range(0, total_rows, self.batch_size):
            end_idx = min(i + self.batch_size, total_rows)
            batch = df.iloc[i:end_idx].copy()
            
            self.logger.debug(f"Batch {i//self.batch_size + 1}/{num_batches}: {len(batch)} rows")
            
            yield batch
    
    def predict_batch(self, 
                     model: Union[BaseModel, BaseEnsemble],
                     batch_df: pd.DataFrame,
                     feature_columns: List[str]) -> np.ndarray:
        """
        Generate predictions for a single batch.
        
        Args:
            model: Trained model
            batch_df: Batch DataFrame
            feature_columns: List of feature column names
            
        Returns:
            Predictions array
        """
        # Extract features
        X_batch = batch_df[feature_columns].values
        
        # Handle missing values
        X_batch = np.nan_to_num(X_batch, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Generate predictions
        predictions = model.predict(X_batch)
        
        # Clean up memory
        del X_batch
        gc.collect()
        
        return predictions
    
    def predict_parallel_batches(self, 
                                model: Union[BaseModel, BaseEnsemble],
                                df: pd.DataFrame,
                                feature_columns: List[str]) -> np.ndarray:
        """
        Generate predictions using parallel batch processing.
        
        Args:
            model: Trained model
            df: Input DataFrame
            feature_columns: List of feature column names
            
        Returns:
            Combined predictions array
        """
        self.logger.info("Starting parallel batch prediction")
        
        # Create prediction function
        predict_func = partial(self.predict_batch, model, feature_columns=feature_columns)
        
        # Process batches in parallel
        all_predictions = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all batches
            future_to_batch = {}
            batch_indices = []
            
            for batch_idx, batch in enumerate(self.create_batches(df)):
                future = executor.submit(predict_func, batch)
                future_to_batch[future] = batch_idx
                batch_indices.append(batch_idx)
            
            # Collect results in order
            batch_results = [None] * len(batch_indices)
            
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    predictions = future.result()
                    batch_results[batch_idx] = predictions
                    self.logger.debug(f"Completed batch {batch_idx}")
                except Exception as e:
                    self.logger.error(f"Batch {batch_idx} failed: {e}")
                    # Create zero predictions as fallback
                    batch_size = min(self.batch_size, len(df) - batch_idx * self.batch_size)
                    batch_results[batch_idx] = np.zeros(batch_size)
        
        # Combine all predictions
        all_predictions = np.concatenate([pred for pred in batch_results if pred is not None])
        
        self.logger.info(f"Parallel batch prediction completed: {len(all_predictions)} predictions")
        
        return all_predictions


class OptimizedPredictionGenerator:
    """
    Performance-optimized prediction generator with batch processing and memory management.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize optimized prediction generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.prediction_config = config.get('prediction', {})
        self.optimization_config = config.get('optimization', {})
        
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # Optimization settings
        self.batch_size = self.optimization_config.get('batch_size', 10000)
        self.max_workers = self.optimization_config.get('max_workers', min(mp.cpu_count(), 8))
        self.memory_limit_gb = self.optimization_config.get('memory_limit_gb', 4.0)
        self.use_parallel = self.optimization_config.get('use_parallel', True)
        
        # Initialize batch optimizer
        self.batch_optimizer = BatchPredictionOptimizer(
            batch_size=self.batch_size,
            max_workers=self.max_workers,
            memory_limit_gb=self.memory_limit_gb
        )
        
        # Post-processing settings
        self.ensure_positive = self.prediction_config.get('post_processing', {}).get('ensure_positive', True)
        self.apply_bounds = self.prediction_config.get('post_processing', {}).get('apply_bounds', True)
        self.max_multiplier = self.prediction_config.get('post_processing', {}).get('max_multiplier', 3.0)
        
        # Target weeks
        self.target_weeks = self.prediction_config.get('target_weeks', [1, 2, 3, 4, 5])
        
    def _setup_logging(self) -> None:
        """Configure logging."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def generate_predictions_optimized(self, 
                                     model: Union[BaseModel, BaseEnsemble],
                                     features_df: pd.DataFrame,
                                     historical_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate predictions with optimization.
        
        Args:
            model: Trained model or ensemble
            features_df: DataFrame with features for prediction
            historical_data: Historical data for reference (optional)
            
        Returns:
            DataFrame with optimized predictions
        """
        self.logger.info("Starting optimized prediction generation")
        
        try:
            # Validate inputs
            self._validate_prediction_inputs(model, features_df)
            
            # Prepare features efficiently
            prediction_features, feature_columns = self._prepare_features_optimized(features_df)
            
            # Generate predictions with batch processing
            if self.use_parallel and len(features_df) > self.batch_size:
                raw_predictions = self.batch_optimizer.predict_parallel_batches(
                    model, prediction_features, feature_columns
                )
            else:
                raw_predictions = self._generate_predictions_single_batch(
                    model, prediction_features, feature_columns
                )
            
            # Apply optimized post-processing
            processed_predictions = self._apply_post_processing_optimized(
                raw_predictions, features_df, historical_data
            )
            
            # Format output efficiently
            formatted_predictions = self._format_predictions_optimized(
                processed_predictions, features_df
            )
            
            # Validate results
            self._validate_predictions(formatted_predictions)
            
            self.logger.info(f"Optimized predictions generated: {len(formatted_predictions)} records")
            
            return formatted_predictions
            
        except Exception as e:
            self.logger.error(f"Optimized prediction generation failed: {str(e)}")
            raise
    
    def _validate_prediction_inputs(self, 
                                  model: Union[BaseModel, BaseEnsemble],
                                  features_df: pd.DataFrame) -> None:
        """Validate prediction inputs."""
        # Check if model is fitted
        if hasattr(model, 'is_fitted') and not model.is_fitted:
            raise ValueError("Model must be fitted before generating predictions")
        
        # Check DataFrame
        if len(features_df) == 0:
            raise ValueError("Features DataFrame is empty")
        
        # Check required columns
        required_columns = ['pdv', 'produto', 'semana']
        missing_columns = [col for col in required_columns if col not in features_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check memory usage
        memory_usage_gb = features_df.memory_usage(deep=True).sum() / (1024**3)
        if memory_usage_gb > self.memory_limit_gb:
            self.logger.warning(f"DataFrame memory usage ({memory_usage_gb:.2f} GB) exceeds limit ({self.memory_limit_gb} GB)")
    
    def _prepare_features_optimized(self, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare features for prediction with memory optimization."""
        self.logger.info("Preparing features for prediction")
        
        # Identify feature columns
        exclude_columns = ['pdv', 'produto', 'semana', 'quantidade', 'data', 'data_semana']
        feature_columns = [col for col in features_df.columns if col not in exclude_columns]
        
        if len(feature_columns) == 0:
            raise ValueError("No valid feature columns found")
        
        # Create optimized feature DataFrame
        prediction_features = features_df[['pdv', 'produto', 'semana'] + feature_columns].copy()
        
        # Optimize data types
        prediction_features = self._optimize_dtypes(prediction_features)
        
        # Handle missing values efficiently
        prediction_features[feature_columns] = prediction_features[feature_columns].fillna(0)
        
        # Remove infinite values
        prediction_features[feature_columns] = prediction_features[feature_columns].replace(
            [np.inf, -np.inf], 0
        )
        
        self.logger.info(f"Features prepared: {len(feature_columns)} columns, {len(prediction_features)} rows")
        
        return prediction_features, feature_columns
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame data types for memory efficiency."""
        optimized_df = df.copy()
        
        for col in optimized_df.columns:
            col_type = optimized_df[col].dtype
            
            if col_type == 'int64':
                # Try to downcast integers
                col_min = optimized_df[col].min()
                col_max = optimized_df[col].max()
                
                if col_min >= 0:
                    if col_max < 255:
                        optimized_df[col] = optimized_df[col].astype('uint8')
                    elif col_max < 65535:
                        optimized_df[col] = optimized_df[col].astype('uint16')
                    elif col_max < 4294967295:
                        optimized_df[col] = optimized_df[col].astype('uint32')
                else:
                    if col_min > -128 and col_max < 127:
                        optimized_df[col] = optimized_df[col].astype('int8')
                    elif col_min > -32768 and col_max < 32767:
                        optimized_df[col] = optimized_df[col].astype('int16')
                    elif col_min > -2147483648 and col_max < 2147483647:
                        optimized_df[col] = optimized_df[col].astype('int32')
            
            elif col_type == 'float64':
                # Try to downcast floats
                optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')
        
        return optimized_df
    
    def _generate_predictions_single_batch(self, 
                                         model: Union[BaseModel, BaseEnsemble],
                                         features_df: pd.DataFrame,
                                         feature_columns: List[str]) -> np.ndarray:
        """Generate predictions for single batch."""
        self.logger.info("Generating predictions (single batch)")
        
        # Extract features
        X = features_df[feature_columns].values
        
        # Generate predictions
        predictions = model.predict(X)
        
        return predictions
    
    def _apply_post_processing_optimized(self, 
                                       predictions: np.ndarray,
                                       features_df: pd.DataFrame,
                                       historical_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Apply optimized post-processing to predictions."""
        self.logger.info("Applying optimized post-processing")
        
        processed_predictions = predictions.copy()
        
        # 1. Ensure non-negative values (vectorized)
        if self.ensure_positive:
            negative_count = (processed_predictions < 0).sum()
            if negative_count > 0:
                self.logger.info(f"Converting {negative_count} negative values to zero")
                processed_predictions = np.maximum(processed_predictions, 0)
        
        # 2. Apply bounds using vectorized operations
        if self.apply_bounds and historical_data is not None:
            processed_predictions = self._apply_bounds_vectorized(
                processed_predictions, features_df, historical_data
            )
        
        # 3. Apply smoothing (vectorized)
        processed_predictions = self._apply_smoothing_vectorized(processed_predictions)
        
        # 4. Round to integers (vectorized)
        processed_predictions = np.round(processed_predictions).astype(np.int32)
        
        self.logger.info(f"Post-processing completed: min={processed_predictions.min()}, "
                        f"max={processed_predictions.max()}, mean={processed_predictions.mean():.2f}")
        
        return processed_predictions
    
    def _apply_bounds_vectorized(self, 
                               predictions: np.ndarray,
                               features_df: pd.DataFrame,
                               historical_data: pd.DataFrame) -> np.ndarray:
        """Apply bounds using vectorized operations."""
        if 'quantidade' not in historical_data.columns:
            return predictions
        
        # Calculate historical statistics efficiently
        hist_stats = historical_data.groupby(['pdv', 'produto'])['quantidade'].agg([
            'mean', 'max'
        ]).reset_index()
        
        # Merge with features DataFrame
        features_with_stats = features_df[['pdv', 'produto']].merge(
            hist_stats, on=['pdv', 'produto'], how='left'
        )
        
        # Calculate bounds vectorized
        hist_mean = features_with_stats['mean'].fillna(0).values
        hist_max = features_with_stats['max'].fillna(1000).values
        
        # Apply bounds
        max_allowed = np.minimum(hist_max * self.max_multiplier, hist_mean * 10)
        max_allowed = np.maximum(max_allowed, 1000)  # Minimum bound
        
        bounded_predictions = np.minimum(predictions, max_allowed)
        
        return bounded_predictions
    
    def _apply_smoothing_vectorized(self, predictions: np.ndarray) -> np.ndarray:
        """Apply smoothing using vectorized operations."""
        # Calculate percentiles
        q99 = np.percentile(predictions, 99)
        q95 = np.percentile(predictions, 95)
        
        # Apply smoothing
        smoothed_predictions = np.where(predictions > q99, q95, predictions)
        
        extreme_count = (predictions > q99).sum()
        if extreme_count > 0:
            self.logger.info(f"Smoothed {extreme_count} extreme values")
        
        return smoothed_predictions
    
    def _format_predictions_optimized(self, 
                                    predictions: np.ndarray,
                                    features_df: pd.DataFrame) -> pd.DataFrame:
        """Format predictions with memory optimization."""
        self.logger.info("Formatting predictions for output")
        
        # Create output DataFrame efficiently
        output_data = {
            'semana': features_df['semana'].values,
            'pdv': features_df['pdv'].values,
            'produto': features_df['produto'].values,
            'quantidade': predictions
        }
        
        output_df = pd.DataFrame(output_data)
        
        # Filter target weeks
        output_df = output_df[output_df['semana'].isin(self.target_weeks)]
        
        # Optimize data types
        output_df['semana'] = output_df['semana'].astype('int16')
        output_df['pdv'] = output_df['pdv'].astype('category')
        output_df['produto'] = output_df['produto'].astype('category')
        output_df['quantidade'] = output_df['quantidade'].astype('int32')
        
        # Sort efficiently
        output_df = output_df.sort_values(['semana', 'pdv', 'produto']).reset_index(drop=True)
        
        self.logger.info(f"Predictions formatted: {len(output_df)} records")
        
        return output_df
    
    def _validate_predictions(self, predictions_df: pd.DataFrame) -> None:
        """Validate predictions with optimized checks."""
        self.logger.info("Validating predictions")
        
        # Basic validation
        if len(predictions_df) == 0:
            raise ValueError("Predictions DataFrame is empty")
        
        # Check required columns
        required_columns = ['semana', 'pdv', 'produto', 'quantidade']
        missing_columns = [col for col in required_columns if col not in predictions_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for null values (vectorized)
        null_counts = predictions_df.isnull().sum()
        if null_counts.sum() > 0:
            raise ValueError(f"Null values found: {null_counts[null_counts > 0].to_dict()}")
        
        # Check for negative values (vectorized)
        negative_count = (predictions_df['quantidade'] < 0).sum()
        if negative_count > 0:
            raise ValueError(f"Found {negative_count} negative quantities")
        
        # Check for duplicates (only warn, don't fail)
        duplicates = predictions_df.duplicated(subset=['semana', 'pdv', 'produto']).sum()
        if duplicates > 0:
            self.logger.warning(f"Found {duplicates} duplicate combinations - this may be expected for test data")
        
        self.logger.info("Prediction validation passed")
    
    def save_predictions_optimized(self, 
                                 predictions_df: pd.DataFrame,
                                 output_path: str,
                                 format_type: str = 'csv',
                                 compression: Optional[str] = None) -> str:
        """
        Save predictions with optimization.
        
        Args:
            predictions_df: DataFrame with predictions
            output_path: Path to save file
            format_type: Format type ('csv' or 'parquet')
            compression: Compression type
            
        Returns:
            Path to saved file
        """
        self.logger.info(f"Saving predictions in optimized {format_type} format")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format_type.lower() == 'csv':
            # Optimize CSV saving
            predictions_df.to_csv(
                output_path,
                index=False,
                sep=';',
                encoding='utf-8',
                compression=compression
            )
        elif format_type.lower() == 'parquet':
            # Optimize Parquet saving
            predictions_df.to_parquet(
                output_path,
                index=False,
                compression=compression or 'snappy',
                engine='pyarrow'
            )
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        self.logger.info(f"Predictions saved: {output_path}")
        
        return str(output_path)
    
    def generate_predictions_streaming(self, 
                                     model: Union[BaseModel, BaseEnsemble],
                                     features_df: pd.DataFrame,
                                     output_path: str) -> str:
        """
        Generate and save predictions using streaming for very large datasets.
        
        Args:
            model: Trained model
            features_df: Features DataFrame
            output_path: Output file path
            
        Returns:
            Path to saved file
        """
        self.logger.info("Starting streaming prediction generation")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare features
        _, feature_columns = self._prepare_features_optimized(features_df)
        
        # Process in batches and write incrementally
        first_batch = True
        
        for batch_idx, batch in enumerate(self.batch_optimizer.create_batches(features_df)):
            # Generate predictions for batch
            batch_predictions = self.batch_optimizer.predict_batch(
                model, batch, feature_columns
            )
            
            # Apply post-processing
            batch_predictions = self._apply_post_processing_optimized(
                batch_predictions, batch, None
            )
            
            # Format batch
            batch_output = self._format_predictions_optimized(batch_predictions, batch)
            
            # Write to file
            if first_batch:
                batch_output.to_csv(output_path, index=False, sep=';', encoding='utf-8')
                first_batch = False
            else:
                batch_output.to_csv(output_path, index=False, sep=';', encoding='utf-8', 
                                  mode='a', header=False)
            
            self.logger.info(f"Processed batch {batch_idx + 1}")
            
            # Clean up memory
            del batch_predictions, batch_output
            gc.collect()
        
        self.logger.info(f"Streaming predictions saved: {output_path}")
        
        return str(output_path)