"""
Optimized feature engineering module with parallelization and caching.

This module provides performance-optimized feature engineering using
parallel processing and intelligent caching mechanisms.
"""

import logging
import warnings
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
import pandas as pd
import numpy as np
import polars as pl
from datetime import datetime, timedelta
import holidays
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
import hashlib
import pickle
from pathlib import Path
import joblib


class FeatureCache:
    """Intelligent caching system for computed features."""
    
    def __init__(self, cache_dir: str = "data/features/cache"):
        """
        Initialize feature cache.
        
        Args:
            cache_dir: Directory to store cached features
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def _get_cache_key(self, data_hash: str, feature_type: str, params: Dict) -> str:
        """Generate cache key for feature set."""
        params_str = str(sorted(params.items()))
        combined = f"{data_hash}_{feature_type}_{params_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _get_data_hash(self, df: pd.DataFrame) -> str:
        """Generate hash for DataFrame to detect changes."""
        # Use shape and column names for quick hash
        shape_str = f"{df.shape[0]}x{df.shape[1]}"
        cols_str = "_".join(sorted(df.columns))
        
        # Sample a few rows for content hash
        if len(df) > 1000:
            sample_df = df.sample(n=100, random_state=42)
        else:
            sample_df = df
        
        content_hash = hashlib.md5(
            pd.util.hash_pandas_object(sample_df).values
        ).hexdigest()[:8]
        
        return f"{shape_str}_{cols_str}_{content_hash}"
    
    def get_cached_features(self, 
                           df: pd.DataFrame, 
                           feature_type: str, 
                           params: Dict) -> Optional[pd.DataFrame]:
        """
        Retrieve cached features if available.
        
        Args:
            df: Input DataFrame
            feature_type: Type of features (e.g., 'temporal', 'lag')
            params: Parameters used for feature generation
            
        Returns:
            Cached features DataFrame or None if not found
        """
        try:
            data_hash = self._get_data_hash(df)
            cache_key = self._get_cache_key(data_hash, feature_type, params)
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            if cache_file.exists():
                self.logger.info(f"Loading cached {feature_type} features")
                return joblib.load(cache_file)
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to load cached features: {e}")
            return None
    
    def save_features(self, 
                     df: pd.DataFrame, 
                     features_df: pd.DataFrame,
                     feature_type: str, 
                     params: Dict) -> None:
        """
        Save computed features to cache.
        
        Args:
            df: Input DataFrame
            features_df: Computed features DataFrame
            feature_type: Type of features
            params: Parameters used for feature generation
        """
        try:
            data_hash = self._get_data_hash(df)
            cache_key = self._get_cache_key(data_hash, feature_type, params)
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            joblib.dump(features_df, cache_file, compress=3)
            self.logger.info(f"Cached {feature_type} features saved")
            
        except Exception as e:
            self.logger.warning(f"Failed to save features to cache: {e}")
    
    def clear_cache(self) -> None:
        """Clear all cached features."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        self.logger.info("Feature cache cleared")


class OptimizedFeatureEngineer:
    """
    Performance-optimized feature engineering with parallelization and caching.
    """
    
    def __init__(self, 
                 country_code: str = 'BR',
                 use_cache: bool = True,
                 cache_dir: str = "data/features/cache",
                 max_workers: Optional[int] = None,
                 use_polars: bool = True):
        """
        Initialize OptimizedFeatureEngineer.
        
        Args:
            country_code: Country code for holiday calendar
            use_cache: Whether to use feature caching
            cache_dir: Directory for cached features
            max_workers: Maximum number of worker processes
            use_polars: Whether to use Polars for optimization
        """
        self.country_code = country_code
        self.use_cache = use_cache
        self.max_workers = max_workers or min(mp.cpu_count(), 8)
        self.use_polars = use_polars
        
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        self._setup_holidays()
        
        if self.use_cache:
            self.cache = FeatureCache(cache_dir)
        
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
    
    def _setup_holidays(self) -> None:
        """Setup holiday calendar."""
        try:
            if self.country_code == 'BR':
                self.holiday_calendar = holidays.Brazil()
            else:
                self.holiday_calendar = holidays.country_holidays(self.country_code)
        except Exception as e:
            self.logger.warning(f"Could not initialize holiday calendar: {e}")
            self.holiday_calendar = {}
    
    def create_all_features_parallel(self, 
                                   df: pd.DataFrame,
                                   feature_types: List[str] = None) -> pd.DataFrame:
        """
        Create all features using parallel processing.
        
        Args:
            df: Input DataFrame
            feature_types: List of feature types to create
            
        Returns:
            DataFrame with all features
        """
        if feature_types is None:
            feature_types = ['temporal', 'product', 'store', 'lag', 'rolling']
        
        self.logger.info(f"Creating features in parallel: {feature_types}")
        
        # Check cache first
        if self.use_cache:
            cache_params = {'feature_types': feature_types}
            cached_features = self.cache.get_cached_features(df, 'all_features', cache_params)
            if cached_features is not None:
                return cached_features
        
        # Create features in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_type = {}
            
            if 'temporal' in feature_types:
                future_to_type[executor.submit(self.create_temporal_features_optimized, df)] = 'temporal'
            
            if 'product' in feature_types:
                future_to_type[executor.submit(self.create_product_features_optimized, df)] = 'product'
            
            if 'store' in feature_types:
                future_to_type[executor.submit(self.create_store_features_optimized, df)] = 'store'
            
            if 'lag' in feature_types:
                future_to_type[executor.submit(self.create_lag_features_optimized, df)] = 'lag'
            
            if 'rolling' in feature_types:
                future_to_type[executor.submit(self.create_rolling_features_optimized, df)] = 'rolling'
            
            # Collect results
            feature_dfs = [df.copy()]
            
            for future in as_completed(future_to_type):
                feature_type = future_to_type[future]
                try:
                    result_df = future.result()
                    feature_dfs.append(result_df)
                    self.logger.info(f"Completed {feature_type} features")
                except Exception as e:
                    self.logger.error(f"Failed to create {feature_type} features: {e}")
        
        # Combine all features
        self.logger.info("Combining all feature sets")
        
        # Use Polars for efficient joining if available
        if self.use_polars and len(feature_dfs) > 1:
            combined_df = self._combine_features_polars(feature_dfs)
        else:
            combined_df = self._combine_features_pandas(feature_dfs)
        
        # Cache the result
        if self.use_cache:
            self.cache.save_features(df, combined_df, 'all_features', cache_params)
        
        self.logger.info(f"All features created: {combined_df.shape}")
        
        return combined_df
    
    def _combine_features_polars(self, feature_dfs: List[pd.DataFrame]) -> pd.DataFrame:
        """Combine feature DataFrames using Polars for efficiency."""
        # Convert to Polars
        pl_dfs = [pl.from_pandas(df) for df in feature_dfs]
        
        # Start with the first DataFrame
        combined = pl_dfs[0]
        
        # Join with other DataFrames
        join_keys = ['pdv', 'produto', 'semana'] if all(
            all(col in df.columns for col in ['pdv', 'produto', 'semana']) 
            for df in feature_dfs
        ) else None
        
        if join_keys:
            for pl_df in pl_dfs[1:]:
                # Get only new columns
                new_columns = [col for col in pl_df.columns if col not in combined.columns]
                if new_columns:
                    select_columns = join_keys + new_columns
                    combined = combined.join(
                        pl_df.select(select_columns),
                        on=join_keys,
                        how='left'
                    )
        else:
            # Fallback to concatenation if no common keys
            combined = pl.concat(pl_dfs, how='horizontal')
        
        return combined.to_pandas()
    
    def _combine_features_pandas(self, feature_dfs: List[pd.DataFrame]) -> pd.DataFrame:
        """Combine feature DataFrames using Pandas."""
        combined_df = feature_dfs[0].copy()
        
        for df in feature_dfs[1:]:
            # Find common columns for joining
            common_cols = list(set(combined_df.columns) & set(df.columns))
            
            if len(common_cols) >= 2:  # Need at least 2 columns for meaningful join
                # Get only new columns
                new_columns = [col for col in df.columns if col not in combined_df.columns]
                if new_columns:
                    join_columns = common_cols + new_columns
                    combined_df = combined_df.merge(
                        df[join_columns],
                        on=common_cols,
                        how='left'
                    )
            else:
                # Concatenate horizontally if no common keys
                combined_df = pd.concat([combined_df, df], axis=1)
        
        return combined_df
    
    def create_temporal_features_optimized(self, 
                                         df: pd.DataFrame,
                                         date_column: str = 'data_semana') -> pd.DataFrame:
        """Create temporal features with optimization."""
        params = {'date_column': date_column}
        
        # Check cache
        if self.use_cache:
            cached = self.cache.get_cached_features(df, 'temporal', params)
            if cached is not None:
                return cached
        
        self.logger.info("Creating optimized temporal features")
        
        if self.use_polars:
            result_df = self._create_temporal_features_polars(df, date_column)
        else:
            result_df = self._create_temporal_features_pandas(df, date_column)
        
        # Cache result
        if self.use_cache:
            self.cache.save_features(df, result_df, 'temporal', params)
        
        return result_df
    
    def _create_temporal_features_polars(self, 
                                       df: pd.DataFrame, 
                                       date_column: str) -> pd.DataFrame:
        """Create temporal features using Polars."""
        # Convert to Polars
        pl_df = pl.from_pandas(df)
        
        # Ensure datetime type
        if date_column in pl_df.columns:
            # Check if already datetime, if not convert
            if pl_df[date_column].dtype != pl.Datetime:
                try:
                    pl_df = pl_df.with_columns(
                        pl.col(date_column).str.strptime(pl.Datetime, format=None, strict=False)
                    )
                except:
                    # If string parsing fails, try direct conversion
                    pl_df = pl_df.with_columns(
                        pl.col(date_column).cast(pl.Datetime)
                    )
        
        # Create temporal features efficiently
        temporal_df = pl_df.with_columns([
            # Basic temporal features
            pl.col(date_column).dt.week().alias('semana_ano'),
            pl.col(date_column).dt.month().alias('mes'),
            pl.col(date_column).dt.quarter().alias('trimestre'),
            pl.col(date_column).dt.year().alias('ano'),
            pl.col(date_column).dt.weekday().alias('dia_semana'),
            pl.col(date_column).dt.day().alias('dia_mes'),
            pl.col(date_column).dt.ordinal_day().alias('dia_ano'),
            
            # Cyclical encoding
            (2 * np.pi * pl.col(date_column).dt.week() / 52).sin().alias('semana_sin'),
            (2 * np.pi * pl.col(date_column).dt.week() / 52).cos().alias('semana_cos'),
            (2 * np.pi * pl.col(date_column).dt.month() / 12).sin().alias('mes_sin'),
            (2 * np.pi * pl.col(date_column).dt.month() / 12).cos().alias('mes_cos'),
            
            # Binary indicators
            (pl.col(date_column).dt.day() <= 7).cast(pl.Int32).alias('is_inicio_mes'),
            (pl.col(date_column).dt.day() > 21).cast(pl.Int32).alias('is_fim_mes'),
            (pl.col(date_column).dt.weekday() >= 5).cast(pl.Int32).alias('is_weekend'),
            (pl.col(date_column).dt.month() == 12).cast(pl.Int32).alias('is_dezembro'),
            (pl.col(date_column).dt.month() == 1).cast(pl.Int32).alias('is_janeiro'),
        ])
        
        return temporal_df.to_pandas()
    
    def _create_temporal_features_pandas(self, 
                                       df: pd.DataFrame, 
                                       date_column: str) -> pd.DataFrame:
        """Create temporal features using Pandas (fallback)."""
        df_temporal = df.copy()
        
        # Ensure datetime type
        df_temporal[date_column] = pd.to_datetime(df_temporal[date_column])
        
        # Basic temporal features
        df_temporal['semana_ano'] = df_temporal[date_column].dt.isocalendar().week
        df_temporal['mes'] = df_temporal[date_column].dt.month
        df_temporal['trimestre'] = df_temporal[date_column].dt.quarter
        df_temporal['ano'] = df_temporal[date_column].dt.year
        df_temporal['dia_semana'] = df_temporal[date_column].dt.dayofweek
        df_temporal['dia_mes'] = df_temporal[date_column].dt.day
        df_temporal['dia_ano'] = df_temporal[date_column].dt.dayofyear
        
        # Cyclical encoding
        df_temporal['semana_sin'] = np.sin(2 * np.pi * df_temporal['semana_ano'] / 52)
        df_temporal['semana_cos'] = np.cos(2 * np.pi * df_temporal['semana_ano'] / 52)
        df_temporal['mes_sin'] = np.sin(2 * np.pi * df_temporal['mes'] / 12)
        df_temporal['mes_cos'] = np.cos(2 * np.pi * df_temporal['mes'] / 12)
        
        # Binary indicators
        df_temporal['is_inicio_mes'] = (df_temporal['dia_mes'] <= 7).astype(int)
        df_temporal['is_fim_mes'] = (df_temporal['dia_mes'] > 21).astype(int)
        df_temporal['is_weekend'] = (df_temporal['dia_semana'] >= 5).astype(int)
        df_temporal['is_dezembro'] = (df_temporal['mes'] == 12).astype(int)
        df_temporal['is_janeiro'] = (df_temporal['mes'] == 1).astype(int)
        
        return df_temporal
    
    def create_lag_features_optimized(self, 
                                    df: pd.DataFrame,
                                    target_column: str = 'quantidade',
                                    group_columns: List[str] = None,
                                    lag_periods: List[int] = None) -> pd.DataFrame:
        """Create lag features with parallel processing."""
        if group_columns is None:
            group_columns = ['pdv', 'produto']
        
        if lag_periods is None:
            lag_periods = [1, 2, 4, 8]
        
        params = {
            'target_column': target_column,
            'group_columns': group_columns,
            'lag_periods': lag_periods
        }
        
        # Check cache
        if self.use_cache:
            cached = self.cache.get_cached_features(df, 'lag', params)
            if cached is not None:
                return cached
        
        self.logger.info(f"Creating lag features for periods: {lag_periods}")
        
        if self.use_polars:
            result_df = self._create_lag_features_polars(df, target_column, group_columns, lag_periods)
        else:
            result_df = self._create_lag_features_pandas(df, target_column, group_columns, lag_periods)
        
        # Cache result
        if self.use_cache:
            self.cache.save_features(df, result_df, 'lag', params)
        
        return result_df
    
    def _create_lag_features_polars(self, 
                                  df: pd.DataFrame,
                                  target_column: str,
                                  group_columns: List[str],
                                  lag_periods: List[int]) -> pd.DataFrame:
        """Create lag features using Polars."""
        pl_df = pl.from_pandas(df)
        
        # Sort by group columns and date
        sort_columns = group_columns + ['data_semana'] if 'data_semana' in df.columns else group_columns
        pl_df = pl_df.sort(sort_columns)
        
        # Create lag features
        lag_expressions = []
        for lag in lag_periods:
            lag_expressions.append(
                pl.col(target_column).shift(lag).over(group_columns).alias(f'lag_{lag}_{target_column}')
            )
        
        # Apply all lag operations at once
        result_df = pl_df.with_columns(lag_expressions)
        
        return result_df.to_pandas()
    
    def _create_lag_features_pandas(self, 
                                  df: pd.DataFrame,
                                  target_column: str,
                                  group_columns: List[str],
                                  lag_periods: List[int]) -> pd.DataFrame:
        """Create lag features using Pandas with parallel processing."""
        df_lag = df.copy()
        
        # Sort data
        sort_columns = group_columns + ['data_semana'] if 'data_semana' in df.columns else group_columns
        df_lag = df_lag.sort_values(sort_columns).reset_index(drop=True)
        
        # Create lag features in parallel
        def create_single_lag(lag_period):
            return df_lag.groupby(group_columns)[target_column].shift(lag_period)
        
        with ThreadPoolExecutor(max_workers=min(len(lag_periods), self.max_workers)) as executor:
            future_to_lag = {
                executor.submit(create_single_lag, lag): lag 
                for lag in lag_periods
            }
            
            for future in as_completed(future_to_lag):
                lag = future_to_lag[future]
                try:
                    lag_series = future.result()
                    df_lag[f'lag_{lag}_{target_column}'] = lag_series
                except Exception as e:
                    self.logger.warning(f"Failed to create lag {lag}: {e}")
        
        return df_lag
    
    def create_rolling_features_optimized(self, 
                                        df: pd.DataFrame,
                                        target_column: str = 'quantidade',
                                        group_columns: List[str] = None,
                                        windows: List[int] = None) -> pd.DataFrame:
        """Create rolling statistics features with optimization."""
        if group_columns is None:
            group_columns = ['pdv', 'produto']
        
        if windows is None:
            windows = [4, 8, 12]
        
        params = {
            'target_column': target_column,
            'group_columns': group_columns,
            'windows': windows
        }
        
        # Check cache
        if self.use_cache:
            cached = self.cache.get_cached_features(df, 'rolling', params)
            if cached is not None:
                return cached
        
        self.logger.info(f"Creating rolling features for windows: {windows}")
        
        if self.use_polars:
            result_df = self._create_rolling_features_polars(df, target_column, group_columns, windows)
        else:
            result_df = self._create_rolling_features_pandas(df, target_column, group_columns, windows)
        
        # Cache result
        if self.use_cache:
            self.cache.save_features(df, result_df, 'rolling', params)
        
        return result_df
    
    def _create_rolling_features_polars(self, 
                                      df: pd.DataFrame,
                                      target_column: str,
                                      group_columns: List[str],
                                      windows: List[int]) -> pd.DataFrame:
        """Create rolling features using Polars."""
        pl_df = pl.from_pandas(df)
        
        # Sort by group columns and date
        sort_columns = group_columns + ['data_semana'] if 'data_semana' in df.columns else group_columns
        pl_df = pl_df.sort(sort_columns)
        
        # Create rolling features
        rolling_expressions = []
        for window in windows:
            rolling_expressions.extend([
                pl.col(target_column).rolling_mean(window).over(group_columns).alias(f'rolling_mean_{window}_{target_column}'),
                pl.col(target_column).rolling_std(window).over(group_columns).alias(f'rolling_std_{window}_{target_column}'),
                pl.col(target_column).rolling_max(window).over(group_columns).alias(f'rolling_max_{window}_{target_column}'),
                pl.col(target_column).rolling_min(window).over(group_columns).alias(f'rolling_min_{window}_{target_column}'),
            ])
        
        # Apply all rolling operations at once
        result_df = pl_df.with_columns(rolling_expressions)
        
        return result_df.to_pandas()
    
    def _create_rolling_features_pandas(self, 
                                      df: pd.DataFrame,
                                      target_column: str,
                                      group_columns: List[str],
                                      windows: List[int]) -> pd.DataFrame:
        """Create rolling features using Pandas."""
        df_rolling = df.copy()
        
        # Sort data
        sort_columns = group_columns + ['data_semana'] if 'data_semana' in df.columns else group_columns
        df_rolling = df_rolling.sort_values(sort_columns).reset_index(drop=True)
        
        # Create rolling features for each window
        for window in windows:
            grouped = df_rolling.groupby(group_columns)[target_column]
            
            df_rolling[f'rolling_mean_{window}_{target_column}'] = grouped.rolling(window, min_periods=1).mean().reset_index(level=group_columns, drop=True)
            df_rolling[f'rolling_std_{window}_{target_column}'] = grouped.rolling(window, min_periods=1).std().reset_index(level=group_columns, drop=True)
            df_rolling[f'rolling_max_{window}_{target_column}'] = grouped.rolling(window, min_periods=1).max().reset_index(level=group_columns, drop=True)
            df_rolling[f'rolling_min_{window}_{target_column}'] = grouped.rolling(window, min_periods=1).min().reset_index(level=group_columns, drop=True)
        
        return df_rolling
    
    def create_product_features_optimized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create product features with optimization."""
        params = {}
        
        # Check cache
        if self.use_cache:
            cached = self.cache.get_cached_features(df, 'product', params)
            if cached is not None:
                return cached
        
        self.logger.info("Creating optimized product features")
        
        # Use existing implementation but with caching
        from .engineering import FeatureEngineer
        base_engineer = FeatureEngineer(self.country_code)
        result_df = base_engineer.create_product_features(df)
        
        # Cache result
        if self.use_cache:
            self.cache.save_features(df, result_df, 'product', params)
        
        return result_df
    
    def create_store_features_optimized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create store features with optimization."""
        params = {}
        
        # Check cache
        if self.use_cache:
            cached = self.cache.get_cached_features(df, 'store', params)
            if cached is not None:
                return cached
        
        self.logger.info("Creating optimized store features")
        
        # Use existing implementation but with caching
        from .engineering import FeatureEngineer
        base_engineer = FeatureEngineer(self.country_code)
        result_df = base_engineer.create_store_features(df)
        
        # Cache result
        if self.use_cache:
            self.cache.save_features(df, result_df, 'store', params)
        
        return result_df