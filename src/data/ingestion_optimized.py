"""
Optimized data ingestion module for Hackathon Forecast Model 2025.

This module provides performance-optimized data loading using Polars
and efficient memory management for large datasets.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp


class OptimizedDataIngestion:
    """
    Performance-optimized data ingestion with Polars and parallel processing.
    
    This class provides methods to efficiently load large Parquet files
    with memory optimization and parallel processing capabilities.
    """
    
    def __init__(self, 
                 use_lazy_loading: bool = True,
                 chunk_size: int = 100000,
                 max_workers: Optional[int] = None):
        """
        Initialize OptimizedDataIngestion instance.
        
        Args:
            use_lazy_loading: Whether to use lazy evaluation for better memory efficiency
            chunk_size: Size of chunks for processing large files
            max_workers: Maximum number of worker threads (default: CPU count)
        """
        self.use_lazy_loading = use_lazy_loading
        self.chunk_size = chunk_size
        self.max_workers = max_workers or min(mp.cpu_count(), 8)
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
    def _setup_logging(self) -> None:
        """Configure logging for data ingestion operations."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def load_transactions_optimized(self, path: Union[str, Path]) -> pd.DataFrame:
        """
        Load transaction data with Polars optimization.
        
        Args:
            path: Path to Parquet file or directory containing Parquet files
            
        Returns:
            DataFrame with transaction data
        """
        self.logger.info(f"Loading transaction data (optimized) from: {path}")
        
        path = Path(path)
        
        if path.is_file():
            return self._load_single_file_optimized(path)
        elif path.is_dir():
            return self._load_directory_optimized(path)
        else:
            raise FileNotFoundError(f"Path does not exist: {path}")
    
    def _load_single_file_optimized(self, file_path: Path) -> pd.DataFrame:
        """Load single Parquet file with Polars optimization."""
        self.logger.info(f"Loading single file: {file_path.name}")
        
        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        self.logger.info(f"File size: {file_size_mb:.2f} MB")
        
        if self.use_lazy_loading and file_size_mb > 100:
            # Use lazy loading for large files
            df_lazy = pl.scan_parquet(file_path)
            
            # Apply basic optimizations
            df_lazy = self._apply_lazy_optimizations(df_lazy)
            
            # Collect to pandas
            df = df_lazy.collect().to_pandas()
        else:
            # Direct loading for smaller files
            df = pl.read_parquet(file_path).to_pandas()
        
        self.logger.info(f"Loaded {len(df)} records from {file_path.name}")
        return df
    
    def _load_directory_optimized(self, dir_path: Path) -> pd.DataFrame:
        """Load multiple Parquet files with parallel processing."""
        parquet_files = list(dir_path.glob("*.parquet"))
        
        if not parquet_files:
            raise FileNotFoundError(f"No Parquet files found in directory: {dir_path}")
        
        self.logger.info(f"Found {len(parquet_files)} Parquet files")
        
        # Load files in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(self._load_single_file_optimized, file): file 
                for file in parquet_files
            }
            
            dataframes = []
            for future in as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    df = future.result()
                    dataframes.append(df)
                except Exception as e:
                    self.logger.warning(f"Failed to load {file}: {str(e)}")
        
        if not dataframes:
            raise RuntimeError("No files could be loaded successfully")
        
        # Combine dataframes efficiently
        self.logger.info("Combining loaded dataframes")
        combined_df = pd.concat(dataframes, ignore_index=True)
        
        self.logger.info(f"Combined dataset: {len(combined_df)} records")
        return combined_df
    
    def _apply_lazy_optimizations(self, df_lazy: pl.LazyFrame) -> pl.LazyFrame:
        """Apply Polars lazy optimizations."""
        # Basic optimizations that can be applied lazily
        optimized = df_lazy
        
        # Filter out null values in critical columns if they exist
        if 'quantidade' in df_lazy.columns:
            optimized = optimized.filter(pl.col('quantidade').is_not_null())
        
        if 'data' in df_lazy.columns:
            optimized = optimized.filter(pl.col('data').is_not_null())
        
        # Sort by date for better compression and access patterns
        if 'data' in df_lazy.columns:
            optimized = optimized.sort('data')
        
        return optimized
    
    def load_with_schema_detection(self, path: Union[str, Path]) -> Dict[str, pd.DataFrame]:
        """
        Load files with automatic schema detection and grouping.
        
        Args:
            path: Path to directory containing Parquet files
            
        Returns:
            Dictionary mapping schema signatures to DataFrames
        """
        path = Path(path)
        parquet_files = list(path.glob("*.parquet"))
        
        if not parquet_files:
            raise FileNotFoundError(f"No Parquet files found in directory: {path}")
        
        self.logger.info(f"Detecting schemas for {len(parquet_files)} files")
        
        # Detect schemas in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(self._detect_file_schema, file): file 
                for file in parquet_files
            }
            
            schema_groups = {}
            for future in as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    schema_sig, sample_df = future.result()
                    
                    if schema_sig not in schema_groups:
                        schema_groups[schema_sig] = []
                    
                    schema_groups[schema_sig].append((file, sample_df))
                    
                except Exception as e:
                    self.logger.warning(f"Failed to detect schema for {file}: {str(e)}")
        
        # Load full files for each schema group
        result = {}
        for schema_sig, file_info_list in schema_groups.items():
            self.logger.info(f"Loading {len(file_info_list)} files with schema: {schema_sig}")
            
            files_to_load = [file_info[0] for file_info in file_info_list]
            
            # Load files in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_file = {
                    executor.submit(self._load_single_file_optimized, file): file 
                    for file in files_to_load
                }
                
                dataframes = []
                for future in as_completed(future_to_file):
                    try:
                        df = future.result()
                        dataframes.append(df)
                    except Exception as e:
                        self.logger.warning(f"Failed to load file: {str(e)}")
            
            if dataframes:
                result[schema_sig] = pd.concat(dataframes, ignore_index=True)
        
        return result
    
    def _detect_file_schema(self, file_path: Path) -> tuple:
        """Detect schema of a single file."""
        # Read just the first few rows to detect schema
        df_sample = pl.scan_parquet(file_path).head(100).collect()
        schema_sig = tuple(sorted(df_sample.columns))
        
        return schema_sig, df_sample.to_pandas()
    
    def load_chunked(self, 
                    path: Union[str, Path],
                    chunk_processor: callable = None) -> pd.DataFrame:
        """
        Load large files in chunks for memory efficiency.
        
        Args:
            path: Path to Parquet file
            chunk_processor: Optional function to process each chunk
            
        Returns:
            Processed DataFrame
        """
        path = Path(path)
        
        if not path.is_file():
            raise ValueError("Chunked loading only supports single files")
        
        self.logger.info(f"Loading file in chunks: {path.name}")
        
        # Use PyArrow for chunked reading
        parquet_file = pq.ParquetFile(path)
        
        processed_chunks = []
        
        for batch in parquet_file.iter_batches(batch_size=self.chunk_size):
            # Convert to pandas
            chunk_df = batch.to_pandas()
            
            # Apply chunk processor if provided
            if chunk_processor:
                chunk_df = chunk_processor(chunk_df)
            
            processed_chunks.append(chunk_df)
            
            self.logger.info(f"Processed chunk with {len(chunk_df)} records")
        
        # Combine all chunks
        result_df = pd.concat(processed_chunks, ignore_index=True)
        
        self.logger.info(f"Chunked loading complete: {len(result_df)} total records")
        
        return result_df
    
    def get_file_info(self, path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get detailed information about Parquet files without loading them.
        
        Args:
            path: Path to file or directory
            
        Returns:
            Dictionary with file information
        """
        path = Path(path)
        
        if path.is_file():
            return self._get_single_file_info(path)
        elif path.is_dir():
            return self._get_directory_info(path)
        else:
            raise FileNotFoundError(f"Path does not exist: {path}")
    
    def _get_single_file_info(self, file_path: Path) -> Dict[str, Any]:
        """Get information about a single Parquet file."""
        try:
            parquet_file = pq.ParquetFile(file_path)
            
            info = {
                'file_path': str(file_path),
                'file_size_mb': file_path.stat().st_size / (1024 * 1024),
                'num_rows': parquet_file.metadata.num_rows,
                'num_columns': parquet_file.metadata.num_columns,
                'num_row_groups': parquet_file.metadata.num_row_groups,
                'schema': [field.name for field in parquet_file.schema],
                'compression': parquet_file.metadata.row_group(0).column(0).compression
            }
            
            return info
            
        except Exception as e:
            return {
                'file_path': str(file_path),
                'error': str(e)
            }
    
    def _get_directory_info(self, dir_path: Path) -> Dict[str, Any]:
        """Get information about all Parquet files in a directory."""
        parquet_files = list(dir_path.glob("*.parquet"))
        
        total_size = 0
        total_rows = 0
        file_infos = []
        
        for file in parquet_files:
            file_info = self._get_single_file_info(file)
            file_infos.append(file_info)
            
            if 'file_size_mb' in file_info:
                total_size += file_info['file_size_mb']
            if 'num_rows' in file_info:
                total_rows += file_info['num_rows']
        
        return {
            'directory_path': str(dir_path),
            'num_files': len(parquet_files),
            'total_size_mb': total_size,
            'total_rows': total_rows,
            'files': file_infos
        }
    
    def preprocess_and_save(self, 
                           input_path: Union[str, Path],
                           output_path: Union[str, Path],
                           preprocessing_func: callable) -> str:
        """
        Load, preprocess, and save data in optimized format.
        
        Args:
            input_path: Path to input data
            output_path: Path to save processed data
            preprocessing_func: Function to apply preprocessing
            
        Returns:
            Path to saved file
        """
        self.logger.info("Loading data for preprocessing")
        
        # Load data optimized
        df = self.load_transactions_optimized(input_path)
        
        # Apply preprocessing
        self.logger.info("Applying preprocessing")
        processed_df = preprocessing_func(df)
        
        # Save in optimized format
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to Polars for efficient saving
        pl_df = pl.from_pandas(processed_df)
        
        # Save as Parquet with compression
        pl_df.write_parquet(
            output_path,
            compression='snappy',
            use_pyarrow=True
        )
        
        self.logger.info(f"Preprocessed data saved to: {output_path}")
        
        return str(output_path)