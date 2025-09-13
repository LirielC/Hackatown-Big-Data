"""
Data ingestion module for Hackathon Forecast Model 2025.

This module handles loading and validation of Parquet files containing
transaction, product, and store data.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
from datetime import datetime


class DataIngestionError(Exception):
    """Custom exception for data ingestion errors."""
    pass


class DataIngestion:
    """
    Handles loading and validation of Parquet data files.
    
    This class provides methods to load transaction, product, and store data
    from Parquet files with built-in validation and quality checks.
    """
    
    def __init__(self, use_polars: bool = False):
        """
        Initialize DataIngestion instance.
        
        Args:
            use_polars: Whether to use Polars for better performance on large datasets
        """
        self.use_polars = use_polars
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
    
    def load_transactions(self, path: Union[str, Path]) -> pd.DataFrame:
        """
        Load transaction data from Parquet file(s).
        
        Args:
            path: Path to Parquet file or directory containing Parquet files
            
        Returns:
            DataFrame with transaction data
            
        Raises:
            DataIngestionError: If file loading or validation fails
        """
        self.logger.info(f"Loading transaction data from: {path}")
        
        try:
            df = self._load_parquet_data(path)
            self.logger.info(f"Loaded {len(df)} transaction records")
            
            # Validate transaction data structure
            validation_result = self.validate_transaction_data(df)
            if not validation_result['is_valid']:
                raise DataIngestionError(f"Transaction data validation failed: {validation_result['errors']}")
            
            self.logger.info("Transaction data validation passed")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load transaction data: {str(e)}")
            raise DataIngestionError(f"Transaction data loading failed: {str(e)}")
    
    def load_products(self, path: Union[str, Path]) -> pd.DataFrame:
        """
        Load product master data from Parquet file(s).
        
        Args:
            path: Path to Parquet file or directory containing Parquet files
            
        Returns:
            DataFrame with product data
            
        Raises:
            DataIngestionError: If file loading or validation fails
        """
        self.logger.info(f"Loading product data from: {path}")
        
        try:
            df = self._load_parquet_data(path)
            self.logger.info(f"Loaded {len(df)} product records")
            
            # Validate product data structure
            validation_result = self.validate_product_data(df)
            if not validation_result['is_valid']:
                raise DataIngestionError(f"Product data validation failed: {validation_result['errors']}")
            
            self.logger.info("Product data validation passed")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load product data: {str(e)}")
            raise DataIngestionError(f"Product data loading failed: {str(e)}")
    
    def load_stores(self, path: Union[str, Path]) -> pd.DataFrame:
        """
        Load store/PDV master data from Parquet file(s).
        
        Args:
            path: Path to Parquet file or directory containing Parquet files
            
        Returns:
            DataFrame with store data
            
        Raises:
            DataIngestionError: If file loading or validation fails
        """
        self.logger.info(f"Loading store data from: {path}")
        
        try:
            df = self._load_parquet_data(path)
            self.logger.info(f"Loaded {len(df)} store records")
            
            # Validate store data structure
            validation_result = self.validate_store_data(df)
            if not validation_result['is_valid']:
                raise DataIngestionError(f"Store data validation failed: {validation_result['errors']}")
            
            self.logger.info("Store data validation passed")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load store data: {str(e)}")
            raise DataIngestionError(f"Store data loading failed: {str(e)}")
    
    def _load_parquet_data(self, path: Union[str, Path]) -> pd.DataFrame:
        """
        Load Parquet data using either Pandas or Polars.
        
        Args:
            path: Path to Parquet file or directory
            
        Returns:
            DataFrame with loaded data
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        
        if path.is_file():
            # Single file
            if self.use_polars:
                df_pl = pl.read_parquet(path)
                return df_pl.to_pandas()
            else:
                return pd.read_parquet(path)
        
        elif path.is_dir():
            # Directory with multiple Parquet files
            parquet_files = list(path.glob("*.parquet"))
            if not parquet_files:
                raise FileNotFoundError(f"No Parquet files found in directory: {path}")
            
            self.logger.info(f"Found {len(parquet_files)} Parquet files")
            
            if self.use_polars:
                # Use Polars for better performance with multiple files
                dfs = []
                for file in parquet_files:
                    try:
                        df = pl.read_parquet(file)
                        dfs.append(df)
                    except Exception as e:
                        self.logger.warning(f"Failed to load {file}: {str(e)}")
                        continue
                
                if not dfs:
                    raise DataIngestionError("No files could be loaded successfully")
                
                # Check if all dataframes have the same schema
                if len(set(tuple(df.columns) for df in dfs)) > 1:
                    self.logger.warning("Files have different schemas, loading separately")
                    # Return the first successfully loaded dataframe
                    return dfs[0].to_pandas()
                
                combined_df = pl.concat(dfs)
                return combined_df.to_pandas()
            else:
                # Use Pandas
                dfs = []
                for file in parquet_files:
                    try:
                        df = pd.read_parquet(file)
                        dfs.append(df)
                    except Exception as e:
                        self.logger.warning(f"Failed to load {file}: {str(e)}")
                        continue
                
                if not dfs:
                    raise DataIngestionError("No files could be loaded successfully")
                
                # Check if all dataframes have the same columns
                if len(set(tuple(df.columns) for df in dfs)) > 1:
                    self.logger.warning("Files have different schemas, loading separately")
                    # Return the first successfully loaded dataframe
                    return dfs[0]
                
                return pd.concat(dfs, ignore_index=True)
        
        else:
            raise ValueError(f"Path is neither file nor directory: {path}")
    
    def validate_transaction_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate transaction data quality and structure.
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            Dictionary with validation results
        """
        errors = []
        warnings = []
        
        # Check required columns for transactions (with mapping)
        column_mapping = {
            'transaction_date': 'data',
            'internal_store_id': 'pdv',
            'internal_product_id': 'produto',
            'quantity': 'quantidade'
        }

        # Map columns if they exist with different names
        df_mapped = df.copy()
        for original_col, expected_col in column_mapping.items():
            if original_col in df_mapped.columns and expected_col not in df_mapped.columns:
                df_mapped[expected_col] = df_mapped[original_col]
                df_mapped = df_mapped.drop(columns=[original_col])

        required_columns = ['data', 'pdv', 'produto', 'quantidade']
        missing_columns = [col for col in required_columns if col not in df_mapped.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        # Check data types and ranges
        if 'quantidade' in df_mapped.columns:
            if df_mapped['quantidade'].dtype not in ['int32', 'int64', 'float32', 'float64']:
                warnings.append("Quantidade column should be numeric")

            negative_quantities = (df_mapped['quantidade'] < 0).sum()
            if negative_quantities > 0:
                warnings.append(f"Found {negative_quantities} negative quantities")

        # Check for null values in critical columns
        for col in required_columns:
            if col in df_mapped.columns:
                null_count = df_mapped[col].isnull().sum()
                if null_count > 0:
                    warnings.append(f"Column '{col}' has {null_count} null values")

        # Check date range (should be 2022 data)
        if 'data' in df_mapped.columns:
            try:
                df_mapped['data'] = pd.to_datetime(df_mapped['data'])
                min_date = df_mapped['data'].min()
                max_date = df_mapped['data'].max()

                if min_date.year != 2022 or max_date.year != 2022:
                    warnings.append(f"Date range outside 2022: {min_date} to {max_date}")

            except Exception as e:
                errors.append(f"Date column conversion failed: {str(e)}")

        # Check for duplicates
        duplicate_count = df_mapped.duplicated().sum()
        if duplicate_count > 0:
            warnings.append(f"Found {duplicate_count} duplicate rows")

        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'row_count': len(df_mapped),
            'column_count': len(df_mapped.columns),
            'null_counts': df_mapped.isnull().sum().to_dict(),
            'data_types': df_mapped.dtypes.to_dict()
        }
    
    def validate_product_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate product master data quality and structure.
        
        Args:
            df: DataFrame with product data
            
        Returns:
            Dictionary with validation results
        """
        errors = []
        warnings = []
        
        # Check required columns for products
        required_columns = ['produto']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        # Check for unique product IDs
        if 'produto' in df.columns:
            duplicate_products = df['produto'].duplicated().sum()
            if duplicate_products > 0:
                errors.append(f"Found {duplicate_products} duplicate product IDs")
        
        # Check for null values in critical columns
        for col in required_columns:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    warnings.append(f"Column '{col}' has {null_count} null values")
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'row_count': len(df),
            'column_count': len(df.columns),
            'null_counts': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict()
        }
    
    def validate_store_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate store/PDV master data quality and structure.
        
        Args:
            df: DataFrame with store data
            
        Returns:
            Dictionary with validation results
        """
        errors = []
        warnings = []
        
        # Check required columns for stores
        required_columns = ['pdv']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        # Check for unique store IDs
        if 'pdv' in df.columns:
            duplicate_stores = df['pdv'].duplicated().sum()
            if duplicate_stores > 0:
                errors.append(f"Found {duplicate_stores} duplicate store IDs")
        
        # Check for null values in critical columns
        for col in required_columns:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    warnings.append(f"Column '{col}' has {null_count} null values")
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'row_count': len(df),
            'column_count': len(df.columns),
            'null_counts': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict()
        }
    
    def validate_data_quality(self, df: pd.DataFrame, data_type: str = "generic") -> Dict[str, Any]:
        """
        General data quality validation method.
        
        Args:
            df: DataFrame to validate
            data_type: Type of data being validated
            
        Returns:
            Dictionary with validation results
        """
        if data_type == "transactions":
            return self.validate_transaction_data(df)
        elif data_type == "products":
            return self.validate_product_data(df)
        elif data_type == "stores":
            return self.validate_store_data(df)
        else:
            # Generic validation
            return {
                'is_valid': True,
                'errors': [],
                'warnings': [],
                'row_count': len(df),
                'column_count': len(df.columns),
                'null_counts': df.isnull().sum().to_dict(),
                'data_types': df.dtypes.to_dict()
            }
    
    def load_multiple_schemas(self, path: Union[str, Path], sample_only: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Load Parquet files that may have different schemas.
        
        Args:
            path: Path to directory containing Parquet files
            sample_only: If True, load only a sample of each file for schema detection
            
        Returns:
            Dictionary mapping schema signatures to DataFrames
        """
        path = Path(path)
        
        if not path.is_dir():
            raise ValueError("Path must be a directory for loading multiple schemas")
        
        parquet_files = list(path.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No Parquet files found in directory: {path}")
        
        schema_groups = {}
        
        for file in parquet_files:
            try:
                self.logger.info(f"Processing file: {file.name}")
                
                # Check file size first
                file_size_mb = file.stat().st_size / (1024 * 1024)
                self.logger.info(f"File size: {file_size_mb:.2f} MB")
                
                if sample_only or file_size_mb > 500:  # Force sampling for large files
                    # Load only first few rows to detect schema
                    if self.use_polars:
                        df = pl.scan_parquet(file).head(1000).collect().to_pandas()
                    else:
                        # For pandas, try to read with pyarrow for better memory management
                        try:
                            import pyarrow.parquet as pq
                            parquet_file = pq.ParquetFile(file)
                            # Read just first batch
                            batch_iter = parquet_file.iter_batches(batch_size=1000)
                            first_batch = next(batch_iter)
                            df = first_batch.to_pandas()
                        except:
                            # Last resort fallback
                            df = pd.read_parquet(file).head(1000)
                else:
                    if self.use_polars:
                        df = pl.read_parquet(file).to_pandas()
                    else:
                        df = pd.read_parquet(file)
                
                # Create schema signature
                schema_sig = tuple(sorted(df.columns))
                
                if schema_sig not in schema_groups:
                    schema_groups[schema_sig] = []
                
                schema_groups[schema_sig].append(df)
                self.logger.info(f"Loaded {len(df)} records with schema: {schema_sig}")
                
            except Exception as e:
                self.logger.warning(f"Failed to load {file}: {str(e)}")
                continue
        
        # Combine files with same schema
        result = {}
        for schema_sig, dfs in schema_groups.items():
            if len(dfs) == 1:
                result[schema_sig] = dfs[0]
            else:
                try:
                    result[schema_sig] = pd.concat(dfs, ignore_index=True)
                except Exception as e:
                    self.logger.warning(f"Failed to combine files with schema {schema_sig}: {str(e)}")
                    # Keep the first dataframe if concat fails
                    result[schema_sig] = dfs[0]
        
        self.logger.info(f"Loaded {len(result)} different schemas from {len(parquet_files)} files")
        return result

    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive data summary.
        
        Args:
            df: DataFrame to summarize
            
        Returns:
            Dictionary with data summary statistics
        """
        summary = {
            'shape': df.shape,
            'columns': list(df.columns),
            'data_types': df.dtypes.to_dict(),
            'null_counts': df.isnull().sum().to_dict(),
            'null_percentages': (df.isnull().sum() / len(df) * 100).to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
        }
        
        # Add numeric column statistics
        numeric_columns = df.select_dtypes(include=['number']).columns
        if len(numeric_columns) > 0:
            summary['numeric_stats'] = df[numeric_columns].describe().to_dict()
        
        # Add categorical column statistics
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_columns) > 0:
            summary['categorical_stats'] = {}
            for col in categorical_columns:
                summary['categorical_stats'][col] = {
                    'unique_count': df[col].nunique(),
                    'top_values': df[col].value_counts().head(5).to_dict()
                }
        
        return summary