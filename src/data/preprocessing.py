"""
Data preprocessing module for Hackathon Forecast Model 2025.

This module handles data cleaning, temporal aggregation, and merging of
transaction, product, and store data.
"""

import logging
import warnings
from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import polars as pl


class DataPreprocessingError(Exception):
    """Custom exception for data preprocessing errors."""
    pass


class DataPreprocessor:
    """
    Handles data cleaning, aggregation, and merging operations.
    
    This class provides methods to clean transaction data, aggregate from daily
    to weekly frequency, and merge with master data (products and stores).
    """
    
    def __init__(self, use_polars: bool = False):
        """
        Initialize DataPreprocessor instance.
        
        Args:
            use_polars: Whether to use Polars for better performance on large datasets
        """
        self.use_polars = use_polars
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
    def _setup_logging(self) -> None:
        """Configure logging for preprocessing operations."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def clean_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize transaction data.
        
        Args:
            df: Raw transaction DataFrame
            
        Returns:
            Cleaned transaction DataFrame
            
        Raises:
            DataPreprocessingError: If critical cleaning operations fail
        """
        self.logger.info(f"Starting transaction cleaning for {len(df)} records")
        
        try:
            df_clean = df.copy()
            
            # Standardize column names to match expected schema
            column_mapping = {
                'internal_store_id': 'pdv',
                'internal_product_id': 'produto', 
                'transaction_date': 'data',
                'quantity': 'quantidade'
            }
            
            # Rename columns if they exist
            for old_col, new_col in column_mapping.items():
                if old_col in df_clean.columns:
                    df_clean = df_clean.rename(columns={old_col: new_col})
            
            # Ensure required columns exist
            required_columns = ['pdv', 'produto', 'data', 'quantidade']
            missing_columns = [col for col in required_columns if col not in df_clean.columns]
            if missing_columns:
                raise DataPreprocessingError(f"Missing required columns after mapping: {missing_columns}")
            
            # Convert data types
            df_clean = self._convert_data_types(df_clean)
            
            # Handle missing values
            df_clean = self._handle_missing_values(df_clean)
            
            # Remove invalid records
            df_clean = self._remove_invalid_records(df_clean)
            
            # Handle outliers
            df_clean = self._handle_outliers(df_clean)
            
            self.logger.info(f"Transaction cleaning completed. Records: {len(df)} -> {len(df_clean)}")
            
            return df_clean
            
        except Exception as e:
            self.logger.error(f"Transaction cleaning failed: {str(e)}")
            raise DataPreprocessingError(f"Transaction cleaning failed: {str(e)}")
    
    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert columns to appropriate data types."""
        df_typed = df.copy()
        
        # Convert date column
        if 'data' in df_typed.columns:
            try:
                df_typed['data'] = pd.to_datetime(df_typed['data'])
                self.logger.info("Date column converted successfully")
            except Exception as e:
                self.logger.warning(f"Date conversion failed: {e}")
        
        # Convert numeric columns
        numeric_columns = ['quantidade', 'gross_value', 'net_value', 'gross_profit', 'discount', 'taxes']
        for col in numeric_columns:
            if col in df_typed.columns:
                try:
                    df_typed[col] = pd.to_numeric(df_typed[col], errors='coerce')
                except Exception as e:
                    self.logger.warning(f"Numeric conversion failed for {col}: {e}")
        
        # Convert ID columns to string (to handle large integers)
        id_columns = ['pdv', 'produto']
        for col in id_columns:
            if col in df_typed.columns:
                df_typed[col] = df_typed[col].astype(str)
        
        return df_typed
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in transaction data."""
        df_filled = df.copy()
        
        # Log missing value counts
        missing_counts = df_filled.isnull().sum()
        if missing_counts.sum() > 0:
            self.logger.info(f"Missing values found:\n{missing_counts[missing_counts > 0]}")
        
        # Handle missing quantities - critical field
        if 'quantidade' in df_filled.columns:
            missing_qty = df_filled['quantidade'].isnull().sum()
            if missing_qty > 0:
                self.logger.warning(f"Found {missing_qty} missing quantities - these records will be removed")
                df_filled = df_filled.dropna(subset=['quantidade'])
        
        # Handle missing dates - critical field
        if 'data' in df_filled.columns:
            missing_dates = df_filled['data'].isnull().sum()
            if missing_dates > 0:
                self.logger.warning(f"Found {missing_dates} missing dates - these records will be removed")
                df_filled = df_filled.dropna(subset=['data'])
        
        # Handle missing IDs - critical fields
        for col in ['pdv', 'produto']:
            if col in df_filled.columns:
                missing_ids = df_filled[col].isnull().sum()
                if missing_ids > 0:
                    self.logger.warning(f"Found {missing_ids} missing {col} - these records will be removed")
                    df_filled = df_filled.dropna(subset=[col])
        
        # Handle missing financial values - fill with 0 or median
        financial_columns = ['gross_value', 'net_value', 'gross_profit', 'discount', 'taxes']
        for col in financial_columns:
            if col in df_filled.columns:
                missing_count = df_filled[col].isnull().sum()
                if missing_count > 0:
                    # Use median for financial values to avoid bias
                    median_value = df_filled[col].median()
                    df_filled[col] = df_filled[col].fillna(median_value)
                    self.logger.info(f"Filled {missing_count} missing values in {col} with median: {median_value}")
        
        return df_filled
    
    def _remove_invalid_records(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove records with invalid data."""
        df_valid = df.copy()
        initial_count = len(df_valid)
        
        # Remove records with negative or zero quantities
        if 'quantidade' in df_valid.columns:
            invalid_qty = (df_valid['quantidade'] <= 0).sum()
            if invalid_qty > 0:
                self.logger.info(f"Removing {invalid_qty} records with invalid quantities")
                df_valid = df_valid[df_valid['quantidade'] > 0]
        
        # Remove records with invalid dates (not in 2022)
        if 'data' in df_valid.columns:
            invalid_dates = ((df_valid['data'].dt.year != 2022)).sum()
            if invalid_dates > 0:
                self.logger.info(f"Removing {invalid_dates} records with dates outside 2022")
                df_valid = df_valid[df_valid['data'].dt.year == 2022]
        
        # Remove duplicate records
        duplicates = df_valid.duplicated().sum()
        if duplicates > 0:
            self.logger.info(f"Removing {duplicates} duplicate records")
            df_valid = df_valid.drop_duplicates()
        
        removed_count = initial_count - len(df_valid)
        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} invalid records ({removed_count/initial_count*100:.2f}%)")
        
        return df_valid
    
    def _handle_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """Handle outliers in quantity data."""
        df_clean = df.copy()
        
        if 'quantidade' not in df_clean.columns:
            return df_clean
        
        initial_count = len(df_clean)
        
        if method == 'iqr':
            # Use IQR method for outlier detection
            Q1 = df_clean['quantidade'].quantile(0.25)
            Q3 = df_clean['quantidade'].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers instead of removing them
            outliers_low = (df_clean['quantidade'] < lower_bound).sum()
            outliers_high = (df_clean['quantidade'] > upper_bound).sum()
            
            if outliers_low > 0 or outliers_high > 0:
                self.logger.info(f"Capping {outliers_low + outliers_high} outliers (low: {outliers_low}, high: {outliers_high})")
                df_clean['quantidade'] = df_clean['quantidade'].clip(lower=max(0, lower_bound), upper=upper_bound)
        
        elif method == 'percentile':
            # Use percentile capping (1% and 99%)
            lower_percentile = df_clean['quantidade'].quantile(0.01)
            upper_percentile = df_clean['quantidade'].quantile(0.99)
            
            outliers = ((df_clean['quantidade'] < lower_percentile) | 
                       (df_clean['quantidade'] > upper_percentile)).sum()
            
            if outliers > 0:
                self.logger.info(f"Capping {outliers} outliers using 1%-99% percentiles")
                df_clean['quantidade'] = df_clean['quantidade'].clip(
                    lower=lower_percentile, upper=upper_percentile
                )
        
        return df_clean
    
    def aggregate_weekly_sales(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate daily transaction data to weekly sales.
        
        Args:
            df: Daily transaction DataFrame
            
        Returns:
            Weekly aggregated DataFrame
            
        Raises:
            DataPreprocessingError: If aggregation fails
        """
        self.logger.info(f"Starting weekly aggregation for {len(df)} daily records")
        
        try:
            if 'data' not in df.columns:
                raise DataPreprocessingError("Date column 'data' not found for aggregation")
            
            df_agg = df.copy()
            
            # Ensure date column is datetime
            df_agg['data'] = pd.to_datetime(df_agg['data'])
            
            # Create week number (ISO week)
            df_agg['ano'] = df_agg['data'].dt.year
            df_agg['semana'] = df_agg['data'].dt.isocalendar().week
            
            # Create week start date for reference
            df_agg['semana_inicio'] = df_agg['data'] - pd.to_timedelta(df_agg['data'].dt.dayofweek, unit='D')
            
            # Define aggregation functions
            agg_functions = {
                'quantidade': 'sum',
                'data': 'min'  # Keep earliest date in the week
            }
            
            # Add financial columns if they exist
            financial_columns = ['gross_value', 'net_value', 'gross_profit', 'discount', 'taxes']
            for col in financial_columns:
                if col in df_agg.columns:
                    agg_functions[col] = 'sum'
            
            # Group by PDV, produto, and week
            groupby_columns = ['pdv', 'produto', 'ano', 'semana']
            
            df_weekly = df_agg.groupby(groupby_columns).agg(agg_functions).reset_index()
            
            # Rename aggregated data column
            df_weekly = df_weekly.rename(columns={'data': 'data_semana'})
            
            # Create a proper week identifier
            df_weekly['semana_ano'] = df_weekly['ano'].astype(str) + '_W' + df_weekly['semana'].astype(str).str.zfill(2)
            
            # Sort by date and identifiers
            df_weekly = df_weekly.sort_values(['pdv', 'produto', 'ano', 'semana']).reset_index(drop=True)
            
            self.logger.info(f"Weekly aggregation completed. Records: {len(df)} -> {len(df_weekly)}")
            self.logger.info(f"Date range: {df_weekly['data_semana'].min()} to {df_weekly['data_semana'].max()}")
            self.logger.info(f"Week range: {df_weekly['semana'].min()} to {df_weekly['semana'].max()}")
            
            return df_weekly
            
        except Exception as e:
            self.logger.error(f"Weekly aggregation failed: {str(e)}")
            raise DataPreprocessingError(f"Weekly aggregation failed: {str(e)}")
    
    def merge_master_data(self, transactions: pd.DataFrame, 
                         products: Optional[pd.DataFrame] = None, 
                         stores: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Merge transaction data with product and store master data.
        
        Args:
            transactions: Transaction DataFrame
            products: Product master DataFrame (optional)
            stores: Store master DataFrame (optional)
            
        Returns:
            Merged DataFrame with master data
            
        Raises:
            DataPreprocessingError: If merge operations fail
        """
        self.logger.info("Starting master data merge")
        
        try:
            df_merged = transactions.copy()
            
            # Merge with store data
            if stores is not None:
                df_merged = self._merge_store_data(df_merged, stores)
            
            # Merge with product data  
            if products is not None:
                df_merged = self._merge_product_data(df_merged, products)
            
            self.logger.info(f"Master data merge completed. Final records: {len(df_merged)}")
            
            return df_merged
            
        except Exception as e:
            self.logger.error(f"Master data merge failed: {str(e)}")
            raise DataPreprocessingError(f"Master data merge failed: {str(e)}")
    
    def _merge_store_data(self, transactions: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
        """Merge transaction data with store master data."""
        stores_clean = stores.copy()
        
        # Standardize store ID column name
        if 'pdv' not in stores_clean.columns:
            # Try to find the store ID column
            possible_store_cols = ['internal_store_id', 'store_id', 'pdv_id']
            store_col = None
            for col in possible_store_cols:
                if col in stores_clean.columns:
                    store_col = col
                    break
            
            if store_col:
                stores_clean = stores_clean.rename(columns={store_col: 'pdv'})
            else:
                self.logger.warning("No store ID column found in store data")
                return transactions
        
        # Ensure consistent data types
        stores_clean['pdv'] = stores_clean['pdv'].astype(str)
        transactions['pdv'] = transactions['pdv'].astype(str)
        
        # Perform left join to keep all transactions
        initial_count = len(transactions)
        df_merged = transactions.merge(stores_clean, on='pdv', how='left', suffixes=('', '_store'))
        
        # Check merge success
        matched_stores = df_merged['pdv'].notna().sum()
        self.logger.info(f"Store merge: {matched_stores}/{initial_count} transactions matched with store data")
        
        # Handle missing store information
        store_info_columns = ['premise', 'categoria_pdv', 'zipcode']
        for col in store_info_columns:
            if col in df_merged.columns:
                missing_count = df_merged[col].isnull().sum()
                if missing_count > 0:
                    # Fill missing categorical data with 'Unknown'
                    if df_merged[col].dtype == 'object':
                        df_merged[col] = df_merged[col].fillna('Unknown')
                    else:
                        # Fill missing numeric data with median
                        median_val = df_merged[col].median()
                        df_merged[col] = df_merged[col].fillna(median_val)
                    
                    self.logger.info(f"Filled {missing_count} missing values in {col}")
        
        return df_merged
    
    def _merge_product_data(self, transactions: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
        """Merge transaction data with product master data."""
        products_clean = products.copy()
        
        # Standardize product ID column name
        if 'produto' not in products_clean.columns:
            # Try to find the product ID column
            possible_product_cols = ['internal_product_id', 'product_id', 'produto_id']
            product_col = None
            for col in possible_product_cols:
                if col in products_clean.columns:
                    product_col = col
                    break
            
            if product_col:
                products_clean = products_clean.rename(columns={product_col: 'produto'})
            else:
                self.logger.warning("No product ID column found in product data")
                return transactions
        
        # Ensure consistent data types
        products_clean['produto'] = products_clean['produto'].astype(str)
        transactions['produto'] = transactions['produto'].astype(str)
        
        # Perform left join to keep all transactions
        initial_count = len(transactions)
        df_merged = transactions.merge(products_clean, on='produto', how='left', suffixes=('', '_product'))
        
        # Check merge success
        matched_products = df_merged['produto'].notna().sum()
        self.logger.info(f"Product merge: {matched_products}/{initial_count} transactions matched with product data")
        
        # Handle missing product information
        product_info_columns = ['categoria', 'subcategoria', 'marca', 'preco_unitario']
        for col in product_info_columns:
            if col in df_merged.columns:
                missing_count = df_merged[col].isnull().sum()
                if missing_count > 0:
                    # Fill missing categorical data with 'Unknown'
                    if df_merged[col].dtype == 'object':
                        df_merged[col] = df_merged[col].fillna('Unknown')
                    else:
                        # Fill missing numeric data with median
                        median_val = df_merged[col].median()
                        df_merged[col] = df_merged[col].fillna(median_val)
                    
                    self.logger.info(f"Filled {missing_count} missing values in {col}")
        
        return df_merged
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional time-based features from date columns.
        
        Args:
            df: DataFrame with date columns
            
        Returns:
            DataFrame with additional time features
        """
        df_time = df.copy()
        
        # Work with the main date column
        date_col = None
        for col in ['data', 'data_semana', 'transaction_date']:
            if col in df_time.columns:
                date_col = col
                break
        
        if date_col is None:
            self.logger.warning("No date column found for time feature creation")
            return df_time
        
        # Ensure datetime type
        df_time[date_col] = pd.to_datetime(df_time[date_col])
        
        # Create time features
        df_time['mes'] = df_time[date_col].dt.month
        df_time['trimestre'] = df_time[date_col].dt.quarter
        df_time['dia_semana'] = df_time[date_col].dt.dayofweek
        df_time['dia_mes'] = df_time[date_col].dt.day
        df_time['semana_mes'] = df_time[date_col].dt.day // 7 + 1
        
        # Create seasonal indicators
        df_time['is_inicio_mes'] = (df_time['dia_mes'] <= 7).astype(int)
        df_time['is_fim_mes'] = (df_time['dia_mes'] >= 24).astype(int)
        df_time['is_weekend'] = (df_time['dia_semana'] >= 5).astype(int)
        
        # Create month names for easier interpretation
        df_time['mes_nome'] = df_time[date_col].dt.strftime('%B')
        
        self.logger.info("Time features created successfully")
        
        return df_time
    
    def validate_processed_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate processed data quality and completeness.
        
        Args:
            df: Processed DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'summary': {}
        }
        
        # Basic data validation
        validation_results['summary']['total_records'] = len(df)
        validation_results['summary']['total_columns'] = len(df.columns)
        
        # Check for required columns
        required_columns = ['pdv', 'produto', 'quantidade']
        missing_required = [col for col in required_columns if col not in df.columns]
        if missing_required:
            validation_results['errors'].append(f"Missing required columns: {missing_required}")
            validation_results['is_valid'] = False
        
        # Check data quality
        if len(df) == 0:
            validation_results['errors'].append("DataFrame is empty")
            validation_results['is_valid'] = False
        
        # Check for null values in critical columns
        for col in required_columns:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    validation_results['warnings'].append(f"Column '{col}' has {null_count} null values")
        
        # Check quantity values
        if 'quantidade' in df.columns:
            negative_qty = (df['quantidade'] < 0).sum()
            zero_qty = (df['quantidade'] == 0).sum()
            
            if negative_qty > 0:
                validation_results['warnings'].append(f"Found {negative_qty} negative quantities")
            if zero_qty > 0:
                validation_results['warnings'].append(f"Found {zero_qty} zero quantities")
            
            validation_results['summary']['avg_quantity'] = df['quantidade'].mean()
            validation_results['summary']['total_quantity'] = df['quantidade'].sum()
        
        # Check date ranges if date columns exist
        date_columns = [col for col in df.columns if 'data' in col.lower() or 'date' in col.lower()]
        for date_col in date_columns:
            try:
                df[date_col] = pd.to_datetime(df[date_col])
                min_date = df[date_col].min()
                max_date = df[date_col].max()
                validation_results['summary'][f'{date_col}_range'] = f"{min_date} to {max_date}"
            except:
                validation_results['warnings'].append(f"Could not parse dates in column {date_col}")
        
        # Check for duplicates
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            validation_results['warnings'].append(f"Found {duplicate_count} duplicate rows")
        
        # Memory usage
        validation_results['summary']['memory_usage_mb'] = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        return validation_results
    
    def get_preprocessing_summary(self, original_df: pd.DataFrame, processed_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary of preprocessing operations.
        
        Args:
            original_df: Original DataFrame before preprocessing
            processed_df: DataFrame after preprocessing
            
        Returns:
            Dictionary with preprocessing summary
        """
        summary = {
            'original_records': len(original_df),
            'processed_records': len(processed_df),
            'records_removed': len(original_df) - len(processed_df),
            'removal_percentage': (len(original_df) - len(processed_df)) / len(original_df) * 100,
            'original_columns': len(original_df.columns),
            'processed_columns': len(processed_df.columns),
            'columns_added': len(processed_df.columns) - len(original_df.columns)
        }
        
        # Memory usage comparison
        original_memory = original_df.memory_usage(deep=True).sum() / 1024 / 1024
        processed_memory = processed_df.memory_usage(deep=True).sum() / 1024 / 1024
        
        summary['original_memory_mb'] = original_memory
        summary['processed_memory_mb'] = processed_memory
        summary['memory_change_mb'] = processed_memory - original_memory
        
        # Data quality improvements
        if 'quantidade' in original_df.columns and 'quantidade' in processed_df.columns:
            original_nulls = original_df['quantidade'].isnull().sum()
            processed_nulls = processed_df['quantidade'].isnull().sum()
            summary['quantity_nulls_removed'] = original_nulls - processed_nulls
        
        return summary