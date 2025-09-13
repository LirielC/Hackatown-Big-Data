"""
Data processing modules for Hackathon Forecast Model 2025.

This module provides data ingestion capabilities for loading and validating
Parquet files containing transaction, product, and store data.
"""

from .ingestion import DataIngestion, DataIngestionError

__all__ = ['DataIngestion', 'DataIngestionError']