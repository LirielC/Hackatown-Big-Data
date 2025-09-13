"""
Feature engineering module for Hackathon Forecast Model 2025.

This module handles creation of temporal, product, store, and lag features
for the sales forecasting model.
"""

import logging
import warnings
from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import holidays


class FeatureEngineeringError(Exception):
    """Custom exception for feature engineering errors."""
    pass


class FeatureEngineer:
    """
    Handles creation of features for sales forecasting model.
    
    This class provides methods to create temporal features, product/store features,
    lag features, and rolling statistics.
    """
    
    def __init__(self, country_code: str = 'BR'):
        """
        Initialize FeatureEngineer instance.
        
        Args:
            country_code: Country code for holiday calendar (default: Brazil)
        """
        self.country_code = country_code
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        self._setup_holidays()
        
    def _setup_logging(self) -> None:
        """Configure logging for feature engineering operations."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _setup_holidays(self) -> None:
        """Setup holiday calendar for the specified country."""
        try:
            if self.country_code == 'BR':
                self.holiday_calendar = holidays.Brazil()
            else:
                self.holiday_calendar = holidays.country_holidays(self.country_code)
            self.logger.info(f"Holiday calendar initialized for {self.country_code}")
        except Exception as e:
            self.logger.warning(f"Could not initialize holiday calendar: {e}")
            self.holiday_calendar = {}
    
    def create_temporal_features(self, df: pd.DataFrame, date_column: str = 'data_semana') -> pd.DataFrame:
        """
        Create comprehensive temporal features from date column.
        
        Args:
            df: DataFrame with date column
            date_column: Name of the date column to use
            
        Returns:
            DataFrame with temporal features added
            
        Raises:
            FeatureEngineeringError: If temporal feature creation fails
        """
        self.logger.info(f"Creating temporal features from column '{date_column}'")
        
        try:
            df_temporal = df.copy()
            
            if date_column not in df_temporal.columns:
                raise FeatureEngineeringError(f"Date column '{date_column}' not found in DataFrame")
            
            # Ensure datetime type
            df_temporal[date_column] = pd.to_datetime(df_temporal[date_column])
            
            # Basic temporal features
            df_temporal = self._create_basic_temporal_features(df_temporal, date_column)
            
            # Seasonality features
            df_temporal = self._create_seasonality_features(df_temporal, date_column)
            
            # Holiday and special event features
            df_temporal = self._create_holiday_features(df_temporal, date_column)
            
            # Trend features
            df_temporal = self._create_trend_features(df_temporal, date_column)
            
            self.logger.info("Temporal features created successfully")
            
            return df_temporal
            
        except Exception as e:
            self.logger.error(f"Temporal feature creation failed: {str(e)}")
            raise FeatureEngineeringError(f"Temporal feature creation failed: {str(e)}")
    
    def _create_basic_temporal_features(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """Create basic temporal features like week, month, quarter."""
        df_basic = df.copy()
        
        # Week-based features
        df_basic['semana_ano'] = df_basic[date_column].dt.isocalendar().week
        df_basic['mes'] = df_basic[date_column].dt.month
        df_basic['trimestre'] = df_basic[date_column].dt.quarter
        df_basic['ano'] = df_basic[date_column].dt.year
        
        # Day-based features (even for weekly data, useful for reference)
        df_basic['dia_semana'] = df_basic[date_column].dt.dayofweek
        df_basic['dia_mes'] = df_basic[date_column].dt.day
        df_basic['dia_ano'] = df_basic[date_column].dt.dayofyear
        
        # Week position in month
        df_basic['semana_mes'] = df_basic[date_column].dt.day // 7 + 1
        
        # Binary indicators
        df_basic['is_inicio_mes'] = (df_basic['dia_mes'] <= 7).astype(int)
        df_basic['is_meio_mes'] = ((df_basic['dia_mes'] > 7) & (df_basic['dia_mes'] <= 21)).astype(int)
        df_basic['is_fim_mes'] = (df_basic['dia_mes'] > 21).astype(int)
        df_basic['is_weekend'] = (df_basic['dia_semana'] >= 5).astype(int)
        
        self.logger.info("Basic temporal features created")
        
        return df_basic
    
    def _create_seasonality_features(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """Create seasonality features using cyclical encoding."""
        df_seasonal = df.copy()
        
        # Cyclical encoding for temporal features to capture seasonality
        # Week of year (52 weeks)
        df_seasonal['semana_sin'] = np.sin(2 * np.pi * df_seasonal[date_column].dt.isocalendar().week / 52)
        df_seasonal['semana_cos'] = np.cos(2 * np.pi * df_seasonal[date_column].dt.isocalendar().week / 52)
        
        # Month (12 months)
        df_seasonal['mes_sin'] = np.sin(2 * np.pi * df_seasonal['mes'] / 12)
        df_seasonal['mes_cos'] = np.cos(2 * np.pi * df_seasonal['mes'] / 12)
        
        # Day of week (7 days)
        df_seasonal['dia_semana_sin'] = np.sin(2 * np.pi * df_seasonal['dia_semana'] / 7)
        df_seasonal['dia_semana_cos'] = np.cos(2 * np.pi * df_seasonal['dia_semana'] / 7)
        
        # Quarter seasonality
        df_seasonal['trimestre_sin'] = np.sin(2 * np.pi * df_seasonal['trimestre'] / 4)
        df_seasonal['trimestre_cos'] = np.cos(2 * np.pi * df_seasonal['trimestre'] / 4)
        
        # Seasonal indicators
        df_seasonal['is_q1'] = (df_seasonal['trimestre'] == 1).astype(int)
        df_seasonal['is_q2'] = (df_seasonal['trimestre'] == 2).astype(int)
        df_seasonal['is_q3'] = (df_seasonal['trimestre'] == 3).astype(int)
        df_seasonal['is_q4'] = (df_seasonal['trimestre'] == 4).astype(int)
        
        # Month indicators for retail seasonality
        df_seasonal['is_janeiro'] = (df_seasonal['mes'] == 1).astype(int)  # Post-holiday
        df_seasonal['is_dezembro'] = (df_seasonal['mes'] == 12).astype(int)  # Holiday season
        df_seasonal['is_junho_julho'] = ((df_seasonal['mes'] == 6) | (df_seasonal['mes'] == 7)).astype(int)  # Winter vacation
        
        self.logger.info("Seasonality features created")
        
        return df_seasonal
    
    def _create_holiday_features(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """Create holiday and special event features."""
        df_holiday = df.copy()

        # Holiday indicators
        df_holiday['is_feriado'] = df_holiday[date_column].apply(
            lambda x: 1 if x.date() in self.holiday_calendar else 0
        )

        # Days before/after holidays - usando abordagem mais segura
        df_holiday['dias_ate_feriado'] = 0
        df_holiday['dias_pos_feriado'] = 0

        # Usar .iterrows() para evitar problemas de indexação
        for idx, row in df_holiday.iterrows():
            date_val = row[date_column]

            # Find next holiday within 30 days
            for days_ahead in range(1, 31):
                future_date = date_val + timedelta(days=days_ahead)
                if future_date.date() in self.holiday_calendar:
                    df_holiday.at[idx, 'dias_ate_feriado'] = days_ahead
                    break

            # Find previous holiday within 30 days
            for days_behind in range(1, 31):
                past_date = date_val - timedelta(days=days_behind)
                if past_date.date() in self.holiday_calendar:
                    df_holiday.at[idx, 'dias_pos_feriado'] = days_behind
                    break
        
        # Holiday proximity indicators
        df_holiday['is_pre_feriado'] = (df_holiday['dias_ate_feriado'] <= 7).astype(int)
        df_holiday['is_pos_feriado'] = (df_holiday['dias_pos_feriado'] <= 7).astype(int)
        
        # Special retail periods
        df_holiday['is_black_friday'] = self._is_black_friday(df_holiday[date_column])
        df_holiday['is_natal'] = self._is_christmas_period(df_holiday[date_column])
        df_holiday['is_volta_aulas'] = self._is_back_to_school(df_holiday[date_column])
        df_holiday['is_dia_maes'] = self._is_mothers_day(df_holiday[date_column])
        
        self.logger.info("Holiday features created")
        
        return df_holiday
    
    def _create_trend_features(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """Create trend-based temporal features."""
        df_trend = df.copy()
        
        # Sort by date to ensure proper trend calculation
        df_trend = df_trend.sort_values(date_column).reset_index(drop=True)
        
        # Linear trend (days since start of dataset)
        min_date = df_trend[date_column].min()
        df_trend['dias_desde_inicio'] = (df_trend[date_column] - min_date).dt.days
        
        # Week number since start
        df_trend['semanas_desde_inicio'] = df_trend['dias_desde_inicio'] // 7
        
        # Normalized trend (0 to 1)
        max_days = df_trend['dias_desde_inicio'].max()
        df_trend['trend_normalizado'] = df_trend['dias_desde_inicio'] / max_days if max_days > 0 else 0
        
        # Quadratic trend for non-linear patterns
        df_trend['trend_quadratico'] = df_trend['trend_normalizado'] ** 2
        
        # Seasonal trend within year
        df_trend['trend_anual'] = df_trend['dia_ano'] / 365.25
        
        self.logger.info("Trend features created")
        
        return df_trend
    
    def _is_black_friday(self, dates: pd.Series) -> pd.Series:
        """Identify Black Friday period (last Friday of November)."""
        black_friday_indicator = []
        
        for date_val in dates:
            # Black Friday is the 4th Thursday of November in the US, but in Brazil it's usually the last Friday
            if date_val.month == 11:
                # Find last Friday of November
                last_day = 30
                while last_day > 0:
                    test_date = date_val.replace(day=last_day)
                    if test_date.weekday() == 4:  # Friday
                        # Black Friday week
                        if abs((date_val - test_date).days) <= 7:
                            black_friday_indicator.append(1)
                        else:
                            black_friday_indicator.append(0)
                        break
                    last_day -= 1
                else:
                    black_friday_indicator.append(0)
            else:
                black_friday_indicator.append(0)
        
        return pd.Series(black_friday_indicator, index=dates.index)
    
    def _is_christmas_period(self, dates: pd.Series) -> pd.Series:
        """Identify Christmas shopping period (December)."""
        return (dates.dt.month == 12).astype(int)
    
    def _is_back_to_school(self, dates: pd.Series) -> pd.Series:
        """Identify back-to-school period (January-February)."""
        return ((dates.dt.month == 1) | (dates.dt.month == 2)).astype(int)
    
    def _is_mothers_day(self, dates: pd.Series) -> pd.Series:
        """Identify Mother's Day period (May in Brazil)."""
        return (dates.dt.month == 5).astype(int)
    
    def create_product_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create product-based features.
        
        Args:
            df: DataFrame with product information
            
        Returns:
            DataFrame with product features added
            
        Raises:
            FeatureEngineeringError: If product feature creation fails
        """
        self.logger.info("Creating product features")
        
        try:
            df_product = df.copy()
            
            # Product category features
            df_product = self._create_product_category_features(df_product)
            
            # Product performance features
            df_product = self._create_product_performance_features(df_product)
            
            # Product ranking features
            df_product = self._create_product_ranking_features(df_product)
            
            self.logger.info("Product features created successfully")
            
            return df_product
            
        except Exception as e:
            self.logger.error(f"Product feature creation failed: {str(e)}")
            raise FeatureEngineeringError(f"Product feature creation failed: {str(e)}")
    
    def create_store_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create store/PDV-based features.
        
        Args:
            df: DataFrame with store information
            
        Returns:
            DataFrame with store features added
            
        Raises:
            FeatureEngineeringError: If store feature creation fails
        """
        self.logger.info("Creating store features")
        
        try:
            df_store = df.copy()
            
            # Store type and location features
            df_store = self._create_store_type_features(df_store)
            
            # Store performance features
            df_store = self._create_store_performance_features(df_store)
            
            # Store ranking features
            df_store = self._create_store_ranking_features(df_store)
            
            self.logger.info("Store features created successfully")
            
            return df_store
            
        except Exception as e:
            self.logger.error(f"Store feature creation failed: {str(e)}")
            raise FeatureEngineeringError(f"Store feature creation failed: {str(e)}")
    
    def _create_product_category_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on product categories."""
        df_cat = df.copy()
        
        # One-hot encoding for product categories if available
        if 'categoria' in df_cat.columns:
            # Get top categories to avoid too many features
            top_categories = df_cat['categoria'].value_counts().head(10).index
            
            for category in top_categories:
                df_cat[f'categoria_{category.lower().replace(" ", "_")}'] = (
                    df_cat['categoria'] == category
                ).astype(int)
            
            # Create "other" category for less frequent categories
            df_cat['categoria_outros'] = (
                ~df_cat['categoria'].isin(top_categories)
            ).astype(int)
        
        # Subcategory features if available
        if 'subcategoria' in df_cat.columns:
            # Count of subcategories per product
            subcategory_counts = df_cat.groupby('produto')['subcategoria'].nunique()
            df_cat['produto_num_subcategorias'] = df_cat['produto'].map(subcategory_counts)
        
        # Brand features if available
        if 'marca' in df_cat.columns:
            # Brand popularity (number of products per brand)
            brand_counts = df_cat.groupby('marca')['produto'].nunique()
            df_cat['marca_num_produtos'] = df_cat['marca'].map(brand_counts)
            
            # Top brands indicator
            top_brands = brand_counts.nlargest(20).index
            df_cat['is_marca_top'] = df_cat['marca'].isin(top_brands).astype(int)
        
        # Price-based category features if available
        if 'preco_unitario' in df_cat.columns:
            # Price quartiles
            price_quartiles = df_cat['preco_unitario'].quantile([0.25, 0.5, 0.75])
            
            df_cat['is_produto_barato'] = (
                df_cat['preco_unitario'] <= price_quartiles[0.25]
            ).astype(int)
            df_cat['is_produto_medio'] = (
                (df_cat['preco_unitario'] > price_quartiles[0.25]) & 
                (df_cat['preco_unitario'] <= price_quartiles[0.75])
            ).astype(int)
            df_cat['is_produto_caro'] = (
                df_cat['preco_unitario'] > price_quartiles[0.75]
            ).astype(int)
        
        self.logger.info("Product category features created")
        
        return df_cat
    
    def _create_product_performance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on historical product performance."""
        df_perf = df.copy()
        
        if 'quantidade' not in df_perf.columns:
            self.logger.warning("Quantity column not found for product performance features")
            return df_perf
        
        # Product-level aggregations
        product_stats = df_perf.groupby('produto')['quantidade'].agg([
            'mean', 'median', 'std', 'min', 'max', 'sum', 'count'
        ]).reset_index()
        
        product_stats.columns = ['produto', 'produto_qty_media', 'produto_qty_mediana', 
                               'produto_qty_std', 'produto_qty_min', 'produto_qty_max',
                               'produto_qty_total', 'produto_num_vendas']
        
        # Calculate coefficient of variation (volatility)
        product_stats['produto_qty_cv'] = (
            product_stats['produto_qty_std'] / product_stats['produto_qty_media']
        ).fillna(0)
        
        # Merge back to main dataframe
        df_perf = df_perf.merge(product_stats, on='produto', how='left')
        
        # Product velocity (sales frequency)
        if 'data_semana' in df_perf.columns:
            # Calculate weeks between sales for each product
            product_velocity = df_perf.groupby('produto')['data_semana'].agg([
                lambda x: (x.max() - x.min()).days / 7 if len(x) > 1 else 0,  # weeks span
                'nunique'  # unique weeks with sales
            ]).reset_index()
            
            product_velocity.columns = ['produto', 'produto_semanas_span', 'produto_semanas_ativas']
            
            # Calculate velocity ratio
            product_velocity['produto_velocidade'] = (
                product_velocity['produto_semanas_ativas'] / 
                (product_velocity['produto_semanas_span'] + 1)
            ).fillna(0)
            
            df_perf = df_perf.merge(product_velocity, on='produto', how='left')
        
        self.logger.info("Product performance features created")
        
        return df_perf
    
    def _create_product_ranking_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ranking and percentile features for products."""
        df_rank = df.copy()

        if 'quantidade' not in df_rank.columns:
            return df_rank

        try:
            # Overall product rankings
            product_totals = df_rank.groupby('produto')['quantidade'].sum().reset_index()
            product_totals['produto_rank_geral'] = product_totals['quantidade'].rank(
                method='dense', ascending=False
            )
            product_totals['produto_percentil_geral'] = product_totals['quantidade'].rank(
                pct=True, method='average'
            )

            # Merge rankings
            df_rank = df_rank.merge(
                product_totals[['produto', 'produto_rank_geral', 'produto_percentil_geral']],
                on='produto', how='left'
            )

            # Top performer indicators
            df_rank['is_produto_top10'] = (df_rank['produto_rank_geral'] <= 10).astype(int)
            df_rank['is_produto_top100'] = (df_rank['produto_rank_geral'] <= 100).astype(int)
            df_rank['is_produto_top_percentil'] = (df_rank['produto_percentil_geral'] >= 0.9).astype(int)

            self.logger.info("Product ranking features created")

        except Exception as e:
            self.logger.warning(f"Product ranking features failed: {e}. Using simplified version.")

            # Fallback: criar features básicas sem rankings complexos
            df_rank['produto_rank_geral'] = 999  # valor padrão
            df_rank['produto_percentil_geral'] = 0.0  # valor padrão
            df_rank['is_produto_top10'] = 0
            df_rank['is_produto_top100'] = 0
            df_rank['is_produto_top_percentil'] = 0

        return df_rank
    
    def _create_store_type_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on store type and location."""
        df_store = df.copy()
        
        # Store type encoding if available
        if 'premise' in df_store.columns:
            # One-hot encoding for store types
            store_types = df_store['premise'].unique()
            for store_type in store_types:
                if pd.notna(store_type):
                    df_store[f'store_type_{store_type.lower().replace("-", "_")}'] = (
                        df_store['premise'] == store_type
                    ).astype(int)
        
        # Category PDV features if available
        if 'categoria_pdv' in df_store.columns:
            pdv_categories = df_store['categoria_pdv'].unique()
            for category in pdv_categories:
                if pd.notna(category):
                    df_store[f'pdv_categoria_{category.lower().replace(" ", "_")}'] = (
                        df_store['categoria_pdv'] == category
                    ).astype(int)
        
        # Location features based on zipcode if available
        if 'zipcode' in df_store.columns:
            # Create region features based on zipcode patterns
            df_store['zipcode_str'] = df_store['zipcode'].astype(str)
            
            # First digit of zipcode often indicates region
            df_store['regiao_zipcode'] = df_store['zipcode_str'].str[0]
            
            # Create region indicators
            for region in df_store['regiao_zipcode'].unique():
                if pd.notna(region) and region != 'n':  # 'n' from 'nan'
                    df_store[f'regiao_{region}'] = (
                        df_store['regiao_zipcode'] == region
                    ).astype(int)
            
            # Urban vs suburban based on zipcode density
            zipcode_counts = df_store['zipcode'].value_counts()
            df_store['zipcode_densidade'] = df_store['zipcode'].map(zipcode_counts)
            
            # High density areas (urban)
            density_threshold = df_store['zipcode_densidade'].quantile(0.75)
            df_store['is_area_urbana'] = (
                df_store['zipcode_densidade'] >= density_threshold
            ).astype(int)
        
        self.logger.info("Store type features created")
        
        return df_store
    
    def _create_store_performance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on historical store performance."""
        df_perf = df.copy()
        
        if 'quantidade' not in df_perf.columns:
            self.logger.warning("Quantity column not found for store performance features")
            return df_perf
        
        # Store-level aggregations
        store_stats = df_perf.groupby('pdv')['quantidade'].agg([
            'mean', 'median', 'std', 'min', 'max', 'sum', 'count'
        ]).reset_index()
        
        store_stats.columns = ['pdv', 'pdv_qty_media', 'pdv_qty_mediana', 
                             'pdv_qty_std', 'pdv_qty_min', 'pdv_qty_max',
                             'pdv_qty_total', 'pdv_num_vendas']
        
        # Calculate coefficient of variation (volatility)
        store_stats['pdv_qty_cv'] = (
            store_stats['pdv_qty_std'] / store_stats['pdv_qty_media']
        ).fillna(0)
        
        # Merge back to main dataframe
        df_perf = df_perf.merge(store_stats, on='pdv', how='left')
        
        # Store diversity (number of unique products sold)
        store_diversity = df_perf.groupby('pdv')['produto'].nunique().reset_index()
        store_diversity.columns = ['pdv', 'pdv_num_produtos']
        df_perf = df_perf.merge(store_diversity, on='pdv', how='left')
        
        # Store activity (weeks with sales)
        if 'data_semana' in df_perf.columns:
            store_activity = df_perf.groupby('pdv')['data_semana'].agg([
                lambda x: (x.max() - x.min()).days / 7 if len(x) > 1 else 0,  # weeks span
                'nunique'  # unique weeks with sales
            ]).reset_index()
            
            store_activity.columns = ['pdv', 'pdv_semanas_span', 'pdv_semanas_ativas']
            
            # Calculate activity ratio
            store_activity['pdv_atividade'] = (
                store_activity['pdv_semanas_ativas'] / 
                (store_activity['pdv_semanas_span'] + 1)
            ).fillna(0)
            
            df_perf = df_perf.merge(store_activity, on='pdv', how='left')
        
        self.logger.info("Store performance features created")
        
        return df_perf
    
    def _create_store_ranking_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ranking and percentile features for stores."""
        df_rank = df.copy()

        if 'quantidade' not in df_rank.columns:
            return df_rank

        try:
            # Overall store rankings
            store_totals = df_rank.groupby('pdv')['quantidade'].sum().reset_index()
            store_totals['pdv_rank_geral'] = store_totals['quantidade'].rank(
                method='dense', ascending=False
            )
            store_totals['pdv_percentil_geral'] = store_totals['quantidade'].rank(
                pct=True, method='average'
            )

            # Merge rankings
            df_rank = df_rank.merge(
                store_totals[['pdv', 'pdv_rank_geral', 'pdv_percentil_geral']],
                on='pdv', how='left'
            )

            # Top performer indicators
            df_rank['is_pdv_top10'] = (df_rank['pdv_rank_geral'] <= 10).astype(int)
            df_rank['is_pdv_top100'] = (df_rank['pdv_rank_geral'] <= 100).astype(int)
            df_rank['is_pdv_top_percentil'] = (df_rank['pdv_percentil_geral'] >= 0.9).astype(int)

            self.logger.info("Store ranking features created")

        except Exception as e:
            self.logger.warning(f"Store ranking features failed: {e}. Using simplified version.")

            # Fallback: criar features básicas sem rankings complexos
            df_rank['pdv_rank_geral'] = 999  # valor padrão
            df_rank['pdv_percentil_geral'] = 0.0  # valor padrão
            df_rank['pdv_rank_tipo'] = 999  # valor padrão
            df_rank['pdv_percentil_tipo'] = 0.0  # valor padrão
            df_rank['is_pdv_top10'] = 0
            df_rank['is_pdv_top100'] = 0
            df_rank['is_pdv_top_percentil'] = 0

        return df_rank
    
    def create_lag_features(self, df: pd.DataFrame, 
                           target_column: str = 'quantidade',
                           date_column: str = 'data_semana',
                           group_columns: List[str] = None,
                           lag_periods: List[int] = None) -> pd.DataFrame:
        """
        Create lag features for time series data.
        
        Args:
            df: DataFrame with time series data
            target_column: Column to create lags for
            date_column: Date column for sorting
            group_columns: Columns to group by (e.g., ['pdv', 'produto'])
            lag_periods: List of lag periods to create (default: [1, 2, 4, 8])
            
        Returns:
            DataFrame with lag features added
            
        Raises:
            FeatureEngineeringError: If lag feature creation fails
        """
        self.logger.info(f"Creating lag features for column '{target_column}'")
        
        try:
            df_lag = df.copy()
            
            if target_column not in df_lag.columns:
                raise FeatureEngineeringError(f"Target column '{target_column}' not found in DataFrame")
            
            if date_column not in df_lag.columns:
                raise FeatureEngineeringError(f"Date column '{date_column}' not found in DataFrame")
            
            # Default parameters
            if group_columns is None:
                group_columns = ['pdv', 'produto']
            
            if lag_periods is None:
                lag_periods = [1, 2, 4, 8]
            
            # Ensure date column is datetime
            df_lag[date_column] = pd.to_datetime(df_lag[date_column])
            
            # Sort by group columns and date
            sort_columns = group_columns + [date_column]
            df_lag = df_lag.sort_values(sort_columns).reset_index(drop=True)
            
            # Create lag features
            for lag in lag_periods:
                lag_col_name = f'{target_column}_lag_{lag}'
                df_lag[lag_col_name] = df_lag.groupby(group_columns)[target_column].shift(lag)
                
                self.logger.info(f"Created lag feature: {lag_col_name}")
            
            # Create lag ratios (current vs previous periods)
            if 1 in lag_periods:
                df_lag[f'{target_column}_ratio_lag1'] = (
                    df_lag[target_column] / (df_lag[f'{target_column}_lag_1'] + 1e-8)
                ).fillna(1.0)
            
            if 2 in lag_periods:
                df_lag[f'{target_column}_ratio_lag2'] = (
                    df_lag[target_column] / (df_lag[f'{target_column}_lag_2'] + 1e-8)
                ).fillna(1.0)
            
            # Create lag differences (growth)
            if 1 in lag_periods:
                df_lag[f'{target_column}_diff_lag1'] = (
                    df_lag[target_column] - df_lag[f'{target_column}_lag_1']
                ).fillna(0)
            
            if 4 in lag_periods:
                df_lag[f'{target_column}_diff_lag4'] = (
                    df_lag[target_column] - df_lag[f'{target_column}_lag_4']
                ).fillna(0)
            
            self.logger.info("Lag features created successfully")
            
            return df_lag
            
        except Exception as e:
            self.logger.error(f"Lag feature creation failed: {str(e)}")
            raise FeatureEngineeringError(f"Lag feature creation failed: {str(e)}")
    
    def create_rolling_features(self, df: pd.DataFrame,
                               target_column: str = 'quantidade',
                               date_column: str = 'data_semana',
                               group_columns: List[str] = None,
                               window_sizes: List[int] = None) -> pd.DataFrame:
        """
        Create rolling statistics features.
        
        Args:
            df: DataFrame with time series data
            target_column: Column to create rolling features for
            date_column: Date column for sorting
            group_columns: Columns to group by (e.g., ['pdv', 'produto'])
            window_sizes: List of window sizes for rolling statistics (default: [4, 8, 12])
            
        Returns:
            DataFrame with rolling features added
            
        Raises:
            FeatureEngineeringError: If rolling feature creation fails
        """
        self.logger.info(f"Creating rolling features for column '{target_column}'")
        
        try:
            df_rolling = df.copy()
            
            if target_column not in df_rolling.columns:
                raise FeatureEngineeringError(f"Target column '{target_column}' not found in DataFrame")
            
            if date_column not in df_rolling.columns:
                raise FeatureEngineeringError(f"Date column '{date_column}' not found in DataFrame")
            
            # Default parameters
            if group_columns is None:
                group_columns = ['pdv', 'produto']
            
            if window_sizes is None:
                window_sizes = [4, 8, 12]
            
            # Ensure date column is datetime
            df_rolling[date_column] = pd.to_datetime(df_rolling[date_column])
            
            # Sort by group columns and date
            sort_columns = group_columns + [date_column]
            df_rolling = df_rolling.sort_values(sort_columns).reset_index(drop=True)
            
            # Create rolling features for each window size
            for window in window_sizes:
                # Rolling mean
                df_rolling[f'{target_column}_rolling_mean_{window}'] = (
                    df_rolling.groupby(group_columns)[target_column]
                    .rolling(window=window, min_periods=1)
                    .mean()
                    .reset_index(level=group_columns, drop=True)
                )
                
                # Rolling standard deviation
                df_rolling[f'{target_column}_rolling_std_{window}'] = (
                    df_rolling.groupby(group_columns)[target_column]
                    .rolling(window=window, min_periods=1)
                    .std()
                    .reset_index(level=group_columns, drop=True)
                    .fillna(0)
                )
                
                # Rolling min and max
                df_rolling[f'{target_column}_rolling_min_{window}'] = (
                    df_rolling.groupby(group_columns)[target_column]
                    .rolling(window=window, min_periods=1)
                    .min()
                    .reset_index(level=group_columns, drop=True)
                )
                
                df_rolling[f'{target_column}_rolling_max_{window}'] = (
                    df_rolling.groupby(group_columns)[target_column]
                    .rolling(window=window, min_periods=1)
                    .max()
                    .reset_index(level=group_columns, drop=True)
                )
                
                # Rolling median
                df_rolling[f'{target_column}_rolling_median_{window}'] = (
                    df_rolling.groupby(group_columns)[target_column]
                    .rolling(window=window, min_periods=1)
                    .median()
                    .reset_index(level=group_columns, drop=True)
                )
                
                self.logger.info(f"Created rolling features for window size: {window}")
            
            # Create volatility features (coefficient of variation)
            for window in window_sizes:
                mean_col = f'{target_column}_rolling_mean_{window}'
                std_col = f'{target_column}_rolling_std_{window}'
                
                df_rolling[f'{target_column}_rolling_cv_{window}'] = (
                    df_rolling[std_col] / (df_rolling[mean_col] + 1e-8)
                ).fillna(0)
            
            # Create trend features (slope of rolling mean)
            for window in window_sizes:
                mean_col = f'{target_column}_rolling_mean_{window}'
                
                # Calculate trend as difference between current and previous rolling mean
                df_rolling[f'{target_column}_rolling_trend_{window}'] = (
                    df_rolling.groupby(group_columns)[mean_col].diff().fillna(0)
                )
            
            self.logger.info("Rolling features created successfully")
            
            return df_rolling
            
        except Exception as e:
            self.logger.error(f"Rolling feature creation failed: {str(e)}")
            raise FeatureEngineeringError(f"Rolling feature creation failed: {str(e)}")
    
    def create_growth_features(self, df: pd.DataFrame,
                              target_column: str = 'quantidade',
                              date_column: str = 'data_semana',
                              group_columns: List[str] = None) -> pd.DataFrame:
        """
        Create growth and percentage change features.
        
        Args:
            df: DataFrame with time series data
            target_column: Column to create growth features for
            date_column: Date column for sorting
            group_columns: Columns to group by (e.g., ['pdv', 'produto'])
            
        Returns:
            DataFrame with growth features added
            
        Raises:
            FeatureEngineeringError: If growth feature creation fails
        """
        self.logger.info(f"Creating growth features for column '{target_column}'")
        
        try:
            df_growth = df.copy()
            
            if target_column not in df_growth.columns:
                raise FeatureEngineeringError(f"Target column '{target_column}' not found in DataFrame")
            
            if date_column not in df_growth.columns:
                raise FeatureEngineeringError(f"Date column '{date_column}' not found in DataFrame")
            
            # Default parameters
            if group_columns is None:
                group_columns = ['pdv', 'produto']
            
            # Ensure date column is datetime
            df_growth[date_column] = pd.to_datetime(df_growth[date_column])
            
            # Sort by group columns and date
            sort_columns = group_columns + [date_column]
            df_growth = df_growth.sort_values(sort_columns).reset_index(drop=True)
            
            # Week-over-week growth
            df_growth[f'{target_column}_pct_change_1w'] = (
                df_growth.groupby(group_columns)[target_column].pct_change(periods=1).fillna(0)
            )
            
            # Month-over-month growth (4 weeks)
            df_growth[f'{target_column}_pct_change_4w'] = (
                df_growth.groupby(group_columns)[target_column].pct_change(periods=4).fillna(0)
            )
            
            # Quarter-over-quarter growth (12 weeks)
            df_growth[f'{target_column}_pct_change_12w'] = (
                df_growth.groupby(group_columns)[target_column].pct_change(periods=12).fillna(0)
            )
            
            # Cumulative growth from start
            df_growth[f'{target_column}_cumulative_growth'] = (
                df_growth.groupby(group_columns)[target_column].apply(
                    lambda x: (x / x.iloc[0] - 1) if len(x) > 0 and x.iloc[0] != 0 else 0
                ).reset_index(level=group_columns, drop=True).fillna(0)
            )
            
            # Growth acceleration (change in growth rate)
            df_growth[f'{target_column}_growth_acceleration'] = (
                df_growth.groupby(group_columns)[f'{target_column}_pct_change_1w'].diff().fillna(0)
            )
            
            # Growth volatility (rolling std of growth rates)
            df_growth[f'{target_column}_growth_volatility'] = (
                df_growth.groupby(group_columns)[f'{target_column}_pct_change_1w']
                .rolling(window=8, min_periods=1)
                .std()
                .reset_index(level=group_columns, drop=True)
                .fillna(0)
            )
            
            self.logger.info("Growth features created successfully")
            
            return df_growth
            
        except Exception as e:
            self.logger.error(f"Growth feature creation failed: {str(e)}")
            raise FeatureEngineeringError(f"Growth feature creation failed: {str(e)}")
    
    def create_all_lag_and_rolling_features(self, df: pd.DataFrame,
                                           target_column: str = 'quantidade',
                                           date_column: str = 'data_semana',
                                           group_columns: List[str] = None) -> pd.DataFrame:
        """
        Create all lag, rolling, and growth features in one call.
        
        Args:
            df: DataFrame with time series data
            target_column: Column to create features for
            date_column: Date column for sorting
            group_columns: Columns to group by (e.g., ['pdv', 'produto'])
            
        Returns:
            DataFrame with all lag and rolling features added
        """
        self.logger.info("Creating all lag and rolling features")
        
        # Default parameters
        if group_columns is None:
            group_columns = ['pdv', 'produto']
        
        # Create features step by step
        df_features = df.copy()
        
        # Lag features
        df_features = self.create_lag_features(
            df_features, target_column, date_column, group_columns
        )
        
        # Rolling features
        df_features = self.create_rolling_features(
            df_features, target_column, date_column, group_columns
        )
        
        # Growth features
        df_features = self.create_growth_features(
            df_features, target_column, date_column, group_columns
        )
        
        self.logger.info("All lag and rolling features created successfully")
        
        return df_features
    
    def create_all_features(self, df: pd.DataFrame,
                           target_column: str = 'quantidade',
                           date_column: str = 'data_semana',
                           group_columns: List[str] = None) -> pd.DataFrame:
        """
        Create all feature types in the correct order.
        
        Args:
            df: DataFrame with time series data
            target_column: Column to create features for
            date_column: Date column for sorting
            group_columns: Columns to group by (e.g., ['pdv', 'produto'])
            
        Returns:
            DataFrame with all features added
        """
        self.logger.info("Creating all features for sales forecasting")
        
        # Default parameters
        if group_columns is None:
            group_columns = ['pdv', 'produto']
        
        df_all_features = df.copy()
        
        # 1. Temporal features (independent of other features)
        df_all_features = self.create_temporal_features(df_all_features, date_column)
        
        # 2. Product features (can use existing columns)
        df_all_features = self.create_product_features(df_all_features)
        
        # 3. Store features (can use existing columns)
        df_all_features = self.create_store_features(df_all_features)
        
        # 4. Lag and rolling features (depend on target column)
        df_all_features = self.create_all_lag_and_rolling_features(
            df_all_features, target_column, date_column, group_columns
        )
        
        self.logger.info(f"All features created successfully. Final shape: {df_all_features.shape}")
        
        return df_all_features
    
    def get_feature_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary of created features.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Dictionary with feature summary
        """
        feature_summary = {
            'total_features': len(df.columns),
            'feature_types': {},
            'missing_values': {},
            'data_types': {}
        }
        
        # Categorize features by type
        feature_types = {
            'temporal': [],
            'product': [],
            'store': [],
            'lag': [],
            'rolling': [],
            'growth': [],
            'original': []
        }
        
        for col in df.columns:
            col_lower = col.lower()
            if any(x in col_lower for x in ['semana', 'mes', 'trimestre', 'dia', 'feriado', 'natal', 'trend']):
                feature_types['temporal'].append(col)
            elif any(x in col_lower for x in ['produto', 'categoria', 'marca', 'preco']):
                feature_types['product'].append(col)
            elif any(x in col_lower for x in ['pdv', 'store', 'premise', 'zipcode', 'regiao']):
                feature_types['store'].append(col)
            elif 'lag' in col_lower:
                feature_types['lag'].append(col)
            elif 'rolling' in col_lower:
                feature_types['rolling'].append(col)
            elif any(x in col_lower for x in ['pct_change', 'growth', 'cumulative']):
                feature_types['growth'].append(col)
            else:
                feature_types['original'].append(col)
        
        # Count features by type
        for feature_type, features in feature_types.items():
            feature_summary['feature_types'][feature_type] = len(features)
        
        # Check missing values
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                feature_summary['missing_values'][col] = missing_count
        
        # Data types
        for dtype in df.dtypes.unique():
            feature_summary['data_types'][str(dtype)] = (df.dtypes == dtype).sum()
        
        return feature_summary