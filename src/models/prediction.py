"""
Módulo de geração de previsões para o Hackathon Forecast Model 2025.

Este módulo implementa a geração de previsões para janeiro/2023,
pós-processamento e validação de integridade das previsões.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from datetime import datetime, timedelta
import warnings
from pathlib import Path
import joblib

from .training import BaseModel
from .ensemble import BaseEnsemble
from .output_formatter import SubmissionFormatter, OutputFormatterError

logger = logging.getLogger(__name__)


class PredictionError(Exception):
    """Exceção customizada para erros de predição."""
    pass


class PredictionGenerator:
    """
    Classe principal para geração de previsões de vendas.
    
    Responsável por gerar previsões para janeiro/2023, aplicar pós-processamento
    e validar a integridade das previsões geradas.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa o gerador de previsões.
        
        Args:
            config: Configuração do modelo e predição
        """
        self.config = config
        self.prediction_config = config.get('prediction', {})
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # Configurações de pós-processamento
        self.ensure_positive = self.prediction_config.get('post_processing', {}).get('ensure_positive', True)
        self.apply_bounds = self.prediction_config.get('post_processing', {}).get('apply_bounds', True)
        self.max_multiplier = self.prediction_config.get('post_processing', {}).get('max_multiplier', 3.0)
        
        # Semanas alvo para predição
        self.target_weeks = self.prediction_config.get('target_weeks', [1, 2, 3, 4, 5])
        
    def _setup_logging(self) -> None:
        """Configura logging para operações de predição."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def generate_predictions(self, 
                           model: Union[BaseModel, BaseEnsemble],
                           features_df: pd.DataFrame,
                           historical_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Gera previsões para janeiro/2023.
        
        Args:
            model: Modelo treinado ou ensemble
            features_df: DataFrame com features para predição
            historical_data: Dados históricos para referência (opcional)
            
        Returns:
            DataFrame com previsões formatadas
            
        Raises:
            PredictionError: Se a geração de previsões falhar
        """
        self.logger.info("Iniciando geração de previsões para janeiro/2023")
        
        try:
            # Validar entrada
            self._validate_prediction_inputs(model, features_df)
            
            # Gerar previsões brutas
            raw_predictions = self._generate_raw_predictions(model, features_df)
            
            # Aplicar pós-processamento
            processed_predictions = self._apply_post_processing(
                raw_predictions, features_df, historical_data
            )
            
            # Formatar saída
            formatted_predictions = self._format_predictions(processed_predictions, features_df)
            
            # Validar integridade
            self._validate_predictions(formatted_predictions)
            
            self.logger.info(f"Previsões geradas com sucesso para {len(formatted_predictions)} registros")
            
            return formatted_predictions
            
        except Exception as e:
            self.logger.error(f"Falha na geração de previsões: {str(e)}")
            raise PredictionError(f"Falha na geração de previsões: {str(e)}")
    
    def _validate_prediction_inputs(self, 
                                  model: Union[BaseModel, BaseEnsemble],
                                  features_df: pd.DataFrame) -> None:
        """Valida entradas para geração de previsões."""
        # Verificar se modelo está treinado
        if hasattr(model, 'is_fitted') and not model.is_fitted:
            raise PredictionError("Modelo deve estar treinado antes de gerar previsões")
        
        # Verificar se DataFrame não está vazio
        if len(features_df) == 0:
            raise PredictionError("DataFrame de features está vazio")
        
        # Verificar colunas essenciais
        required_columns = ['pdv', 'produto', 'semana']
        missing_columns = [col for col in required_columns if col not in features_df.columns]
        if missing_columns:
            raise PredictionError(f"Colunas obrigatórias ausentes: {missing_columns}")
        
        # Verificar se há dados para as semanas alvo
        available_weeks = features_df['semana'].unique()
        missing_weeks = [week for week in self.target_weeks if week not in available_weeks]
        if missing_weeks:
            self.logger.warning(f"Semanas alvo ausentes nos dados: {missing_weeks}")
    
    def _generate_raw_predictions(self, 
                                model: Union[BaseModel, BaseEnsemble],
                                features_df: pd.DataFrame) -> np.ndarray:
        """Gera previsões brutas usando o modelo."""
        self.logger.info("Gerando previsões brutas")
        
        # Preparar features para predição
        prediction_features = self._prepare_prediction_features(features_df)
        
        # Gerar previsões
        predictions = model.predict(prediction_features)
        
        self.logger.info(f"Previsões brutas geradas: min={predictions.min():.2f}, "
                        f"max={predictions.max():.2f}, mean={predictions.mean():.2f}")
        
        return predictions
    
    def _prepare_prediction_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Prepara features para predição, removendo apenas colunas não necessárias."""
        # Colunas que não devem ser usadas como features (mas manter 'semana' pois é uma feature válida)
        exclude_columns = ['pdv', 'produto', 'quantidade', 'data', 'data_semana']

        # Selecionar apenas colunas de features
        feature_columns = [col for col in features_df.columns if col not in exclude_columns]

        # Verificar se há features suficientes
        if len(feature_columns) == 0:
            raise PredictionError("Nenhuma feature válida encontrada para predição")

        prediction_features = features_df[feature_columns].copy()

        # Tratar valores faltantes
        prediction_features = prediction_features.fillna(0)

        self.logger.info(f"Features preparadas: {len(feature_columns)} colunas")
        self.logger.info(f"Features que serão usadas: {sorted(feature_columns)}")

        return prediction_features
    
    def _apply_post_processing(self, 
                             predictions: np.ndarray,
                             features_df: pd.DataFrame,
                             historical_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Aplica pós-processamento às previsões.
        
        Args:
            predictions: Previsões brutas
            features_df: DataFrame com features
            historical_data: Dados históricos para referência
            
        Returns:
            Previsões pós-processadas
        """
        self.logger.info("Aplicando pós-processamento às previsões")
        
        processed_predictions = predictions.copy()
        
        # 1. Garantir valores não-negativos
        if self.ensure_positive:
            negative_count = (processed_predictions < 0).sum()
            if negative_count > 0:
                self.logger.info(f"Convertendo {negative_count} valores negativos para zero")
                processed_predictions = np.maximum(processed_predictions, 0)
        
        # 2. Aplicar limites baseados em dados históricos
        if self.apply_bounds and historical_data is not None:
            processed_predictions = self._apply_historical_bounds(
                processed_predictions, features_df, historical_data
            )
        
        # 3. Aplicar suavização para valores extremos
        processed_predictions = self._apply_smoothing(processed_predictions)
        
        # 4. Arredondar para valores inteiros (quantidades)
        processed_predictions = np.round(processed_predictions).astype(int)
        
        self.logger.info(f"Pós-processamento concluído: min={processed_predictions.min()}, "
                        f"max={processed_predictions.max()}, mean={processed_predictions.mean():.2f}")
        
        return processed_predictions
    
    def _apply_historical_bounds(self, 
                               predictions: np.ndarray,
                               features_df: pd.DataFrame,
                               historical_data: pd.DataFrame) -> np.ndarray:
        """Aplica limites baseados em dados históricos."""
        bounded_predictions = predictions.copy()
        
        # Calcular estatísticas históricas por PDV/produto
        if 'quantidade' in historical_data.columns:
            historical_stats = historical_data.groupby(['pdv', 'produto'])['quantidade'].agg([
                'mean', 'std', 'max', 'quantile'
            ]).reset_index()
            
            # Aplicar limites por PDV/produto
            for idx, row in features_df.iterrows():
                pdv = row['pdv']
                produto = row['produto']
                
                # Encontrar estatísticas históricas
                hist_stats = historical_stats[
                    (historical_stats['pdv'] == pdv) & 
                    (historical_stats['produto'] == produto)
                ]
                
                if len(hist_stats) > 0:
                    hist_mean = hist_stats['mean'].iloc[0]
                    hist_max = hist_stats['max'].iloc[0]
                    
                    # Aplicar limite máximo baseado no histórico
                    max_allowed = min(hist_max * self.max_multiplier, hist_mean * 10)
                    
                    if bounded_predictions[idx] > max_allowed:
                        bounded_predictions[idx] = max_allowed
                else:
                    # Se não há histórico, aplicar limite conservador
                    if bounded_predictions[idx] > 1000:  # Limite arbitrário
                        bounded_predictions[idx] = 1000
        
        return bounded_predictions
    
    def _apply_smoothing(self, predictions: np.ndarray) -> np.ndarray:
        """Aplica suavização para reduzir valores extremos."""
        smoothed_predictions = predictions.copy()
        
        # Calcular percentis para identificar outliers
        q99 = np.percentile(predictions, 99)
        q95 = np.percentile(predictions, 95)
        q75 = np.percentile(predictions, 75)
        
        # Suavizar valores muito altos
        extreme_mask = smoothed_predictions > q99
        if extreme_mask.sum() > 0:
            # Reduzir valores extremos para o percentil 95
            smoothed_predictions[extreme_mask] = q95
            self.logger.info(f"Suavizados {extreme_mask.sum()} valores extremos")
        
        return smoothed_predictions
    
    def _format_predictions(self, 
                          predictions: np.ndarray,
                          features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Formata previsões no formato de saída esperado.
        
        Args:
            predictions: Array com previsões
            features_df: DataFrame com informações de PDV/produto/semana
            
        Returns:
            DataFrame formatado com previsões
        """
        self.logger.info("Formatando previsões para saída")
        
        # Criar DataFrame de saída
        output_df = pd.DataFrame({
            'semana': features_df['semana'].values,
            'pdv': features_df['pdv'].values,
            'produto': features_df['produto'].values,
            'quantidade': predictions
        })
        
        # Filtrar apenas semanas alvo
        output_df = output_df[output_df['semana'].isin(self.target_weeks)]
        
        # Ordenar por semana, PDV e produto
        output_df = output_df.sort_values(['semana', 'pdv', 'produto']).reset_index(drop=True)
        
        # Garantir tipos de dados corretos
        output_df['semana'] = output_df['semana'].astype(int)
        output_df['pdv'] = output_df['pdv'].astype(str)
        output_df['produto'] = output_df['produto'].astype(str)
        output_df['quantidade'] = output_df['quantidade'].astype(int)
        
        self.logger.info(f"Previsões formatadas: {len(output_df)} registros")
        
        return output_df
    
    def _validate_predictions(self, predictions_df: pd.DataFrame) -> None:
        """
        Valida integridade das previsões geradas.
        
        Args:
            predictions_df: DataFrame com previsões
            
        Raises:
            PredictionError: Se validação falhar
        """
        self.logger.info("Validando integridade das previsões")
        
        # Verificar se DataFrame não está vazio
        if len(predictions_df) == 0:
            raise PredictionError("DataFrame de previsões está vazio")
        
        # Verificar colunas obrigatórias
        required_columns = ['semana', 'pdv', 'produto', 'quantidade']
        missing_columns = [col for col in required_columns if col not in predictions_df.columns]
        if missing_columns:
            raise PredictionError(f"Colunas obrigatórias ausentes: {missing_columns}")
        
        # Verificar valores nulos
        null_counts = predictions_df.isnull().sum()
        if null_counts.sum() > 0:
            raise PredictionError(f"Valores nulos encontrados: {null_counts[null_counts > 0].to_dict()}")
        
        # Verificar valores negativos
        negative_quantities = (predictions_df['quantidade'] < 0).sum()
        if negative_quantities > 0:
            raise PredictionError(f"Encontradas {negative_quantities} quantidades negativas")
        
        # Verificar semanas alvo
        predicted_weeks = set(predictions_df['semana'].unique())
        expected_weeks = set(self.target_weeks)
        missing_weeks = expected_weeks - predicted_weeks
        if missing_weeks:
            self.logger.warning(f"Semanas alvo ausentes nas previsões: {missing_weeks}")
        
        # Verificar duplicatas
        duplicates = predictions_df.duplicated(subset=['semana', 'pdv', 'produto']).sum()
        if duplicates > 0:
            raise PredictionError(f"Encontradas {duplicates} combinações duplicadas de semana/PDV/produto")
        
        # Estatísticas de validação
        stats = {
            'total_predictions': len(predictions_df),
            'unique_pdvs': predictions_df['pdv'].nunique(),
            'unique_products': predictions_df['produto'].nunique(),
            'weeks_covered': sorted(predictions_df['semana'].unique()),
            'quantity_stats': {
                'min': predictions_df['quantidade'].min(),
                'max': predictions_df['quantidade'].max(),
                'mean': predictions_df['quantidade'].mean(),
                'sum': predictions_df['quantidade'].sum()
            }
        }
        
        self.logger.info(f"Validação concluída com sucesso: {stats}")
    
    def save_predictions(self, 
                        predictions_df: pd.DataFrame,
                        output_path: str,
                        format_type: str = 'csv') -> str:
        """
        Salva previsões em arquivo usando o formatador de submissão.
        
        Args:
            predictions_df: DataFrame com previsões
            output_path: Caminho para salvar arquivo
            format_type: Formato do arquivo ('csv' ou 'parquet')
            
        Returns:
            Caminho do arquivo salvo
            
        Raises:
            PredictionError: Se salvamento falhar
        """
        self.logger.info(f"Salvando previsões em formato {format_type}")
        
        try:
            # Usar formatador de submissão
            formatter = SubmissionFormatter(self.config)
            
            if format_type.lower() == 'csv':
                saved_path = formatter.format_submission_csv(predictions_df, output_path)
            elif format_type.lower() == 'parquet':
                saved_path = formatter.format_submission_parquet(predictions_df, output_path)
            else:
                raise PredictionError(f"Formato não suportado: {format_type}")
            
            self.logger.info(f"Previsões salvas com sucesso: {saved_path}")
            
            return saved_path
            
        except OutputFormatterError as e:
            self.logger.error(f"Falha na formatação: {str(e)}")
            raise PredictionError(f"Falha na formatação: {str(e)}")
        except Exception as e:
            self.logger.error(f"Falha ao salvar previsões: {str(e)}")
            raise PredictionError(f"Falha ao salvar previsões: {str(e)}")
    
    def generate_prediction_summary(self, predictions_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Gera resumo das previsões geradas.
        
        Args:
            predictions_df: DataFrame com previsões
            
        Returns:
            Dicionário com resumo das previsões
        """
        summary = {
            'total_predictions': len(predictions_df),
            'prediction_period': {
                'weeks': sorted(predictions_df['semana'].unique()),
                'start_week': predictions_df['semana'].min(),
                'end_week': predictions_df['semana'].max()
            },
            'coverage': {
                'unique_pdvs': predictions_df['pdv'].nunique(),
                'unique_products': predictions_df['produto'].nunique(),
                'total_combinations': len(predictions_df)
            },
            'quantity_statistics': {
                'total_quantity': int(predictions_df['quantidade'].sum()),
                'average_quantity': float(predictions_df['quantidade'].mean()),
                'median_quantity': float(predictions_df['quantidade'].median()),
                'min_quantity': int(predictions_df['quantidade'].min()),
                'max_quantity': int(predictions_df['quantidade'].max()),
                'std_quantity': float(predictions_df['quantidade'].std())
            },
            'distribution': {
                'zero_predictions': int((predictions_df['quantidade'] == 0).sum()),
                'low_predictions': int((predictions_df['quantidade'] <= 10).sum()),
                'high_predictions': int((predictions_df['quantidade'] >= 100).sum())
            }
        }
        
        # Estatísticas por semana
        weekly_stats = predictions_df.groupby('semana')['quantidade'].agg([
            'count', 'sum', 'mean', 'std'
        ]).round(2).to_dict('index')
        summary['weekly_statistics'] = weekly_stats
        
        # Top PDVs por volume
        top_pdvs = predictions_df.groupby('pdv')['quantidade'].sum().nlargest(10).to_dict()
        summary['top_pdvs_by_volume'] = top_pdvs
        
        # Top produtos por volume
        top_products = predictions_df.groupby('produto')['quantidade'].sum().nlargest(10).to_dict()
        summary['top_products_by_volume'] = top_products
        
        return summary


class PredictionValidator:
    """Classe para validação específica de previsões."""
    
    @staticmethod
    def validate_prediction_format(predictions_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Valida formato das previsões para submissão.
        
        Args:
            predictions_df: DataFrame com previsões
            
        Returns:
            Dicionário com resultados da validação
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'format_check': {}
        }
        
        # Verificar colunas obrigatórias
        required_columns = ['semana', 'pdv', 'produto', 'quantidade']
        missing_columns = [col for col in required_columns if col not in predictions_df.columns]
        if missing_columns:
            validation_results['errors'].append(f"Colunas obrigatórias ausentes: {missing_columns}")
            validation_results['is_valid'] = False
        
        # Verificar tipos de dados
        if 'quantidade' in predictions_df.columns:
            if not pd.api.types.is_numeric_dtype(predictions_df['quantidade']):
                validation_results['errors'].append("Coluna 'quantidade' deve ser numérica")
                validation_results['is_valid'] = False
                return validation_results  # Retornar cedo se tipo inválido
        
        # Verificar valores válidos
        if 'quantidade' in predictions_df.columns:
            # Verificar valores nulos primeiro
            null_count = predictions_df['quantidade'].isnull().sum()
            if null_count > 0:
                validation_results['errors'].append(f"Encontrados {null_count} valores nulos em quantidade")
                validation_results['is_valid'] = False
            
            # Verificar valores negativos apenas se não há nulos e tipo é numérico
            if pd.api.types.is_numeric_dtype(predictions_df['quantidade']):
                negative_count = (predictions_df['quantidade'] < 0).sum()
                if negative_count > 0:
                    validation_results['errors'].append(f"Encontradas {negative_count} quantidades negativas")
                    validation_results['is_valid'] = False
        
        # Verificar duplicatas
        if len(required_columns) <= len(predictions_df.columns):
            duplicates = predictions_df.duplicated(subset=['semana', 'pdv', 'produto']).sum()
            if duplicates > 0:
                validation_results['errors'].append(f"Encontradas {duplicates} combinações duplicadas")
                validation_results['is_valid'] = False
        
        # Estatísticas de formato
        validation_results['format_check'] = {
            'total_rows': len(predictions_df),
            'total_columns': len(predictions_df.columns),
            'column_names': list(predictions_df.columns),
            'data_types': predictions_df.dtypes.to_dict()
        }
        
        return validation_results
    
    @staticmethod
    def compare_with_baseline(predictions_df: pd.DataFrame, 
                            baseline_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compara previsões com baseline.
        
        Args:
            predictions_df: DataFrame com previsões
            baseline_df: DataFrame com baseline
            
        Returns:
            Dicionário com comparação
        """
        pred_total = predictions_df['quantidade'].sum() if 'quantidade' in predictions_df.columns else 0
        base_total = baseline_df['quantidade'].sum() if 'quantidade' in baseline_df.columns else 0
        
        comparison = {
            'prediction_total': pred_total,
            'baseline_total': base_total,
            'difference_absolute': pred_total - base_total,
            'difference_percentage': 0
        }
        
        if base_total != 0:
            comparison['difference_percentage'] = (pred_total - base_total) / base_total * 100
        
        return comparison
    
    @staticmethod
    def validate_business_rules(predictions_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Valida regras de negócio nas previsões.
        
        Args:
            predictions_df: DataFrame com previsões
            
        Returns:
            Dicionário com validação de regras de negócio
        """
        validation = {
            'is_valid': True,
            'violations': [],
            'statistics': {}
        }
        
        if 'quantidade' in predictions_df.columns:
            # Regra: Quantidades muito altas (outliers)
            q99 = predictions_df['quantidade'].quantile(0.99)
            high_quantities = (predictions_df['quantidade'] > q99 * 2).sum()
            if high_quantities > 0:
                validation['violations'].append(f"Encontradas {high_quantities} quantidades suspeitas (muito altas)")
            
            # Regra: Distribuição de zeros
            zero_predictions = (predictions_df['quantidade'] == 0).sum()
            zero_percentage = zero_predictions / len(predictions_df) * 100
            if zero_percentage > 50:
                validation['violations'].append(f"Muitas previsões zero: {zero_percentage:.1f}%")
            
            # Estatísticas para análise
            validation['statistics'] = {
                'zero_predictions_pct': zero_percentage,
                'high_quantity_count': high_quantities,
                'avg_quantity': predictions_df['quantidade'].mean(),
                'quantity_std': predictions_df['quantidade'].std()
            }
        
        if validation['violations']:
            validation['is_valid'] = False
        
        return validation