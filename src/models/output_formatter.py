"""
Módulo de formatação de saída para o Hackathon Forecast Model 2025.

Este módulo implementa a geração de arquivos de submissão em formato CSV e Parquet,
com validação de formato específico para o hackathon.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
import logging
from pathlib import Path
from datetime import datetime
import warnings

logger = logging.getLogger(__name__)


class OutputFormatterError(Exception):
    """Exceção customizada para erros de formatação de saída."""
    pass


class SubmissionFormatter:
    """
    Classe para formatação de arquivos de submissão do hackathon.
    
    Responsável por gerar arquivos CSV e Parquet no formato específico
    requerido pela competição, com validações de integridade.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa o formatador de submissão.
        
        Args:
            config: Configuração de formatação de saída
        """
        self.config = config
        self.output_config = config.get('output', {})
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # Configurações de formato CSV
        self.csv_separator = self.output_config.get('csv_separator', ';')
        self.csv_encoding = self.output_config.get('csv_encoding', 'utf-8')
        self.csv_decimal = self.output_config.get('csv_decimal', '.')
        
        # Configurações de formato Parquet
        self.parquet_compression = self.output_config.get('parquet_compression', 'snappy')
        
        # Configurações de validação
        self.validate_format = self.output_config.get('validate_format', True)
        self.add_timestamp = self.output_config.get('add_timestamp', True)
        
    def _setup_logging(self) -> None:
        """Configura logging para operações de formatação."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def format_submission_csv(self, 
                            predictions_df: pd.DataFrame,
                            output_path: str,
                            validate: bool = True) -> str:
        """
        Formata e salva previsões em arquivo CSV para submissão.
        
        Args:
            predictions_df: DataFrame com previsões
            output_path: Caminho para salvar arquivo CSV
            validate: Se deve validar formato antes de salvar
            
        Returns:
            Caminho do arquivo CSV salvo
            
        Raises:
            OutputFormatterError: Se formatação falhar
        """
        self.logger.info("Formatando arquivo CSV para submissão")
        
        try:
            # Validar formato se solicitado
            if validate and self.validate_format:
                self._validate_submission_format(predictions_df)
            
            # Preparar DataFrame para CSV
            csv_df = self._prepare_csv_format(predictions_df)
            
            # Adicionar timestamp ao nome do arquivo se configurado
            if self.add_timestamp:
                output_path = self._add_timestamp_to_path(output_path)
            
            # Criar diretório se não existir
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Salvar CSV com configurações específicas
            csv_df.to_csv(
                output_path,
                sep=self.csv_separator,
                encoding=self.csv_encoding,
                decimal=self.csv_decimal,
                index=False,
                float_format='%.0f'  # Garantir que quantidades sejam inteiras
            )
            
            # Verificar se arquivo foi criado
            if not Path(output_path).exists():
                raise OutputFormatterError(f"Arquivo CSV não foi criado: {output_path}")
            
            # Log de informações do arquivo
            file_size = Path(output_path).stat().st_size
            self.logger.info(f"Arquivo CSV salvo: {output_path} ({file_size} bytes)")
            
            # Validar arquivo salvo
            if validate:
                self._validate_saved_csv(output_path)
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Falha na formatação CSV: {str(e)}")
            raise OutputFormatterError(f"Falha na formatação CSV: {str(e)}")
    
    def format_submission_parquet(self, 
                                predictions_df: pd.DataFrame,
                                output_path: str,
                                validate: bool = True) -> str:
        """
        Formata e salva previsões em arquivo Parquet para submissão.
        
        Args:
            predictions_df: DataFrame com previsões
            output_path: Caminho para salvar arquivo Parquet
            validate: Se deve validar formato antes de salvar
            
        Returns:
            Caminho do arquivo Parquet salvo
            
        Raises:
            OutputFormatterError: Se formatação falhar
        """
        self.logger.info("Formatando arquivo Parquet para submissão")
        
        try:
            # Validar formato se solicitado
            if validate and self.validate_format:
                self._validate_submission_format(predictions_df)
            
            # Preparar DataFrame para Parquet
            parquet_df = self._prepare_parquet_format(predictions_df)
            
            # Adicionar timestamp ao nome do arquivo se configurado
            if self.add_timestamp:
                output_path = self._add_timestamp_to_path(output_path)
            
            # Criar diretório se não existir
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Salvar Parquet com configurações específicas
            parquet_df.to_parquet(
                output_path,
                compression=self.parquet_compression,
                index=False
            )
            
            # Verificar se arquivo foi criado
            if not Path(output_path).exists():
                raise OutputFormatterError(f"Arquivo Parquet não foi criado: {output_path}")
            
            # Log de informações do arquivo
            file_size = Path(output_path).stat().st_size
            self.logger.info(f"Arquivo Parquet salvo: {output_path} ({file_size} bytes)")
            
            # Validar arquivo salvo
            if validate:
                self._validate_saved_parquet(output_path)
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Falha na formatação Parquet: {str(e)}")
            raise OutputFormatterError(f"Falha na formatação Parquet: {str(e)}")
    
    def _prepare_csv_format(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara DataFrame para formato CSV de submissão.
        
        Args:
            predictions_df: DataFrame com previsões
            
        Returns:
            DataFrame formatado para CSV
        """
        csv_df = predictions_df.copy()
        
        # Garantir ordem das colunas conforme especificação
        required_columns = ['semana', 'pdv', 'produto', 'quantidade']
        csv_df = csv_df[required_columns]
        
        # Garantir tipos de dados corretos para CSV
        csv_df['semana'] = csv_df['semana'].astype(int)
        csv_df['pdv'] = csv_df['pdv'].astype(str)
        csv_df['produto'] = csv_df['produto'].astype(str)
        csv_df['quantidade'] = csv_df['quantidade'].astype(int)
        
        # Ordenar por semana, PDV e produto
        csv_df = csv_df.sort_values(['semana', 'pdv', 'produto']).reset_index(drop=True)
        
        self.logger.info(f"DataFrame preparado para CSV: {len(csv_df)} registros")
        
        return csv_df
    
    def _prepare_parquet_format(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara DataFrame para formato Parquet de submissão.
        
        Args:
            predictions_df: DataFrame com previsões
            
        Returns:
            DataFrame formatado para Parquet
        """
        parquet_df = predictions_df.copy()
        
        # Garantir ordem das colunas conforme especificação
        required_columns = ['semana', 'pdv', 'produto', 'quantidade']
        parquet_df = parquet_df[required_columns]
        
        # Otimizar tipos de dados para Parquet
        parquet_df['semana'] = parquet_df['semana'].astype('int8')  # Semanas 1-5
        parquet_df['pdv'] = parquet_df['pdv'].astype('category')
        parquet_df['produto'] = parquet_df['produto'].astype('category')
        parquet_df['quantidade'] = parquet_df['quantidade'].astype('int32')
        
        # Ordenar por semana, PDV e produto
        parquet_df = parquet_df.sort_values(['semana', 'pdv', 'produto']).reset_index(drop=True)
        
        self.logger.info(f"DataFrame preparado para Parquet: {len(parquet_df)} registros")
        
        return parquet_df
    
    def _validate_submission_format(self, predictions_df: pd.DataFrame) -> None:
        """
        Valida formato das previsões para submissão.
        
        Args:
            predictions_df: DataFrame com previsões
            
        Raises:
            OutputFormatterError: Se validação falhar
        """
        self.logger.info("Validando formato de submissão")
        
        # Verificar se DataFrame não está vazio
        if len(predictions_df) == 0:
            raise OutputFormatterError("DataFrame de previsões está vazio")
        
        # Verificar colunas obrigatórias
        required_columns = ['semana', 'pdv', 'produto', 'quantidade']
        missing_columns = [col for col in required_columns if col not in predictions_df.columns]
        if missing_columns:
            raise OutputFormatterError(f"Colunas obrigatórias ausentes: {missing_columns}")
        
        # Verificar valores nulos
        null_counts = predictions_df[required_columns].isnull().sum()
        if null_counts.sum() > 0:
            raise OutputFormatterError(f"Valores nulos encontrados: {null_counts[null_counts > 0].to_dict()}")
        
        # Verificar valores negativos em quantidade
        negative_quantities = (predictions_df['quantidade'] < 0).sum()
        if negative_quantities > 0:
            raise OutputFormatterError(f"Encontradas {negative_quantities} quantidades negativas")
        
        # Verificar duplicatas
        duplicates = predictions_df.duplicated(subset=['semana', 'pdv', 'produto']).sum()
        if duplicates > 0:
            raise OutputFormatterError(f"Encontradas {duplicates} combinações duplicadas de semana/PDV/produto")
        
        # Verificar semanas válidas (1-5 para janeiro)
        valid_weeks = {1, 2, 3, 4, 5}
        invalid_weeks = set(predictions_df['semana'].unique()) - valid_weeks
        if invalid_weeks:
            raise OutputFormatterError(f"Semanas inválidas encontradas: {invalid_weeks}")
        
        # Verificar tipos de dados
        if not pd.api.types.is_numeric_dtype(predictions_df['quantidade']):
            raise OutputFormatterError("Coluna 'quantidade' deve ser numérica")
        
        self.logger.info("Validação de formato concluída com sucesso")
    
    def _validate_saved_csv(self, file_path: str) -> None:
        """
        Valida arquivo CSV salvo.
        
        Args:
            file_path: Caminho do arquivo CSV
            
        Raises:
            OutputFormatterError: Se validação falhar
        """
        try:
            # Carregar arquivo salvo para validação
            saved_df = pd.read_csv(
                file_path,
                sep=self.csv_separator,
                encoding=self.csv_encoding
            )
            
            # Verificar se carregamento foi bem-sucedido
            if len(saved_df) == 0:
                raise OutputFormatterError("Arquivo CSV salvo está vazio")
            
            # Verificar colunas
            expected_columns = ['semana', 'pdv', 'produto', 'quantidade']
            if list(saved_df.columns) != expected_columns:
                raise OutputFormatterError(f"Colunas incorretas no CSV: {list(saved_df.columns)}")
            
            self.logger.info(f"Arquivo CSV validado: {len(saved_df)} registros")
            
        except Exception as e:
            raise OutputFormatterError(f"Falha na validação do CSV salvo: {str(e)}")
    
    def _validate_saved_parquet(self, file_path: str) -> None:
        """
        Valida arquivo Parquet salvo.
        
        Args:
            file_path: Caminho do arquivo Parquet
            
        Raises:
            OutputFormatterError: Se validação falhar
        """
        try:
            # Carregar arquivo salvo para validação
            saved_df = pd.read_parquet(file_path)
            
            # Verificar se carregamento foi bem-sucedido
            if len(saved_df) == 0:
                raise OutputFormatterError("Arquivo Parquet salvo está vazio")
            
            # Verificar colunas
            expected_columns = ['semana', 'pdv', 'produto', 'quantidade']
            if list(saved_df.columns) != expected_columns:
                raise OutputFormatterError(f"Colunas incorretas no Parquet: {list(saved_df.columns)}")
            
            self.logger.info(f"Arquivo Parquet validado: {len(saved_df)} registros")
            
        except Exception as e:
            raise OutputFormatterError(f"Falha na validação do Parquet salvo: {str(e)}")
    
    def _add_timestamp_to_path(self, file_path: str) -> str:
        """
        Adiciona timestamp ao nome do arquivo.
        
        Args:
            file_path: Caminho original do arquivo
            
        Returns:
            Caminho com timestamp adicionado
        """
        path_obj = Path(file_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Inserir timestamp antes da extensão
        new_name = f"{path_obj.stem}_{timestamp}{path_obj.suffix}"
        new_path = path_obj.parent / new_name
        
        return str(new_path)
    
    def generate_submission_summary(self, predictions_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Gera resumo da submissão.
        
        Args:
            predictions_df: DataFrame com previsões
            
        Returns:
            Dicionário com resumo da submissão
        """
        summary = {
            'submission_info': {
                'total_predictions': len(predictions_df),
                'format_version': '1.0',
                'generated_at': datetime.now().isoformat(),
                'columns': list(predictions_df.columns)
            },
            'data_coverage': {
                'weeks': sorted(predictions_df['semana'].unique()),
                'unique_pdvs': predictions_df['pdv'].nunique(),
                'unique_products': predictions_df['produto'].nunique(),
                'total_combinations': len(predictions_df)
            },
            'quantity_summary': {
                'total_quantity': int(predictions_df['quantidade'].sum()),
                'average_quantity': float(predictions_df['quantidade'].mean()),
                'min_quantity': int(predictions_df['quantidade'].min()),
                'max_quantity': int(predictions_df['quantidade'].max()),
                'zero_predictions': int((predictions_df['quantidade'] == 0).sum()),
                'zero_percentage': float((predictions_df['quantidade'] == 0).mean() * 100)
            },
            'weekly_breakdown': {}
        }
        
        # Breakdown por semana
        for week in sorted(predictions_df['semana'].unique()):
            week_data = predictions_df[predictions_df['semana'] == week]
            summary['weekly_breakdown'][f'week_{week}'] = {
                'predictions': len(week_data),
                'total_quantity': int(week_data['quantidade'].sum()),
                'avg_quantity': float(week_data['quantidade'].mean()),
                'unique_pdvs': week_data['pdv'].nunique(),
                'unique_products': week_data['produto'].nunique()
            }
        
        return summary


class SubmissionValidator:
    """Classe para validação específica de arquivos de submissão."""
    
    @staticmethod
    def validate_csv_format(file_path: str, 
                          separator: str = ';',
                          encoding: str = 'utf-8') -> Dict[str, Any]:
        """
        Valida formato de arquivo CSV de submissão.
        
        Args:
            file_path: Caminho do arquivo CSV
            separator: Separador usado no CSV
            encoding: Encoding do arquivo
            
        Returns:
            Dicionário com resultados da validação
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'file_info': {},
            'data_info': {}
        }
        
        try:
            # Verificar se arquivo existe
            if not Path(file_path).exists():
                validation_results['errors'].append(f"Arquivo não encontrado: {file_path}")
                validation_results['is_valid'] = False
                return validation_results
            
            # Informações do arquivo
            file_size = Path(file_path).stat().st_size
            validation_results['file_info'] = {
                'path': file_path,
                'size_bytes': file_size,
                'size_mb': round(file_size / (1024 * 1024), 2)
            }
            
            # Carregar e validar CSV
            df = pd.read_csv(file_path, sep=separator, encoding=encoding)
            
            # Validar estrutura
            expected_columns = ['semana', 'pdv', 'produto', 'quantidade']
            if list(df.columns) != expected_columns:
                validation_results['errors'].append(
                    f"Colunas incorretas. Esperado: {expected_columns}, Encontrado: {list(df.columns)}"
                )
                validation_results['is_valid'] = False
            
            # Validar dados
            if len(df) == 0:
                validation_results['errors'].append("Arquivo CSV está vazio")
                validation_results['is_valid'] = False
            else:
                # Informações dos dados
                validation_results['data_info'] = {
                    'total_rows': len(df),
                    'weeks': sorted(df['semana'].unique()) if 'semana' in df.columns else [],
                    'unique_pdvs': df['pdv'].nunique() if 'pdv' in df.columns else 0,
                    'unique_products': df['produto'].nunique() if 'produto' in df.columns else 0
                }
                
                # Validar valores
                if 'quantidade' in df.columns:
                    null_count = df['quantidade'].isnull().sum()
                    if null_count > 0:
                        validation_results['errors'].append(f"Valores nulos em quantidade: {null_count}")
                        validation_results['is_valid'] = False
                    
                    negative_count = (df['quantidade'] < 0).sum()
                    if negative_count > 0:
                        validation_results['errors'].append(f"Valores negativos em quantidade: {negative_count}")
                        validation_results['is_valid'] = False
        
        except Exception as e:
            validation_results['errors'].append(f"Erro ao validar CSV: {str(e)}")
            validation_results['is_valid'] = False
        
        return validation_results
    
    @staticmethod
    def validate_parquet_format(file_path: str) -> Dict[str, Any]:
        """
        Valida formato de arquivo Parquet de submissão.
        
        Args:
            file_path: Caminho do arquivo Parquet
            
        Returns:
            Dicionário com resultados da validação
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'file_info': {},
            'data_info': {}
        }
        
        try:
            # Verificar se arquivo existe
            if not Path(file_path).exists():
                validation_results['errors'].append(f"Arquivo não encontrado: {file_path}")
                validation_results['is_valid'] = False
                return validation_results
            
            # Informações do arquivo
            file_size = Path(file_path).stat().st_size
            validation_results['file_info'] = {
                'path': file_path,
                'size_bytes': file_size,
                'size_mb': round(file_size / (1024 * 1024), 2)
            }
            
            # Carregar e validar Parquet
            df = pd.read_parquet(file_path)
            
            # Validar estrutura
            expected_columns = ['semana', 'pdv', 'produto', 'quantidade']
            if list(df.columns) != expected_columns:
                validation_results['errors'].append(
                    f"Colunas incorretas. Esperado: {expected_columns}, Encontrado: {list(df.columns)}"
                )
                validation_results['is_valid'] = False
            
            # Validar dados
            if len(df) == 0:
                validation_results['errors'].append("Arquivo Parquet está vazio")
                validation_results['is_valid'] = False
            else:
                # Informações dos dados
                validation_results['data_info'] = {
                    'total_rows': len(df),
                    'weeks': sorted(df['semana'].unique()) if 'semana' in df.columns else [],
                    'unique_pdvs': df['pdv'].nunique() if 'pdv' in df.columns else 0,
                    'unique_products': df['produto'].nunique() if 'produto' in df.columns else 0,
                    'data_types': df.dtypes.to_dict()
                }
                
                # Validar valores
                if 'quantidade' in df.columns:
                    null_count = df['quantidade'].isnull().sum()
                    if null_count > 0:
                        validation_results['errors'].append(f"Valores nulos em quantidade: {null_count}")
                        validation_results['is_valid'] = False
                    
                    negative_count = (df['quantidade'] < 0).sum()
                    if negative_count > 0:
                        validation_results['errors'].append(f"Valores negativos em quantidade: {negative_count}")
                        validation_results['is_valid'] = False
        
        except Exception as e:
            validation_results['errors'].append(f"Erro ao validar Parquet: {str(e)}")
            validation_results['is_valid'] = False
        
        return validation_results
    
    @staticmethod
    def compare_formats(csv_path: str, parquet_path: str) -> Dict[str, Any]:
        """
        Compara arquivos CSV e Parquet para verificar consistência.
        
        Args:
            csv_path: Caminho do arquivo CSV
            parquet_path: Caminho do arquivo Parquet
            
        Returns:
            Dicionário com comparação entre formatos
        """
        comparison = {
            'files_consistent': True,
            'differences': [],
            'csv_info': {},
            'parquet_info': {}
        }
        
        try:
            # Carregar ambos os arquivos
            csv_df = pd.read_csv(csv_path, sep=';', encoding='utf-8')
            parquet_df = pd.read_parquet(parquet_path)
            
            # Informações básicas
            comparison['csv_info'] = {
                'rows': len(csv_df),
                'file_size': Path(csv_path).stat().st_size
            }
            comparison['parquet_info'] = {
                'rows': len(parquet_df),
                'file_size': Path(parquet_path).stat().st_size
            }
            
            # Comparar número de linhas
            if len(csv_df) != len(parquet_df):
                comparison['differences'].append(
                    f"Número de linhas diferente: CSV={len(csv_df)}, Parquet={len(parquet_df)}"
                )
                comparison['files_consistent'] = False
            
            # Comparar colunas
            if list(csv_df.columns) != list(parquet_df.columns):
                comparison['differences'].append(
                    f"Colunas diferentes: CSV={list(csv_df.columns)}, Parquet={list(parquet_df.columns)}"
                )
                comparison['files_consistent'] = False
            
            # Comparar dados (se mesmo tamanho)
            if len(csv_df) == len(parquet_df) and list(csv_df.columns) == list(parquet_df.columns):
                # Ordenar ambos para comparação
                csv_sorted = csv_df.sort_values(['semana', 'pdv', 'produto']).reset_index(drop=True)
                parquet_sorted = parquet_df.sort_values(['semana', 'pdv', 'produto']).reset_index(drop=True)
                
                # Normalizar tipos para comparação
                for col in ['semana', 'quantidade']:
                    csv_sorted[col] = csv_sorted[col].astype(int)
                    parquet_sorted[col] = parquet_sorted[col].astype(int)
                
                # Para PDV e produto, normalizar considerando que CSV pode converter para int
                for col in ['pdv', 'produto']:
                    csv_col = csv_sorted[col]
                    parquet_col = parquet_sorted[col]
                    
                    # Se CSV converteu para int, converter de volta para string com zeros à esquerda
                    if pd.api.types.is_numeric_dtype(csv_col):
                        # Assumir formato de 3 dígitos para PDV
                        if col == 'pdv':
                            csv_sorted[col] = csv_col.astype(int).astype(str).str.zfill(3)
                        else:
                            csv_sorted[col] = csv_col.astype(str)
                    else:
                        csv_sorted[col] = csv_col.astype(str)
                    
                    parquet_sorted[col] = parquet_col.astype(str)
                
                # Comparar valores coluna por coluna para melhor diagnóstico
                data_matches = True
                for col in csv_sorted.columns:
                    if not csv_sorted[col].equals(parquet_sorted[col]):
                        comparison['differences'].append(f"Coluna '{col}' difere entre CSV e Parquet")
                        data_matches = False
                
                if not data_matches:
                    comparison['files_consistent'] = False
        
        except Exception as e:
            comparison['differences'].append(f"Erro na comparação: {str(e)}")
            comparison['files_consistent'] = False
        
        return comparison