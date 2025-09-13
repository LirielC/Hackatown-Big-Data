"""
Pipeline Rápido de Submissão - Hackathon 2025
Pipeline otimizado para geração rápida de múltiplas submissões.
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import yaml
import joblib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
import hashlib

from .submission_manager import SubmissionManager, SubmissionMetadata
from .performance_utils import PerformanceOptimizer

logger = logging.getLogger(__name__)


@dataclass
class FastPipelineConfig:
    """Configuração do pipeline rápido."""
    use_cached_features: bool = True
    use_cached_models: bool = True
    parallel_training: bool = True
    max_workers: Optional[int] = None
    reduced_hyperparameter_search: bool = True
    sample_data_for_tuning: bool = True
    cache_ttl_hours: int = 24


class FeatureCache:
    """Sistema de cache para features."""
    
    def __init__(self, cache_dir: str = "cache/features"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get_cache_key(self, data_hash: str, feature_config: Dict[str, Any]) -> str:
        """Gera chave única para cache baseada nos dados e configuração."""
        config_str = str(sorted(feature_config.items()))
        combined = f"{data_hash}_{config_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def is_cache_valid(self, cache_key: str, ttl_hours: int = 24) -> bool:
        """Verifica se cache é válido baseado no TTL."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if not cache_file.exists():
            return False
        
        # Verificar TTL
        file_age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        return file_age_hours < ttl_hours
    
    def save_features(self, cache_key: str, features_df: pd.DataFrame, 
                     metadata: Dict[str, Any]) -> None:
        """Salva features no cache."""
        cache_data = {
            'features': features_df,
            'metadata': metadata,
            'timestamp': time.time()
        }
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        joblib.dump(cache_data, cache_file, compress=3)
        
        logger.info(f"Features salvas no cache: {cache_key}")
    
    def load_features(self, cache_key: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Carrega features do cache."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if not cache_file.exists():
            raise FileNotFoundError(f"Cache não encontrado: {cache_key}")
        
        cache_data = joblib.load(cache_file)
        
        logger.info(f"Features carregadas do cache: {cache_key}")
        return cache_data['features'], cache_data['metadata']


class ModelCache:
    """Sistema de cache para modelos treinados."""
    
    def __init__(self, cache_dir: str = "cache/models"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_model_key(self, model_type: str, model_config: Dict[str, Any], 
                     data_hash: str) -> str:
        """Gera chave única para modelo."""
        config_str = str(sorted(model_config.items()))
        combined = f"{model_type}_{data_hash}_{config_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def save_model(self, model_key: str, model, metadata: Dict[str, Any]) -> None:
        """Salva modelo no cache."""
        model_data = {
            'model': model,
            'metadata': metadata,
            'timestamp': time.time()
        }
        
        cache_file = self.cache_dir / f"{model_key}.pkl"
        joblib.dump(model_data, cache_file, compress=3)
        
        logger.info(f"Modelo salvo no cache: {model_key}")
    
    def load_model(self, model_key: str):
        """Carrega modelo do cache."""
        cache_file = self.cache_dir / f"{model_key}.pkl"
        
        if not cache_file.exists():
            raise FileNotFoundError(f"Modelo não encontrado no cache: {model_key}")
        
        model_data = joblib.load(cache_file)
        
        logger.info(f"Modelo carregado do cache: {model_key}")
        return model_data['model'], model_data['metadata']
    
    def is_model_cached(self, model_key: str, ttl_hours: int = 24) -> bool:
        """Verifica se modelo está no cache e é válido."""
        cache_file = self.cache_dir / f"{model_key}.pkl"
        
        if not cache_file.exists():
            return False
        
        # Verificar TTL
        file_age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        return file_age_hours < ttl_hours


class FastSubmissionPipeline:
    """Pipeline otimizado para geração rápida de submissões."""
    
    def __init__(self, config_path: str = "configs/submission_strategies.yaml"):
        self.submission_manager = SubmissionManager(config_path)
        self.config = self.submission_manager.config
        
        # Configurar caches
        fast_config = self.config.get('fast_execution', {})
        cache_config = fast_config.get('cache', {})
        
        self.feature_cache = FeatureCache(
            cache_config.get('features_cache_dir', 'cache/features')
        )
        self.model_cache = ModelCache(
            cache_config.get('models_cache_dir', 'cache/models')
        )
        
        # Configurar otimizações
        self.fast_config = FastPipelineConfig(
            use_cached_features=fast_config.get('optimizations', {}).get('use_cached_features', True),
            use_cached_models=fast_config.get('optimizations', {}).get('use_cached_models', True),
            parallel_training=fast_config.get('optimizations', {}).get('parallel_training', True),
            max_workers=fast_config.get('parallel', {}).get('max_workers'),
            reduced_hyperparameter_search=fast_config.get('optimizations', {}).get('reduced_hyperparameter_search', True),
            sample_data_for_tuning=fast_config.get('optimizations', {}).get('sample_data_for_tuning', True),
            cache_ttl_hours=cache_config.get('ttl_hours', 24)
        )
        
        self.performance_optimizer = PerformanceOptimizer()
    
    def generate_all_submissions(self, data_path: str = "data/processed/features_engineered.parquet") -> List[SubmissionMetadata]:
        """Gera submissões para todas as estratégias configuradas."""
        logger.info("=== INICIANDO GERAÇÃO RÁPIDA DE SUBMISSÕES ===")
        
        start_time = time.time()
        submissions = []
        
        # Carregar dados base
        logger.info(f"Carregando dados de: {data_path}")
        base_data = pd.read_parquet(data_path)
        data_hash = self._get_data_hash(base_data)
        
        strategies = self.submission_manager.list_strategies()
        logger.info(f"Estratégias a processar: {len(strategies)}")
        
        if self.fast_config.parallel_training and len(strategies) > 1:
            # Processamento paralelo
            submissions = self._generate_submissions_parallel(base_data, data_hash, strategies)
        else:
            # Processamento sequencial
            submissions = self._generate_submissions_sequential(base_data, data_hash, strategies)
        
        total_time = time.time() - start_time
        logger.info(f"=== GERAÇÃO CONCLUÍDA ===")
        logger.info(f"Total de submissões: {len(submissions)}")
        logger.info(f"Tempo total: {total_time:.2f}s ({total_time/60:.2f}min)")
        logger.info(f"Tempo médio por submissão: {total_time/len(submissions):.2f}s")
        
        return submissions
    
    def generate_single_submission(self, strategy_name: str,
                                 data_path: str = "data/processed/features_engineered.parquet",
                                 version_type: str = "patch") -> SubmissionMetadata:
        """Gera submissão para uma estratégia específica."""
        logger.info(f"=== GERANDO SUBMISSÃO: {strategy_name} ===")
        
        start_time = time.time()
        
        # Carregar dados
        base_data = pd.read_parquet(data_path)
        data_hash = self._get_data_hash(base_data)
        
        # Processar estratégia
        submission = self._process_strategy(strategy_name, base_data, data_hash, version_type)
        
        total_time = time.time() - start_time
        logger.info(f"Submissão gerada em {total_time:.2f}s")
        
        return submission
    
    def _generate_submissions_parallel(self, base_data: pd.DataFrame, data_hash: str, 
                                     strategies: List[str]) -> List[SubmissionMetadata]:
        """Gera submissões em paralelo."""
        logger.info("Processamento paralelo habilitado")
        
        max_workers = self.fast_config.max_workers or min(len(strategies), os.cpu_count())
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for strategy_name in strategies:
                future = executor.submit(
                    self._process_strategy, 
                    strategy_name, 
                    base_data, 
                    data_hash
                )
                futures.append((strategy_name, future))
            
            submissions = []
            for strategy_name, future in futures:
                try:
                    submission = future.result(timeout=1800)  # 30 min timeout
                    submissions.append(submission)
                    logger.info(f"✓ Estratégia concluída: {strategy_name}")
                except Exception as e:
                    logger.error(f"✗ Erro na estratégia {strategy_name}: {e}")
        
        return submissions
    
    def _generate_submissions_sequential(self, base_data: pd.DataFrame, data_hash: str, 
                                       strategies: List[str]) -> List[SubmissionMetadata]:
        """Gera submissões sequencialmente."""
        logger.info("Processamento sequencial")
        
        submissions = []
        
        for i, strategy_name in enumerate(strategies, 1):
            logger.info(f"Processando estratégia {i}/{len(strategies)}: {strategy_name}")
            
            try:
                submission = self._process_strategy(strategy_name, base_data, data_hash)
                submissions.append(submission)
                logger.info(f"✓ Estratégia concluída: {strategy_name}")
            except Exception as e:
                logger.error(f"✗ Erro na estratégia {strategy_name}: {e}")
        
        return submissions
    
    def _process_strategy(self, strategy_name: str, base_data: pd.DataFrame, 
                         data_hash: str, version_type: str = "patch") -> SubmissionMetadata:
        """Processa uma estratégia específica."""
        strategy_start = time.time()
        
        # Obter configuração da estratégia
        strategy_config = self.submission_manager.get_strategy_config(strategy_name)
        
        # 1. Feature Engineering (com cache)
        features_df = self._get_or_create_features(
            base_data, strategy_config, data_hash
        )
        
        # 2. Treinamento de Modelos (com cache)
        models = self._get_or_train_models(
            features_df, strategy_config, data_hash
        )
        
        # 3. Geração de Previsões
        predictions_df = self._generate_predictions(
            models, features_df, strategy_config
        )
        
        # 4. Validação e Métricas
        performance_metrics = self._calculate_performance_metrics(
            predictions_df, features_df, strategy_config
        )
        
        # 5. Criar Submissão
        submission = self.submission_manager.create_submission(
            strategy_name=strategy_name,
            predictions_df=predictions_df,
            performance_metrics=performance_metrics,
            model_parameters=self._extract_model_parameters(models),
            version_type=version_type
        )
        
        strategy_time = time.time() - strategy_start
        logger.info(f"Estratégia {strategy_name} processada em {strategy_time:.2f}s")
        
        return submission
    
    def _get_or_create_features(self, base_data: pd.DataFrame, 
                              strategy_config: Dict[str, Any], 
                              data_hash: str) -> pd.DataFrame:
        """Obtém features do cache ou cria novas."""
        if not self.fast_config.use_cached_features:
            return self._create_features(base_data, strategy_config)
        
        # Gerar chave do cache
        feature_config = strategy_config.get('features', {})
        cache_key = self.feature_cache.get_cache_key(data_hash, feature_config)
        
        # Verificar cache
        if self.feature_cache.is_cache_valid(cache_key, self.fast_config.cache_ttl_hours):
            try:
                features_df, metadata = self.feature_cache.load_features(cache_key)
                logger.info(f"Features carregadas do cache para {strategy_config['name']}")
                return features_df
            except Exception as e:
                logger.warning(f"Erro ao carregar features do cache: {e}")
        
        # Criar features
        features_df = self._create_features(base_data, strategy_config)
        
        # Salvar no cache
        try:
            metadata = {
                'strategy_name': strategy_config['name'],
                'feature_config': feature_config,
                'data_shape': features_df.shape
            }
            self.feature_cache.save_features(cache_key, features_df, metadata)
        except Exception as e:
            logger.warning(f"Erro ao salvar features no cache: {e}")
        
        return features_df
    
    def _create_features(self, base_data: pd.DataFrame, 
                        strategy_config: Dict[str, Any]) -> pd.DataFrame:
        """Cria features baseado na configuração da estratégia."""
        from ..features.engineering import FeatureEngineer
        
        feature_config = strategy_config.get('features', {})
        
        # Configurar feature engineer
        engineer = FeatureEngineer()
        
        # Criar features
        features_df = engineer.create_all_features(base_data)
        
        # Seleção de features se configurada
        max_features = feature_config.get('max_features')
        if max_features and len(features_df.columns) > max_features:
            from ..features.selection import FeatureSelector
            
            selector = FeatureSelector()
            features_df = selector.select_features(
                features_df, 
                method=feature_config.get('feature_selection_method', 'rfe'),
                max_features=max_features
            )
        
        logger.info(f"Features criadas: {features_df.shape}")
        return features_df
    
    def _get_or_train_models(self, features_df: pd.DataFrame, 
                           strategy_config: Dict[str, Any], 
                           data_hash: str) -> Dict[str, Any]:
        """Obtém modelos do cache ou treina novos."""
        models = {}
        
        for model_type, model_config in strategy_config['models'].items():
            if not model_config.get('enabled', True):
                continue
            
            if self.fast_config.use_cached_models:
                # Tentar carregar do cache
                model_key = self.model_cache.get_model_key(
                    model_type, model_config, data_hash
                )
                
                if self.model_cache.is_model_cached(model_key, self.fast_config.cache_ttl_hours):
                    try:
                        model, metadata = self.model_cache.load_model(model_key)
                        models[model_type] = model
                        logger.info(f"Modelo {model_type} carregado do cache")
                        continue
                    except Exception as e:
                        logger.warning(f"Erro ao carregar modelo {model_type} do cache: {e}")
            
            # Treinar modelo
            model = self._train_model(model_type, model_config, features_df)
            models[model_type] = model
            
            # Salvar no cache
            if self.fast_config.use_cached_models:
                try:
                    metadata = {
                        'model_type': model_type,
                        'config': model_config,
                        'data_shape': features_df.shape
                    }
                    model_key = self.model_cache.get_model_key(
                        model_type, model_config, data_hash
                    )
                    self.model_cache.save_model(model_key, model, metadata)
                except Exception as e:
                    logger.warning(f"Erro ao salvar modelo {model_type} no cache: {e}")
        
        return models
    
    def _train_model(self, model_type: str, model_config: Dict[str, Any], 
                    features_df: pd.DataFrame):
        """Treina um modelo específico."""
        from ..models.training import ModelTrainer
        
        trainer = ModelTrainer()
        
        # Configurar hiperparâmetros
        hyperparameters = model_config.get('hyperparameters', {})
        
        # Reduzir busca de hiperparâmetros se configurado
        if self.fast_config.reduced_hyperparameter_search:
            hyperparameters = self._reduce_hyperparameter_space(hyperparameters)
        
        # Usar amostra dos dados para tuning se configurado
        training_data = features_df
        if self.fast_config.sample_data_for_tuning and len(features_df) > 50000:
            sample_size = min(50000, len(features_df))
            training_data = features_df.sample(n=sample_size, random_state=42)
            logger.info(f"Usando amostra de {sample_size} registros para treinamento")
        
        # Treinar modelo
        if model_type == 'xgboost':
            model = trainer.train_xgboost(training_data, hyperparameters)
        elif model_type == 'lightgbm':
            model = trainer.train_lightgbm(training_data, hyperparameters)
        elif model_type == 'prophet':
            model = trainer.train_prophet(training_data, hyperparameters)
        else:
            raise ValueError(f"Tipo de modelo não suportado: {model_type}")
        
        logger.info(f"Modelo {model_type} treinado")
        return model
    
    def _generate_predictions(self, models: Dict[str, Any], features_df: pd.DataFrame, 
                            strategy_config: Dict[str, Any]) -> pd.DataFrame:
        """Gera previsões usando os modelos."""
        from ..models.prediction import PredictionGenerator
        from ..models.ensemble import EnsembleModel
        
        # Se há apenas um modelo, usar diretamente
        if len(models) == 1:
            model_type, model = next(iter(models.items()))
            # Para modelo único, usar o modelo diretamente para predição
            try:
                import numpy as np
                # Usar apenas as features que o modelo conhece (excluindo target)
                if hasattr(model, 'feature_names_in_'):
                    model_features = model.feature_names_in_
                    # Filtrar apenas features que existem no dataset e são conhecidas pelo modelo
                    available_features = [col for col in model_features if col in features_df.columns]
                    prediction_data = features_df[available_features]
                else:
                    # Fallback: usar features numéricas exceto target
                    numeric_features = features_df.select_dtypes(include=[np.number])
                    if 'quantidade' in numeric_features.columns:
                        prediction_data = numeric_features.drop('quantidade', axis=1)
                    else:
                        prediction_data = numeric_features

                predictions = model.predict(prediction_data)

                # Criar DataFrame de resultado
                result_df = features_df[['pdv', 'produto', 'semana']].copy()
                result_df['quantidade'] = predictions
                predictions = result_df

            except Exception as e:
                logger.error(f"Erro na predição direta do modelo {model_type}: {e}")
                # Fallback: previsões zeradas
                result_df = features_df[['pdv', 'produto', 'semana']].copy()
                result_df['quantidade'] = 0
                predictions = result_df
        else:
            # Usar ensemble
            ensemble_config = strategy_config.get('ensemble', {})
            weights = {
                model_type: strategy_config['models'][model_type].get('weight', 1.0)
                for model_type in models.keys()
            }
            
            ensemble = EnsembleModel(models, weights)
            predictions = ensemble.predict(features_df)
        
        # Pós-processamento
        predictions = self._apply_post_processing(predictions, strategy_config)
        
        logger.info(f"Previsões geradas: {len(predictions)} registros")
        return predictions
    
    def _apply_post_processing(self, predictions: pd.DataFrame, 
                             strategy_config: Dict[str, Any]) -> pd.DataFrame:
        """Aplica pós-processamento nas previsões."""
        post_config = strategy_config.get('post_processing', {})
        
        # Garantir valores positivos
        predictions['quantidade'] = predictions['quantidade'].clip(lower=0)
        
        # Aplicar suavização se configurada
        smoothing_factor = post_config.get('smoothing_factor', 0)
        if smoothing_factor > 0:
            # Suavização exponencial simples
            predictions['quantidade'] = predictions['quantidade'].ewm(
                alpha=smoothing_factor
            ).mean()
        
        # Aplicar limites de outliers
        outlier_multiplier = post_config.get('outlier_cap_multiplier', 3.0)
        if outlier_multiplier > 0:
            # Calcular limite baseado na média histórica
            mean_quantity = predictions['quantidade'].mean()
            max_limit = mean_quantity * outlier_multiplier
            predictions['quantidade'] = predictions['quantidade'].clip(upper=max_limit)
        
        return predictions
    
    def _calculate_performance_metrics(self, predictions_df: pd.DataFrame, 
                                     features_df: pd.DataFrame, 
                                     strategy_config: Dict[str, Any]) -> Dict[str, float]:
        """Calcula métricas de performance usando validação cruzada."""
        from ..models.validation import ModelValidator
        
        validator = ModelValidator()
        
        # Usar validação temporal rápida
        metrics = validator.quick_validation(predictions_df, features_df)
        
        logger.info(f"Métricas calculadas: {metrics}")
        return metrics
    
    def _extract_model_parameters(self, models: Dict[str, Any]) -> Dict[str, Any]:
        """Extrai parâmetros dos modelos treinados."""
        parameters = {}
        
        for model_type, model in models.items():
            try:
                if hasattr(model, 'get_params'):
                    parameters[model_type] = model.get_params()
                elif hasattr(model, 'params'):
                    parameters[model_type] = model.params
                else:
                    parameters[model_type] = str(type(model))
            except Exception as e:
                logger.warning(f"Erro ao extrair parâmetros do modelo {model_type}: {e}")
                parameters[model_type] = "N/A"
        
        return parameters
    
    def _reduce_hyperparameter_space(self, hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """Reduz espaço de busca de hiperparâmetros para execução rápida."""
        reduced = hyperparameters.copy()
        
        # Reduzir n_estimators para modelos de árvore
        if 'n_estimators' in reduced:
            reduced['n_estimators'] = min(reduced['n_estimators'], 500)
        
        # Reduzir profundidade máxima
        if 'max_depth' in reduced:
            reduced['max_depth'] = min(reduced['max_depth'], 6)
        
        return reduced
    
    def _get_data_hash(self, data: pd.DataFrame) -> str:
        """Gera hash dos dados para cache."""
        # Usar shape e algumas estatísticas para gerar hash
        data_info = f"{data.shape}_{data.dtypes.to_dict()}_{data.describe().to_dict()}"
        return hashlib.md5(data_info.encode()).hexdigest()
    
    def clear_cache(self, cache_type: str = "all") -> None:
        """Limpa cache especificado."""
        if cache_type in ["all", "features"]:
            import shutil
            if self.feature_cache.cache_dir.exists():
                shutil.rmtree(self.feature_cache.cache_dir)
                self.feature_cache.cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Cache de features limpo")
        
        if cache_type in ["all", "models"]:
            import shutil
            if self.model_cache.cache_dir.exists():
                shutil.rmtree(self.model_cache.cache_dir)
                self.model_cache.cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Cache de modelos limpo")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Obtém estatísticas dos caches."""
        stats = {
            'features_cache': {
                'files': len(list(self.feature_cache.cache_dir.glob("*.pkl"))),
                'size_mb': sum(f.stat().st_size for f in self.feature_cache.cache_dir.glob("*.pkl")) / 1024 / 1024
            },
            'models_cache': {
                'files': len(list(self.model_cache.cache_dir.glob("*.pkl"))),
                'size_mb': sum(f.stat().st_size for f in self.model_cache.cache_dir.glob("*.pkl")) / 1024 / 1024
            }
        }
        
        return stats