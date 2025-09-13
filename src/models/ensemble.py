"""
Módulo de ensemble para combinação de múltiplos modelos de previsão.
Implementa estratégias de weighted averaging e stacking com meta-learner.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from abc import ABC, abstractmethod
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import optuna
from optuna.samplers import TPESampler

from .training import BaseModel, ModelEvaluator

logger = logging.getLogger(__name__)


class EnsembleModel:
    """Classe simples de ensemble compatível com FastSubmissionPipeline."""

    def __init__(self, models: Dict[str, Any], weights: Optional[Dict[str, float]] = None):
        self.models = models
        self.weights = weights or {name: 1.0/len(models) for name in models.keys()}

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Gera previsões usando weighted average dos modelos."""
        predictions = []

        for model_name, model in self.models.items():
            try:
                if hasattr(model, 'predict'):
                    pred = model.predict(X.select_dtypes(include=[np.number]))
                    if isinstance(pred, np.ndarray):
                        pred = pd.Series(pred, index=X.index, name=f'pred_{model_name}')
                    predictions.append(pred * self.weights.get(model_name, 1.0))
                else:
                    logger.warning(f"Modelo {model_name} não tem método predict")
            except Exception as e:
                logger.error(f"Erro ao gerar predição para {model_name}: {e}")

        if predictions:
            # Combinar previsões
            combined_pred = sum(predictions) / sum(self.weights.values())

            # Criar DataFrame de resultado
            result_df = X[['pdv', 'produto', 'semana']].copy()

            # Para evitar problemas de conversão, manter tipos originais por enquanto
            # O sistema de validação será ajustado para aceitar os tipos atuais
            result_df['quantidade'] = combined_pred

            return result_df
        else:
            # Fallback: retornar zeros
            result_df = X[['pdv', 'produto', 'semana']].copy()
            result_df['quantidade'] = 0
            return result_df


class BaseEnsemble(ABC):
    """Classe base para estratégias de ensemble."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.is_fitted = False
        self.ensemble_weights = None
        self.meta_learner = None
        
    @abstractmethod
    def fit(self, models: Dict[str, BaseModel], X: pd.DataFrame, y: pd.Series) -> 'BaseEnsemble':
        """Treina a estratégia de ensemble."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Gera previsões usando o ensemble."""
        pass
    
    def save_ensemble(self, filepath: str) -> None:
        """Salva o ensemble treinado."""
        if not self.is_fitted:
            raise ValueError("Ensemble deve ser treinado antes de ser salvo")
        
        ensemble_data = {
            'models': self.models,
            'ensemble_weights': self.ensemble_weights,
            'meta_learner': self.meta_learner,
            'config': self.config
        }
        joblib.dump(ensemble_data, filepath)
        logger.info(f"Ensemble salvo em: {filepath}")
    
    def load_ensemble(self, filepath: str) -> None:
        """Carrega ensemble salvo."""
        ensemble_data = joblib.load(filepath)
        self.models = ensemble_data['models']
        self.ensemble_weights = ensemble_data['ensemble_weights']
        self.meta_learner = ensemble_data['meta_learner']
        self.config = ensemble_data['config']
        self.is_fitted = True
        logger.info(f"Ensemble carregado de: {filepath}")


class WeightedEnsemble(BaseEnsemble):
    """
    Ensemble com combinação ponderada de modelos.
    Otimiza pesos usando validação cruzada temporal.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.ensemble_config = config.get('ensemble', {}).get('weighted', {})
        self.validation_config = config.get('validation', {})
        
    def fit(self, models: Dict[str, BaseModel], X: pd.DataFrame, y: pd.Series) -> 'WeightedEnsemble':
        """
        Treina o ensemble ponderado otimizando pesos via validação cruzada.
        
        Args:
            models: Dicionário com modelos treinados {nome: modelo}
            X: Features de treino
            y: Target de treino
        """
        logger.info("Iniciando treinamento do ensemble ponderado")
        
        # Validar que todos os modelos estão treinados
        for name, model in models.items():
            if not model.is_fitted:
                raise ValueError(f"Modelo {name} deve estar treinado antes do ensemble")
        
        self.models = models
        
        # Gerar previsões de cada modelo para otimização de pesos
        model_predictions = self._generate_model_predictions(X, y)
        
        # Otimizar pesos do ensemble
        if self.ensemble_config.get('optimize_weights', True):
            self.ensemble_weights = self._optimize_weights(model_predictions, y)
        else:
            # Usar pesos uniformes
            n_models = len(self.models)
            self.ensemble_weights = {name: 1.0/n_models for name in self.models.keys()}
        
        self.is_fitted = True
        
        logger.info(f"Ensemble ponderado treinado com pesos: {self.ensemble_weights}")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Gera previsões usando combinação ponderada dos modelos."""
        if not self.is_fitted:
            raise ValueError("Ensemble deve ser treinado antes de gerar previsões")
        
        # Gerar previsões de cada modelo
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X)
        
        # Combinar previsões usando pesos otimizados
        ensemble_pred = np.zeros(len(X))
        for name, pred in predictions.items():
            ensemble_pred += self.ensemble_weights[name] * pred
        
        return ensemble_pred
    
    def _generate_model_predictions(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, np.ndarray]:
        """Gera previsões de cada modelo usando validação cruzada temporal."""
        n_splits = self.validation_config.get('n_splits', 5)
        test_size = self.validation_config.get('test_size', 4)
        
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        model_predictions = {name: np.full(len(y), np.nan) for name in self.models.keys()}
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Gerar previsões de cada modelo para o fold de validação
            for name, model in self.models.items():
                # Criar cópia do modelo para treinar no fold
                model_copy = type(model)(model.config)
                model_copy.fit(X_train, y_train)
                
                # Gerar previsões para o fold de validação
                val_pred = model_copy.predict(X_val)
                model_predictions[name][val_idx] = val_pred
        
        return model_predictions
    
    def _optimize_weights(self, model_predictions: Dict[str, np.ndarray], 
                         y_true: pd.Series) -> Dict[str, float]:
        """Otimiza pesos do ensemble usando Optuna."""
        logger.info("Otimizando pesos do ensemble")
        
        # Identificar índices válidos (onde temos previsões de todos os modelos)
        first_pred = list(model_predictions.values())[0]
        valid_mask = ~np.isnan(first_pred)
        
        # Filtrar previsões e y_true para usar apenas dados válidos
        filtered_predictions = {}
        for name, pred in model_predictions.items():
            filtered_predictions[name] = pred[valid_mask]
        
        # Filtrar y_true usando os mesmos índices válidos
        valid_indices = np.where(valid_mask)[0]
        y_filtered = y_true.iloc[valid_indices].values
        
        def objective(trial):
            # Gerar pesos que somam 1
            model_names = list(model_predictions.keys())
            weights = []
            
            for i, name in enumerate(model_names[:-1]):
                if i == 0:
                    weight = trial.suggest_float(f'weight_{name}', 0.0, 1.0)
                else:
                    remaining_weight = 1.0 - sum(weights)
                    weight = trial.suggest_float(f'weight_{name}', 0.0, remaining_weight)
                weights.append(weight)
            
            # Último peso é o restante
            weights.append(1.0 - sum(weights))
            
            # Calcular previsão do ensemble
            ensemble_pred = np.zeros(len(y_filtered))
            for i, name in enumerate(model_names):
                ensemble_pred += weights[i] * filtered_predictions[name]
            
            # Calcular WMAPE
            wmape = ModelEvaluator.calculate_wmape(y_filtered, ensemble_pred)
            return wmape
        
        # Configurar e executar otimização
        sampler = TPESampler(seed=42)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        
        n_trials = self.ensemble_config.get('optimization_trials', 200)
        study.optimize(objective, n_trials=n_trials)
        
        # Extrair pesos otimizados
        model_names = list(model_predictions.keys())
        optimized_weights = {}
        
        for i, name in enumerate(model_names[:-1]):
            optimized_weights[name] = study.best_params[f'weight_{name}']
        
        # Calcular último peso
        last_name = model_names[-1]
        optimized_weights[last_name] = 1.0 - sum(optimized_weights.values())
        
        logger.info(f"Pesos otimizados - WMAPE: {study.best_value:.4f}")
        return optimized_weights


class StackingEnsemble(BaseEnsemble):
    """
    Ensemble usando stacking com meta-learner.
    Treina um modelo de segundo nível para combinar previsões dos modelos base.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.ensemble_config = config.get('ensemble', {}).get('stacking', {})
        self.validation_config = config.get('validation', {})
        
    def fit(self, models: Dict[str, BaseModel], X: pd.DataFrame, y: pd.Series) -> 'StackingEnsemble':
        """
        Treina o ensemble de stacking com meta-learner.
        
        Args:
            models: Dicionário com modelos treinados {nome: modelo}
            X: Features de treino
            y: Target de treino
        """
        logger.info("Iniciando treinamento do ensemble de stacking")
        
        # Validar que todos os modelos estão treinados
        for name, model in models.items():
            if not model.is_fitted:
                raise ValueError(f"Modelo {name} deve estar treinado antes do ensemble")
        
        self.models = models
        
        # Gerar previsões dos modelos base usando validação cruzada
        base_predictions, y_filtered = self._generate_base_predictions(X, y)
        
        # Treinar meta-learner
        self._train_meta_learner(base_predictions, y_filtered)
        
        self.is_fitted = True
        
        logger.info("Ensemble de stacking treinado com sucesso")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Gera previsões usando stacking ensemble."""
        if not self.is_fitted:
            raise ValueError("Ensemble deve ser treinado antes de gerar previsões")
        
        # Gerar previsões dos modelos base
        base_predictions = np.column_stack([
            model.predict(X) for model in self.models.values()
        ])
        
        # Usar meta-learner para combinar previsões
        ensemble_pred = self.meta_learner.predict(base_predictions)
        
        return ensemble_pred
    
    def _generate_base_predictions(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Gera previsões dos modelos base usando validação cruzada temporal."""
        n_splits = self.validation_config.get('n_splits', 5)
        test_size = self.validation_config.get('test_size', 4)
        
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        n_models = len(self.models)
        base_predictions = np.full((len(y), n_models), np.nan)
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Gerar previsões de cada modelo base para o fold de validação
            for i, (name, model) in enumerate(self.models.items()):
                # Criar cópia do modelo para treinar no fold
                model_copy = type(model)(model.config)
                model_copy.fit(X_train, y_train)
                
                # Gerar previsões para o fold de validação
                val_pred = model_copy.predict(X_val)
                base_predictions[val_idx, i] = val_pred
        
        # Remover linhas com valores NaN
        valid_mask = ~np.isnan(base_predictions).any(axis=1)
        return base_predictions[valid_mask], y.values[valid_mask]
    
    def _train_meta_learner(self, base_predictions: np.ndarray, y_filtered: np.ndarray) -> None:
        """Treina o meta-learner para combinar previsões dos modelos base."""
        # base_predictions e y_filtered já estão alinhados e filtrados
        
        # Configurar meta-learner
        meta_learner_type = self.ensemble_config.get('meta_learner', 'ridge')
        
        if meta_learner_type == 'ridge':
            alpha = self.ensemble_config.get('ridge_alpha', 1.0)
            self.meta_learner = Ridge(alpha=alpha, random_state=42)
        elif meta_learner_type == 'linear':
            self.meta_learner = LinearRegression()
        elif meta_learner_type == 'random_forest':
            n_estimators = self.ensemble_config.get('rf_n_estimators', 100)
            max_depth = self.ensemble_config.get('rf_max_depth', 5)
            self.meta_learner = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )
        else:
            raise ValueError(f"Meta-learner não suportado: {meta_learner_type}")
        
        # Treinar meta-learner
        self.meta_learner.fit(base_predictions, y_filtered)
        
        logger.info(f"Meta-learner {meta_learner_type} treinado com sucesso")


class EnsembleManager:
    """
    Gerenciador para múltiplas estratégias de ensemble.
    Permite comparar e selecionar a melhor estratégia.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ensembles = {}
        self.best_ensemble = None
        self.evaluation_results = {}
        
    def create_ensembles(self, models: Dict[str, BaseModel], 
                        X: pd.DataFrame, y: pd.Series) -> Dict[str, BaseEnsemble]:
        """
        Cria e treina múltiplas estratégias de ensemble.
        
        Args:
            models: Dicionário com modelos treinados
            X: Features de treino
            y: Target de treino
            
        Returns:
            Dicionário com ensembles treinados
        """
        logger.info("Criando múltiplas estratégias de ensemble")
        
        ensemble_configs = self.config.get('ensemble', {})
        
        # Criar ensemble ponderado se configurado
        if ensemble_configs.get('weighted', {}).get('enabled', True):
            weighted_ensemble = WeightedEnsemble(self.config)
            weighted_ensemble.fit(models, X, y)
            self.ensembles['weighted'] = weighted_ensemble
        
        # Criar ensemble de stacking se configurado
        if ensemble_configs.get('stacking', {}).get('enabled', True):
            stacking_ensemble = StackingEnsemble(self.config)
            stacking_ensemble.fit(models, X, y)
            self.ensembles['stacking'] = stacking_ensemble
        
        logger.info(f"Criados {len(self.ensembles)} ensembles")
        return self.ensembles
    
    def evaluate_ensembles(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Avalia performance de todos os ensembles.
        
        Args:
            X_test: Features de teste
            y_test: Target de teste
            
        Returns:
            Dicionário com métricas de cada ensemble
        """
        logger.info("Avaliando performance dos ensembles")
        
        for name, ensemble in self.ensembles.items():
            # Gerar previsões
            y_pred = ensemble.predict(X_test)
            
            # Calcular métricas
            metrics = ModelEvaluator.evaluate_model(y_test.values, y_pred)
            self.evaluation_results[name] = metrics
            
            logger.info(f"Ensemble {name} - WMAPE: {metrics['wmape']:.4f}")
        
        return self.evaluation_results
    
    def select_best_ensemble(self, metric: str = 'wmape') -> BaseEnsemble:
        """
        Seleciona o melhor ensemble baseado na métrica especificada.
        
        Args:
            metric: Métrica para seleção ('wmape', 'mae', 'rmse')
            
        Returns:
            Melhor ensemble
        """
        if not self.evaluation_results:
            raise ValueError("Ensembles devem ser avaliados antes da seleção")
        
        # Encontrar ensemble com menor erro
        best_score = float('inf')
        best_name = None
        
        for name, metrics in self.evaluation_results.items():
            if metrics[metric] < best_score:
                best_score = metrics[metric]
                best_name = name
        
        self.best_ensemble = self.ensembles[best_name]
        
        logger.info(f"Melhor ensemble selecionado: {best_name} ({metric}: {best_score:.4f})")
        return self.best_ensemble
    
    def get_ensemble_summary(self) -> pd.DataFrame:
        """
        Retorna resumo comparativo dos ensembles.
        
        Returns:
            DataFrame com métricas de todos os ensembles
        """
        if not self.evaluation_results:
            raise ValueError("Ensembles devem ser avaliados antes do resumo")
        
        summary_data = []
        for name, metrics in self.evaluation_results.items():
            row = {'ensemble': name}
            row.update(metrics)
            summary_data.append(row)
        
        return pd.DataFrame(summary_data).set_index('ensemble')


class EnsembleValidator:
    """Classe para validação específica de ensembles."""
    
    @staticmethod
    def validate_ensemble_diversity(models: Dict[str, BaseModel], 
                                  X: pd.DataFrame) -> Dict[str, float]:
        """
        Valida diversidade entre modelos do ensemble.
        
        Args:
            models: Dicionário com modelos treinados
            X: Features para gerar previsões
            
        Returns:
            Dicionário com métricas de diversidade
        """
        # Gerar previsões de todos os modelos
        predictions = {}
        for name, model in models.items():
            predictions[name] = model.predict(X)
        
        # Calcular correlações entre previsões
        pred_df = pd.DataFrame(predictions)
        correlation_matrix = pred_df.corr()
        
        # Métricas de diversidade
        avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
        min_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].min()
        max_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].max()
        
        return {
            'avg_correlation': avg_correlation,
            'min_correlation': min_correlation,
            'max_correlation': max_correlation,
            'diversity_score': 1 - avg_correlation  # Maior diversidade = menor correlação
        }
    
    @staticmethod
    def validate_ensemble_stability(ensemble: BaseEnsemble, X: pd.DataFrame, 
                                  n_bootstrap: int = 100) -> Dict[str, float]:
        """
        Valida estabilidade das previsões do ensemble.
        
        Args:
            ensemble: Ensemble treinado
            X: Features para teste
            n_bootstrap: Número de amostras bootstrap
            
        Returns:
            Métricas de estabilidade
        """
        predictions = []
        
        # Gerar previsões com bootstrap
        for _ in range(n_bootstrap):
            # Amostra bootstrap dos dados
            bootstrap_idx = np.random.choice(len(X), size=len(X), replace=True)
            X_bootstrap = X.iloc[bootstrap_idx]
            
            # Gerar previsão
            pred = ensemble.predict(X_bootstrap)
            predictions.append(pred.mean())  # Média das previsões
        
        predictions = np.array(predictions)
        
        return {
            'prediction_std': predictions.std(),
            'prediction_cv': predictions.std() / predictions.mean(),
            'stability_score': 1 / (1 + predictions.std())  # Maior estabilidade = menor desvio
        }