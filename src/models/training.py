"""
Módulo de treinamento de modelos para previsão de vendas.
Implementa classes para XGBoost, LightGBM e Prophet com otimização de hiperparâmetros.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from abc import ABC, abstractmethod
import logging
from datetime import datetime
import joblib
import yaml

import xgboost as xgb
import lightgbm as lgb
from prophet import Prophet
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import optuna
from optuna.samplers import TPESampler

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Classe base para todos os modelos de previsão."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.is_fitted = False
        self.feature_importance_ = None
        self.training_history = {}
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'BaseModel':
        """Treina o modelo."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Gera previsões."""
        pass
    
    def save_model(self, filepath: str) -> None:
        """Salva o modelo treinado."""
        if not self.is_fitted:
            raise ValueError("Modelo deve ser treinado antes de ser salvo")
        joblib.dump(self.model, filepath)
        logger.info(f"Modelo salvo em: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Carrega modelo salvo."""
        self.model = joblib.load(filepath)
        self.is_fitted = True
        logger.info(f"Modelo carregado de: {filepath}")


class XGBoostModel(BaseModel):
    """Implementação do modelo XGBoost com otimização de hiperparâmetros."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_config = config.get('models', {}).get('xgboost', {})
        self.validation_config = config.get('validation', {})
        self.tuning_config = config.get('hyperparameter_tuning', {})
        
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, 
            y_val: Optional[pd.Series] = None,
            optimize_hyperparams: bool = True) -> 'XGBoostModel':
        """
        Treina o modelo XGBoost com validação temporal.
        
        Args:
            X: Features de treino
            y: Target de treino
            X_val: Features de validação (opcional)
            y_val: Target de validação (opcional)
            optimize_hyperparams: Se deve otimizar hiperparâmetros
        """
        logger.info("Iniciando treinamento do modelo XGBoost")
        
        if optimize_hyperparams and self.tuning_config.get('enabled', False):
            best_params = self._optimize_hyperparameters(X, y)
            self.model_config.update(best_params)
        
        # Configurar parâmetros do modelo
        params = {
            'n_estimators': self.model_config.get('n_estimators', 1000),
            'max_depth': self.model_config.get('max_depth', 6),
            'learning_rate': self.model_config.get('learning_rate', 0.1),
            'subsample': self.model_config.get('subsample', 0.8),
            'colsample_bytree': self.model_config.get('colsample_bytree', 0.8),
            'random_state': self.model_config.get('random_state', 42),
            'n_jobs': self.config.get('general', {}).get('n_jobs', -1),
            'eval_metric': self.model_config.get('eval_metric', 'mae')
        }
        
        # Adicionar early stopping aos parâmetros se dados de validação fornecidos
        if X_val is not None and y_val is not None:
            params['early_stopping_rounds'] = self.model_config.get('early_stopping_rounds', 50)
        
        self.model = xgb.XGBRegressor(**params)
        
        # Treinar modelo com early stopping se dados de validação fornecidos
        if X_val is not None and y_val is not None:
            self.model.fit(
                X, y,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X, y)
        self.is_fitted = True
        
        # Armazenar importância das features
        self.feature_importance_ = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("Treinamento do XGBoost concluído")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Gera previsões usando o modelo treinado."""
        if not self.is_fitted:
            raise ValueError("Modelo deve ser treinado antes de gerar previsões")
        return self.model.predict(X)
    
    def _optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Otimiza hiperparâmetros XGBoost usando Optuna com validação temporal.
        
        Executa busca bayesiana de hiperparâmetros usando Optuna,
        com validação cruzada temporal para evitar data leakage.
        
        Args:
            X: Features de treinamento (shape: [n_samples, n_features])
            y: Target variable (shape: [n_samples])
            
        Returns:
            Dicionário com melhores hiperparâmetros encontrados
            
        Note:
            Usa WMAPE como métrica de otimização (métrica oficial do hackathon).
            O processo pode levar vários minutos dependendo do search_space.
        """
        logger.info("Iniciando otimização de hiperparâmetros para XGBoost")
        
        def objective(trial):
            # Definir espaço de busca
            search_space = self.tuning_config.get('search_space', {}).get('xgboost', {})
            
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 
                                                search_space.get('n_estimators', [500, 2000])[0],
                                                search_space.get('n_estimators', [500, 2000])[1]),
                'max_depth': trial.suggest_int('max_depth',
                                             search_space.get('max_depth', [3, 10])[0],
                                             search_space.get('max_depth', [3, 10])[1]),
                'learning_rate': trial.suggest_float('learning_rate',
                                                   search_space.get('learning_rate', [0.01, 0.3])[0],
                                                   search_space.get('learning_rate', [0.01, 0.3])[1]),
                'subsample': trial.suggest_float('subsample',
                                               search_space.get('subsample', [0.6, 1.0])[0],
                                               search_space.get('subsample', [0.6, 1.0])[1]),
                'colsample_bytree': trial.suggest_float('colsample_bytree',
                                                      search_space.get('colsample_bytree', [0.6, 1.0])[0],
                                                      search_space.get('colsample_bytree', [0.6, 1.0])[1]),
                'random_state': 42,
                'n_jobs': -1
            }
            
            # Validação cruzada temporal
            scores = self._temporal_cross_validation(X, y, params)
            return np.mean(scores)
        
        # Configurar estudo Optuna
        sampler = TPESampler(seed=42)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        
        # Executar otimização
        n_trials = self.tuning_config.get('n_trials', 100)
        timeout = self.tuning_config.get('timeout', 3600)
        
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        logger.info(f"Melhor score: {study.best_value:.4f}")
        logger.info(f"Melhores parâmetros: {study.best_params}")
        
        return study.best_params
    
    def _temporal_cross_validation(self, X: pd.DataFrame, y: pd.Series, 
                                 params: Dict[str, Any]) -> List[float]:
        """
        Executa validação cruzada temporal respeitando ordem cronológica.
        
        Implementa walk-forward validation para séries temporais,
        garantindo que dados futuros não sejam usados para treinar
        modelos que predizem o passado.
        
        Args:
            X: Features com coluna de data para ordenação temporal
            y: Target variable correspondente
            params: Hiperparâmetros do modelo para avaliação
            
        Returns:
            Lista com scores WMAPE de cada fold da validação
            
        Note:
            Usa 5 folds por padrão com tamanho crescente de treino.
            Cada fold testa em período subsequente ao treinamento.
        """
        n_splits = self.validation_config.get('n_splits', 5)
        test_size = self.validation_config.get('test_size', 4)
        
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Adicionar early stopping aos parâmetros
            cv_params = params.copy()
            cv_params['early_stopping_rounds'] = 50
            
            model = xgb.XGBRegressor(**cv_params)
            model.fit(X_train, y_train, 
                     eval_set=[(X_val, y_val)],
                     verbose=False)
            
            y_pred = model.predict(X_val)
            score = self._calculate_wmape(y_val, y_pred)
            scores.append(score)
        
        return scores


class ModelEvaluator:
    """Classe para avaliação de modelos com múltiplas métricas."""
    
    @staticmethod
    def calculate_wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calcula WMAPE (Weighted Mean Absolute Percentage Error).
        
        WMAPE é a métrica oficial do Hackathon Forecast 2025.
        Diferente do MAPE tradicional, o WMAPE pondera os erros
        pelo valor absoluto dos dados reais.
        
        Args:
            y_true: Valores reais (shape: [n_samples])
            y_pred: Valores preditos (shape: [n_samples])
            
        Returns:
            WMAPE como percentual (0-100)
            
        Formula:
            WMAPE = (Σ|y_true - y_pred|) / (Σ|y_true|) * 100
            
        Example:
            >>> wmape = calculate_wmape(y_true, y_pred)
            >>> print(f"WMAPE: {wmape:.2f}%")
        """
        return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100
    
    @staticmethod
    def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcula MAE (Mean Absolute Error)."""
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcula RMSE (Root Mean Square Error)."""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcula MAPE (Mean Absolute Percentage Error)."""
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    @classmethod
    def evaluate_model(cls, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calcula todas as métricas de avaliação."""
        return {
            'wmape': cls.calculate_wmape(y_true, y_pred),
            'mae': cls.calculate_mae(y_true, y_pred),
            'rmse': cls.calculate_rmse(y_true, y_pred),
            'mape': cls.calculate_mape(y_true, y_pred)
        }


class LightGBMModel(BaseModel):
    """Implementação do modelo LightGBM com otimização e early stopping."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_config = config.get('models', {}).get('lightgbm', {})
        self.validation_config = config.get('validation', {})
        self.tuning_config = config.get('hyperparameter_tuning', {})
        
    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None,
            optimize_hyperparams: bool = True) -> 'LightGBMModel':
        """
        Treina o modelo LightGBM com validação temporal.
        
        Args:
            X: Features de treino
            y: Target de treino
            X_val: Features de validação (opcional)
            y_val: Target de validação (opcional)
            optimize_hyperparams: Se deve otimizar hiperparâmetros
        """
        logger.info("Iniciando treinamento do modelo LightGBM")
        
        if optimize_hyperparams and self.tuning_config.get('enabled', False):
            best_params = self._optimize_hyperparameters(X, y)
            self.model_config.update(best_params)
        
        # Configurar parâmetros do modelo
        params = {
            'n_estimators': self.model_config.get('n_estimators', 1000),
            'max_depth': self.model_config.get('max_depth', 6),
            'learning_rate': self.model_config.get('learning_rate', 0.1),
            'subsample': self.model_config.get('subsample', 0.8),
            'colsample_bytree': self.model_config.get('colsample_bytree', 0.8),
            'random_state': self.model_config.get('random_state', 42),
            'n_jobs': self.config.get('general', {}).get('n_jobs', -1),
            'metric': self.model_config.get('metric', 'mae'),
            'verbose': self.model_config.get('verbose', -1),
            'force_col_wise': True
        }
        
        self.model = lgb.LGBMRegressor(**params)
        
        # Treinar modelo com early stopping se dados de validação fornecidos
        if X_val is not None and y_val is not None:
            self.model.fit(
                X, y,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(self.model_config.get('early_stopping_rounds', 50))]
            )
        else:
            self.model.fit(X, y)
        self.is_fitted = True
        
        # Armazenar importância das features
        self.feature_importance_ = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("Treinamento do LightGBM concluído")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Gera previsões usando o modelo treinado."""
        if not self.is_fitted:
            raise ValueError("Modelo deve ser treinado antes de gerar previsões")
        return self.model.predict(X)
    
    def _optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Otimiza hiperparâmetros usando Optuna."""
        logger.info("Iniciando otimização de hiperparâmetros para LightGBM")
        
        def objective(trial):
            # Definir espaço de busca
            search_space = self.tuning_config.get('search_space', {}).get('lightgbm', {})
            
            params = {
                'n_estimators': trial.suggest_int('n_estimators',
                                                search_space.get('n_estimators', [500, 2000])[0],
                                                search_space.get('n_estimators', [500, 2000])[1]),
                'max_depth': trial.suggest_int('max_depth',
                                             search_space.get('max_depth', [3, 10])[0],
                                             search_space.get('max_depth', [3, 10])[1]),
                'learning_rate': trial.suggest_float('learning_rate',
                                                   search_space.get('learning_rate', [0.01, 0.3])[0],
                                                   search_space.get('learning_rate', [0.01, 0.3])[1]),
                'subsample': trial.suggest_float('subsample',
                                               search_space.get('subsample', [0.6, 1.0])[0],
                                               search_space.get('subsample', [0.6, 1.0])[1]),
                'colsample_bytree': trial.suggest_float('colsample_bytree',
                                                      search_space.get('colsample_bytree', [0.6, 1.0])[0],
                                                      search_space.get('colsample_bytree', [0.6, 1.0])[1]),
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1,
                'force_col_wise': True
            }
            
            # Validação cruzada temporal
            scores = self._temporal_cross_validation(X, y, params)
            return np.mean(scores)
        
        # Configurar estudo Optuna
        sampler = TPESampler(seed=42)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        
        # Executar otimização
        n_trials = self.tuning_config.get('n_trials', 100)
        timeout = self.tuning_config.get('timeout', 3600)
        
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        logger.info(f"Melhor score: {study.best_value:.4f}")
        logger.info(f"Melhores parâmetros: {study.best_params}")
        
        return study.best_params
    
    def _temporal_cross_validation(self, X: pd.DataFrame, y: pd.Series,
                                 params: Dict[str, Any]) -> List[float]:
        """Executa validação cruzada temporal."""
        n_splits = self.validation_config.get('n_splits', 5)
        test_size = self.validation_config.get('test_size', 4)
        
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train,
                     eval_set=[(X_val, y_val)],
                     callbacks=[lgb.early_stopping(50)])
            
            y_pred = model.predict(X_val)
            score = self._calculate_wmape(y_val, y_pred)
            scores.append(score)
        
        return scores


class ProphetModel(BaseModel):
    """Wrapper para Prophet adaptado ao problema de previsão de vendas."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_config = config.get('models', {}).get('prophet', {})
        self.validation_config = config.get('validation', {})
        
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'ProphetModel':
        """
        Treina o modelo Prophet.
        
        Args:
            X: DataFrame com coluna 'ds' (data) e features adicionais
            y: Target (vendas)
        """
        logger.info("Iniciando treinamento do modelo Prophet")
        
        # Preparar dados no formato Prophet
        df_prophet = self._prepare_prophet_data(X, y)
        
        # Configurar parâmetros do Prophet
        prophet_params = {
            'seasonality_mode': self.model_config.get('seasonality_mode', 'multiplicative'),
            'yearly_seasonality': self.model_config.get('yearly_seasonality', True),
            'weekly_seasonality': self.model_config.get('weekly_seasonality', True),
            'daily_seasonality': self.model_config.get('daily_seasonality', False),
            'changepoint_prior_scale': self.model_config.get('changepoint_prior_scale', 0.05),
            'seasonality_prior_scale': self.model_config.get('seasonality_prior_scale', 10.0)
        }
        
        self.model = Prophet(**prophet_params)
        
        # Adicionar regressores externos se disponíveis
        external_regressors = self._get_external_regressors(X)
        for regressor in external_regressors:
            self.model.add_regressor(regressor)
        
        # Treinar modelo
        self.model.fit(df_prophet)
        self.is_fitted = True
        
        logger.info("Treinamento do Prophet concluído")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Gera previsões usando o modelo Prophet."""
        if not self.is_fitted:
            raise ValueError("Modelo deve ser treinado antes de gerar previsões")
        
        # Preparar dados para predição
        future = self._prepare_future_data(X)
        
        # Gerar previsões
        forecast = self.model.predict(future)
        
        return forecast['yhat'].values
    
    def _prepare_prophet_data(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Prepara dados no formato esperado pelo Prophet."""
        df = pd.DataFrame({
            'ds': pd.to_datetime(X.index) if 'ds' not in X.columns else X['ds'],
            'y': y.values
        })
        
        # Adicionar regressores externos
        external_regressors = self._get_external_regressors(X)
        for regressor in external_regressors:
            if regressor in X.columns:
                df[regressor] = X[regressor].values
        
        return df
    
    def _prepare_future_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prepara dados futuros para predição."""
        future = pd.DataFrame({
            'ds': pd.to_datetime(X.index) if 'ds' not in X.columns else X['ds']
        })
        
        # Adicionar regressores externos
        external_regressors = self._get_external_regressors(X)
        for regressor in external_regressors:
            if regressor in X.columns:
                future[regressor] = X[regressor].values
        
        return future
    
    def _get_external_regressors(self, X: pd.DataFrame) -> List[str]:
        """Identifica regressores externos válidos para Prophet."""
        # Lista de features que podem ser usadas como regressores
        potential_regressors = [col for col in X.columns 
                              if col not in ['ds', 'pdv', 'produto', 'semana']]
        
        # Filtrar apenas features numéricas
        numeric_regressors = []
        for col in potential_regressors:
            if X[col].dtype in ['int64', 'float64']:
                numeric_regressors.append(col)
        
        return numeric_regressors[:5]  # Limitar a 5 regressores para evitar overfitting


# Adicionar métodos WMAPE às classes
XGBoostModel._calculate_wmape = staticmethod(ModelEvaluator.calculate_wmape)
LightGBMModel._calculate_wmape = staticmethod(ModelEvaluator.calculate_wmape)


class ModelTrainer:
    """Classe para treinamento de modelos de ML compatível com FastSubmissionPipeline."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def train_xgboost(self, data: pd.DataFrame, hyperparameters: Dict[str, Any]):
        """Treina modelo XGBoost."""
        # Preparar dados removendo colunas não numéricas
        numeric_data = data.select_dtypes(include=[np.number])
        if 'quantidade' not in numeric_data.columns:
            # Se quantidade não estiver nas colunas numéricas, tentar usar dados originais
            X = data.drop('quantidade', axis=1, errors='ignore')
            y = data['quantidade']
        else:
            X = numeric_data.drop('quantidade', axis=1, errors='ignore')
            y = numeric_data['quantidade']

        model = XGBoostModel({})
        model.fit(X, y)
        return model.model  # Retornar o modelo interno

    def train_lightgbm(self, data: pd.DataFrame, hyperparameters: Dict[str, Any]):
        """Treina modelo LightGBM."""
        # Preparar dados removendo colunas não numéricas
        numeric_data = data.select_dtypes(include=[np.number])
        if 'quantidade' not in numeric_data.columns:
            X = data.drop('quantidade', axis=1, errors='ignore')
            y = data['quantidade']
        else:
            X = numeric_data.drop('quantidade', axis=1, errors='ignore')
            y = numeric_data['quantidade']

        model = LightGBMModel({})
        model.fit(X, y)
        return model.model  # Retornar o modelo interno

    def train_prophet(self, data: pd.DataFrame, hyperparameters: Dict[str, Any]):
        """Treina modelo Prophet."""
        model = ProphetModel({})
        model.fit(data.drop('quantidade', axis=1, errors='ignore'), data['quantidade'])
        return model.model  # Retornar o modelo interno