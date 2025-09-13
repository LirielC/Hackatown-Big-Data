"""
Ensemble avançado para melhorar precisão do modelo de previsão de vendas.
Objetivo: Reduzir WMAPE de 68% para abaixo de 20%.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_percentage_error
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)

class AdvancedEnsemble:
    """
    Ensemble avançado com múltiplas técnicas para reduzir WMAPE.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.best_weights = None
        self.feature_names = None
        self.logger = logging.getLogger(__name__)

    def create_base_models(self) -> Dict[str, Any]:
        """Cria modelos base otimizados para ensemble."""

        models = {
            'xgboost': xgb.XGBRegressor(
                n_estimators=1000,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=-1,
                eval_metric='mae'
            ),

            'lightgbm': lgb.LGBMRegressor(
                n_estimators=1000,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            ),

            'random_forest': RandomForestRegressor(
                n_estimators=500,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=self.random_state,
                n_jobs=-1
            )
        }

        self.models = models
        return models

    def weighted_ensemble(self, predictions_dict: Dict[str, np.ndarray],
                          y_true: np.ndarray,
                          method: str = 'performance') -> np.ndarray:
        """
        Cria ensemble ponderado baseado em performance histórica.
        """

        if method == 'performance':
            # Pesos baseados na performance individual (inverso do erro)
            weights = {}
            for model_name, y_pred in predictions_dict.items():
                wmape = self._calculate_wmape(y_true, y_pred)
                weights[model_name] = 1 / (wmape + 0.001)  # Evitar divisão por zero

            # Normalizar pesos
            total_weight = sum(weights.values())
            weights = {k: v/total_weight for k, v in weights.items()}

        elif method == 'equal':
            # Pesos iguais
            n_models = len(predictions_dict)
            weights = {name: 1/n_models for name in predictions_dict.keys()}

        self.best_weights = weights
        self.logger.info(f"Pesos do ensemble ({method}): {weights}")

        # Combinar previsões
        final_predictions = np.zeros_like(y_true, dtype=float)
        for model_name, y_pred in predictions_dict.items():
            weight = weights[model_name]
            final_predictions += weight * y_pred

        return final_predictions

    def stacking_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series,
                         X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Any:
        """
        Implementa stacking ensemble com meta-modelo.
        """

        # Preparar estimadores base
        base_estimators = [
            ('xgb', self.models['xgboost']),
            ('lgb', self.models['lightgbm']),
            ('rf', self.models['random_forest'])
        ]

        # Meta-modelo
        meta_model = Ridge(alpha=0.1, random_state=self.random_state)

        # Stacking regressor
        stacking_model = StackingRegressor(
            estimators=base_estimators,
            final_estimator=meta_model,
            cv=5,
            n_jobs=-1,
            passthrough=True  # Incluir features originais no meta-modelo
        )

        # Treinar
        self.logger.info("Treinando stacking ensemble...")
        stacking_model.fit(X_train, y_train)

        return stacking_model

    def time_series_cross_validation(self, X: pd.DataFrame, y: pd.Series,
                                   n_splits: int = 5) -> Dict[str, List[float]]:
        """
        Validação cruzada específica para séries temporais.
        """

        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = {model_name: [] for model_name in self.models.keys()}

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            self.logger.info(f"Treinando fold {fold + 1}/{n_splits}")

            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

            fold_predictions = {}

            for model_name, model in self.models.items():
                # Treinar modelo
                model_clone = model.__class__(**model.get_params())
                model_clone.fit(X_train_fold, y_train_fold)

                # Fazer previsões
                y_pred = model_clone.predict(X_val_fold)

                # Calcular WMAPE
                wmape = self._calculate_wmape(y_val_fold.values, y_pred)
                cv_scores[model_name].append(wmape)

                fold_predictions[model_name] = y_pred

            # Ensemble para este fold
            ensemble_pred = self.weighted_ensemble(
                {name: fold_predictions[name] for name in self.models.keys()},
                y_val_fold.values
            )

            ensemble_wmape = self._calculate_wmape(y_val_fold.values, ensemble_pred)
            cv_scores['ensemble'].append(ensemble_wmape)

        return cv_scores

    def optimize_weights(self, X: pd.DataFrame, y: pd.Series,
                        n_trials: int = 50) -> Dict[str, float]:
        """
        Otimiza pesos do ensemble usando busca simples.
        """

        best_weights = None
        best_score = float('inf')

        np.random.seed(self.random_state)

        for trial in range(n_trials):
            # Gerar pesos aleatórios que somam 1
            weights = np.random.random(len(self.models))
            weights = weights / weights.sum()

            # Treinar modelos e fazer previsões
            predictions = {}
            for i, model_name in enumerate(self.models.keys()):
                model = self.models[model_name].__class__(**self.models[model_name].get_params())
                model.fit(X, y)
                predictions[model_name] = model.predict(X)

            # Combinar previsões com pesos
            ensemble_pred = np.zeros_like(y.values, dtype=float)
            for i, model_name in enumerate(self.models.keys()):
                ensemble_pred += weights[i] * predictions[model_name]

            # Calcular score
            wmape = self._calculate_wmape(y.values, ensemble_pred)

            if wmape < best_score:
                best_score = wmape
                best_weights = dict(zip(self.models.keys(), weights))

        self.logger.info(f"Melhores pesos encontrados: {best_weights}")
        self.logger.info(f"WMAPE com pesos otimizados: {best_score:.4f}")

        return best_weights

    def predict_with_ensemble(self, X: pd.DataFrame,
                            ensemble_type: str = 'weighted') -> np.ndarray:
        """
        Faz previsões usando ensemble.
        """

        if ensemble_type == 'weighted':
            if self.best_weights is None:
                raise ValueError("Pesos não foram definidos. Execute optimize_weights primeiro.")

            predictions = {}
            for model_name, model in self.models.items():
                predictions[model_name] = model.predict(X)

            # Combinar com pesos
            final_predictions = np.zeros(X.shape[0])
            for model_name, preds in predictions.items():
                weight = self.best_weights[model_name]
                final_predictions += weight * preds

        elif ensemble_type == 'stacking':
            # Usar modelo de stacking (deve ser implementado)
            raise NotImplementedError("Stacking ainda não implementado")

        else:
            raise ValueError(f"Tipo de ensemble não suportado: {ensemble_type}")

        return final_predictions

    def _calculate_wmape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcula Weighted Mean Absolute Percentage Error."""
        return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))

    def get_feature_importance(self, X: pd.DataFrame) -> Dict[str, pd.Series]:
        """Retorna importância das features para cada modelo."""

        importance_dict = {}

        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_dict[model_name] = pd.Series(
                    model.feature_importances_,
                    index=X.columns
                ).sort_values(ascending=False)
            elif hasattr(model, 'get_booster'):
                # XGBoost
                importance_dict[model_name] = pd.Series(
                    model.get_booster().get_score(importance_type='gain')
                ).sort_values(ascending=False)

        return importance_dict

    def summary(self) -> Dict[str, Any]:
        """Retorna resumo do ensemble."""

        return {
            'models': list(self.models.keys()),
            'weights': self.best_weights,
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'has_stacking': False,  # TODO: implementar
            'validation_method': 'time_series_cv'
        }
