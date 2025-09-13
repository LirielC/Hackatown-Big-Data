"""
Testes para o módulo de treinamento de modelos.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import tempfile
import os

from src.models.training import (
    XGBoostModel, LightGBMModel, ProphetModel, 
    ModelEvaluator, BaseModel
)


class TestModelEvaluator:
    """Testes para a classe ModelEvaluator."""
    
    def test_calculate_wmape(self):
        """Testa cálculo do WMAPE."""
        y_true = np.array([100, 200, 150, 300])
        y_pred = np.array([90, 210, 140, 320])
        
        wmape = ModelEvaluator.calculate_wmape(y_true, y_pred)
        
        expected_wmape = (abs(100-90) + abs(200-210) + abs(150-140) + abs(300-320)) / (100+200+150+300) * 100
        assert abs(wmape - expected_wmape) < 0.001
    
    def test_calculate_mae(self):
        """Testa cálculo do MAE."""
        y_true = np.array([100, 200, 150, 300])
        y_pred = np.array([90, 210, 140, 320])
        
        mae = ModelEvaluator.calculate_mae(y_true, y_pred)
        expected_mae = np.mean([10, 10, 10, 20])
        
        assert abs(mae - expected_mae) < 0.001
    
    def test_calculate_rmse(self):
        """Testa cálculo do RMSE."""
        y_true = np.array([100, 200, 150, 300])
        y_pred = np.array([90, 210, 140, 320])
        
        rmse = ModelEvaluator.calculate_rmse(y_true, y_pred)
        expected_rmse = np.sqrt(np.mean([100, 100, 100, 400]))
        
        assert abs(rmse - expected_rmse) < 0.001
    
    def test_calculate_mape(self):
        """Testa cálculo do MAPE."""
        y_true = np.array([100, 200, 150, 300])
        y_pred = np.array([90, 210, 140, 320])
        
        mape = ModelEvaluator.calculate_mape(y_true, y_pred)
        expected_mape = np.mean([10/100, 10/200, 10/150, 20/300]) * 100
        
        assert abs(mape - expected_mape) < 0.001
    
    def test_evaluate_model(self):
        """Testa avaliação completa do modelo."""
        y_true = np.array([100, 200, 150, 300])
        y_pred = np.array([90, 210, 140, 320])
        
        metrics = ModelEvaluator.evaluate_model(y_true, y_pred)
        
        assert 'wmape' in metrics
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'mape' in metrics
        assert all(isinstance(v, float) for v in metrics.values())


class TestXGBoostModel:
    """Testes para a classe XGBoostModel."""
    
    @pytest.fixture
    def sample_config(self):
        """Configuração de exemplo para testes."""
        return {
            'general': {'n_jobs': 1, 'random_seed': 42},
            'models': {
                'xgboost': {
                    'n_estimators': 10,
                    'max_depth': 3,
                    'learning_rate': 0.1,
                    'random_state': 42,
                    'early_stopping_rounds': 5
                }
            },
            'validation': {
                'n_splits': 3,
                'test_size': 2
            },
            'hyperparameter_tuning': {
                'enabled': False,
                'n_trials': 5,
                'search_space': {
                    'xgboost': {
                        'n_estimators': [5, 20],
                        'max_depth': [2, 5],
                        'learning_rate': [0.05, 0.2],
                        'subsample': [0.7, 1.0],
                        'colsample_bytree': [0.7, 1.0]
                    }
                }
            }
        }
    
    @pytest.fixture
    def sample_data(self):
        """Dados de exemplo para testes."""
        np.random.seed(42)
        n_samples = 100
        
        X = pd.DataFrame({
            'feature_1': np.random.randn(n_samples),
            'feature_2': np.random.randn(n_samples),
            'feature_3': np.random.randn(n_samples),
            'lag_1': np.random.randn(n_samples),
            'rolling_mean_4': np.random.randn(n_samples)
        })
        
        # Target com alguma correlação com as features
        y = pd.Series(
            X['feature_1'] * 2 + X['feature_2'] * 1.5 + np.random.randn(n_samples) * 0.5 + 100
        )
        
        return X, y
    
    def test_model_initialization(self, sample_config):
        """Testa inicialização do modelo."""
        model = XGBoostModel(sample_config)
        
        assert model.config == sample_config
        assert model.model is None
        assert not model.is_fitted
        assert model.feature_importance_ is None
    
    def test_model_fit_basic(self, sample_config, sample_data):
        """Testa treinamento básico do modelo."""
        X, y = sample_data
        model = XGBoostModel(sample_config)
        
        # Treinar sem otimização de hiperparâmetros
        fitted_model = model.fit(X, y, optimize_hyperparams=False)
        
        assert fitted_model.is_fitted
        assert fitted_model.model is not None
        assert fitted_model.feature_importance_ is not None
        assert len(fitted_model.feature_importance_) == len(X.columns)
    
    def test_model_fit_with_validation(self, sample_config, sample_data):
        """Testa treinamento com dados de validação."""
        X, y = sample_data
        
        # Dividir dados
        split_idx = len(X) // 2
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        model = XGBoostModel(sample_config)
        fitted_model = model.fit(X_train, y_train, X_val, y_val, optimize_hyperparams=False)
        
        assert fitted_model.is_fitted
        assert fitted_model.model is not None
    
    def test_model_predict(self, sample_config, sample_data):
        """Testa geração de previsões."""
        X, y = sample_data
        model = XGBoostModel(sample_config)
        
        # Treinar modelo
        model.fit(X, y, optimize_hyperparams=False)
        
        # Gerar previsões
        predictions = model.predict(X)
        
        assert len(predictions) == len(X)
        assert isinstance(predictions, np.ndarray)
        assert not np.isnan(predictions).any()
    
    def test_model_predict_without_fit(self, sample_config, sample_data):
        """Testa erro ao prever sem treinar."""
        X, y = sample_data
        model = XGBoostModel(sample_config)
        
        with pytest.raises(ValueError, match="Modelo deve ser treinado"):
            model.predict(X)
    
    def test_temporal_cross_validation(self, sample_config, sample_data):
        """Testa validação cruzada temporal."""
        X, y = sample_data
        model = XGBoostModel(sample_config)
        
        params = {
            'n_estimators': 10,
            'max_depth': 3,
            'learning_rate': 0.1,
            'random_state': 42,
            'n_jobs': 1
        }
        
        scores = model._temporal_cross_validation(X, y, params)
        
        assert len(scores) == sample_config['validation']['n_splits']
        assert all(isinstance(score, float) for score in scores)
        assert all(score >= 0 for score in scores)  # WMAPE deve ser positivo
    
    @patch('optuna.create_study')
    def test_hyperparameter_optimization(self, mock_create_study, sample_config, sample_data):
        """Testa otimização de hiperparâmetros."""
        X, y = sample_data
        
        # Mock do estudo Optuna
        mock_study = Mock()
        mock_study.best_value = 15.5
        mock_study.best_params = {
            'n_estimators': 15,
            'max_depth': 4,
            'learning_rate': 0.15
        }
        mock_create_study.return_value = mock_study
        
        # Habilitar otimização
        sample_config['hyperparameter_tuning']['enabled'] = True
        
        model = XGBoostModel(sample_config)
        best_params = model._optimize_hyperparameters(X, y)
        
        assert best_params == mock_study.best_params
        mock_study.optimize.assert_called_once()
    
    def test_save_and_load_model(self, sample_config, sample_data):
        """Testa salvamento e carregamento do modelo."""
        X, y = sample_data
        model = XGBoostModel(sample_config)
        
        # Treinar modelo
        model.fit(X, y, optimize_hyperparams=False)
        
        # Salvar modelo
        with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as tmp_file:
            tmp_filename = tmp_file.name
        
        try:
            model.save_model(tmp_filename)
            
            # Criar novo modelo e carregar
            new_model = XGBoostModel(sample_config)
            new_model.load_model(tmp_filename)
            
            # Testar se previsões são iguais
            pred_original = model.predict(X)
            pred_loaded = new_model.predict(X)
            
            np.testing.assert_array_almost_equal(pred_original, pred_loaded)
            
        finally:
            # Limpar arquivo temporário
            if os.path.exists(tmp_filename):
                os.unlink(tmp_filename)
    
    def test_save_model_without_fit(self, sample_config):
        """Testa erro ao salvar modelo não treinado."""
        model = XGBoostModel(sample_config)
        
        with pytest.raises(ValueError, match="Modelo deve ser treinado"):
            model.save_model("dummy_path.joblib")


class TestLightGBMModel:
    """Testes para a classe LightGBMModel."""
    
    @pytest.fixture
    def sample_config(self):
        """Configuração de exemplo para testes."""
        return {
            'general': {'n_jobs': 1, 'random_seed': 42},
            'models': {
                'lightgbm': {
                    'n_estimators': 10,
                    'max_depth': 3,
                    'learning_rate': 0.1,
                    'random_state': 42,
                    'early_stopping_rounds': 5,
                    'verbose': -1
                }
            },
            'validation': {
                'n_splits': 3,
                'test_size': 2
            },
            'hyperparameter_tuning': {
                'enabled': False
            }
        }
    
    @pytest.fixture
    def sample_data(self):
        """Dados de exemplo para testes."""
        np.random.seed(42)
        n_samples = 100
        
        X = pd.DataFrame({
            'feature_1': np.random.randn(n_samples),
            'feature_2': np.random.randn(n_samples),
            'feature_3': np.random.randn(n_samples)
        })
        
        y = pd.Series(
            X['feature_1'] * 2 + X['feature_2'] * 1.5 + np.random.randn(n_samples) * 0.5 + 100
        )
        
        return X, y
    
    def test_lightgbm_fit_and_predict(self, sample_config, sample_data):
        """Testa treinamento e predição do LightGBM."""
        X, y = sample_data
        model = LightGBMModel(sample_config)
        
        # Treinar modelo
        fitted_model = model.fit(X, y, optimize_hyperparams=False)
        
        assert fitted_model.is_fitted
        assert fitted_model.model is not None
        
        # Gerar previsões
        predictions = fitted_model.predict(X)
        
        assert len(predictions) == len(X)
        assert isinstance(predictions, np.ndarray)


class TestProphetModel:
    """Testes para a classe ProphetModel."""
    
    @pytest.fixture
    def sample_config(self):
        """Configuração de exemplo para testes."""
        return {
            'models': {
                'prophet': {
                    'seasonality_mode': 'multiplicative',
                    'yearly_seasonality': True,
                    'weekly_seasonality': True,
                    'daily_seasonality': False
                }
            }
        }
    
    @pytest.fixture
    def sample_time_series_data(self):
        """Dados de série temporal para testes."""
        dates = pd.date_range('2022-01-01', '2022-12-31', freq='W')
        n_samples = len(dates)
        
        X = pd.DataFrame({
            'ds': dates,
            'feature_1': np.random.randn(n_samples),
            'feature_2': np.random.randn(n_samples)
        })
        X.index = dates
        
        # Série temporal com tendência e sazonalidade
        trend = np.linspace(100, 200, n_samples)
        seasonal = 10 * np.sin(2 * np.pi * np.arange(n_samples) / 52)  # Sazonalidade anual
        noise = np.random.randn(n_samples) * 5
        
        y = pd.Series(trend + seasonal + noise, index=dates)
        
        return X, y
    
    def test_prophet_initialization(self, sample_config):
        """Testa inicialização do Prophet."""
        model = ProphetModel(sample_config)
        
        assert model.config == sample_config
        assert model.model is None
        assert not model.is_fitted
    
    def test_prepare_prophet_data(self, sample_config, sample_time_series_data):
        """Testa preparação dos dados para Prophet."""
        X, y = sample_time_series_data
        model = ProphetModel(sample_config)
        
        df_prophet = model._prepare_prophet_data(X, y)
        
        assert 'ds' in df_prophet.columns
        assert 'y' in df_prophet.columns
        assert len(df_prophet) == len(X)
        assert pd.api.types.is_datetime64_any_dtype(df_prophet['ds'])
    
    def test_get_external_regressors(self, sample_config, sample_time_series_data):
        """Testa identificação de regressores externos."""
        X, y = sample_time_series_data
        model = ProphetModel(sample_config)
        
        regressors = model._get_external_regressors(X)
        
        assert isinstance(regressors, list)
        assert 'feature_1' in regressors
        assert 'feature_2' in regressors
        assert 'ds' not in regressors  # ds não deve ser regressor
    
    @patch('src.models.training.Prophet')
    def test_prophet_fit(self, mock_prophet_class, sample_config, sample_time_series_data):
        """Testa treinamento do Prophet."""
        X, y = sample_time_series_data
        
        # Mock do Prophet
        mock_prophet_instance = Mock()
        mock_prophet_class.return_value = mock_prophet_instance
        
        model = ProphetModel(sample_config)
        fitted_model = model.fit(X, y)
        
        assert fitted_model.is_fitted
        mock_prophet_instance.fit.assert_called_once()
        
        # Verificar se regressores foram adicionados
        expected_calls = len(model._get_external_regressors(X))
        assert mock_prophet_instance.add_regressor.call_count == expected_calls


if __name__ == "__main__":
    pytest.main([__file__])