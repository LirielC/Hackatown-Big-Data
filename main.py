#!/usr/bin/env python3
"""
Hackathon Forecast Model 2025 - Pipeline Principal
Ponto de entrada para execução do pipeline completo de previsão de vendas.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any
import yaml
from datetime import datetime

from src.utils.experiment_tracker import ExperimentTracker
from src.utils.mlflow_integration import MLflowModelTracker, setup_mlflow_autolog

def setup_logging(verbose: bool = False, log_file: str = None) -> logging.Logger:
    """Configura sistema de logging detalhado."""
    log_level = logging.DEBUG if verbose else logging.INFO

    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    if log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'pipeline_{timestamp}.log'

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logging.getLogger('mlflow').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)

    logger.info(f"Sistema de logging configurado - Arquivo: {log_file}")
    logger.info(f"Nível de logging: {logging.getLevelName(log_level)}")

    return logger

logger = setup_logging()


def load_config(config_path: str = "configs/model_config.yaml") -> Dict[str, Any]:
    """Carrega configurações do arquivo YAML."""
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        logger.info(f"Configuração carregada de {config_path}")
        return config
    except Exception as e:
        logger.error(f"Erro ao carregar configuração: {e}")
        raise


def setup_environment(config: Dict[str, Any]) -> ExperimentTracker:
    """Configura ambiente de execução e experiment tracking."""
    import os
    import numpy as np
    import random

    seed = config['general']['random_seed']

    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        logger.debug("TensorFlow seed configurado")
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        logger.debug("PyTorch seed configurado")
    except ImportError:
        pass

    n_jobs = config['general']['n_jobs']
    if n_jobs != -1:
        os.environ['OMP_NUM_THREADS'] = str(n_jobs)
        os.environ['MKL_NUM_THREADS'] = str(n_jobs)
        os.environ['NUMEXPR_NUM_THREADS'] = str(n_jobs)

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    logger.info(f"Ambiente configurado para reprodutibilidade completa:")
    logger.info(f"  - Random seed: {seed}")
    logger.info(f"  - Threads: {n_jobs}")
    logger.info(f"  - PYTHONHASHSEED: {os.environ.get('PYTHONHASHSEED')}")

    try:
        setup_mlflow_autolog()
        experiment_name = config.get('experiment_tracking', {}).get('experiment_name', 'hackathon-forecast-2025')
        experiment_tracker = ExperimentTracker(experiment_name)
        logger.info(f"Sistema de experiment tracking inicializado: {experiment_name}")
        return experiment_tracker
    except Exception as e:
        logger.warning(f"Erro ao inicializar experiment tracking: {e}")
        return None


def run_data_ingestion(config: Dict[str, Any]) -> None:
    """Executa módulo de ingestão de dados."""
    logger.info("=== INICIANDO INGESTÃO DE DADOS ===")

    from src.data.ingestion import DataIngestion

    use_polars = config.get('performance', {}).get('use_polars', True)
    ingestion = DataIngestion(use_polars=use_polars)

    raw_data_path = config['data']['raw_data_path']
    logger.info(f"Carregando dados de: {raw_data_path}")

    try:
        schemas = ingestion.load_multiple_schemas(raw_data_path, sample_only=False)

        logger.info(f"Encontrados {len(schemas)} schemas diferentes:")
        for i, (schema_sig, df) in enumerate(schemas.items()):
            logger.info(f"Schema {i+1}: {list(schema_sig)} - {df.shape[0]} registros")

            summary = ingestion.get_data_summary(df)
            logger.info(f"  Uso de memória: {summary['memory_usage_mb']:.2f} MB")

            if 'data' in df.columns or 'transaction_date' in df.columns:
                validation = ingestion.validate_data_quality(df, "transactions")
                data_type = "TRANSAÇÕES"
            elif 'produto' in df.columns:
                validation = ingestion.validate_data_quality(df, "products")
                data_type = "PRODUTOS"
            elif 'pdv' in df.columns:
                validation = ingestion.validate_data_quality(df, "stores")
                data_type = "PDVs"
            else:
                validation = ingestion.validate_data_quality(df, "generic")
                data_type = "GENÉRICO"

            status = "VÁLIDO" if validation['is_valid'] else "INVÁLIDO"
            logger.info(f"  Tipo: {data_type} - Status: {status}")

            if validation['errors']:
                logger.warning(f"  Erros: {validation['errors']}")
            if validation['warnings']:
                logger.info(f"  Avisos: {len(validation['warnings'])} encontrados")

        logger.info("Ingestão de dados concluída com sucesso")

    except Exception as e:
        logger.error(f"Erro durante ingestão de dados: {e}")
        raise


def run_data_preprocessing(config: Dict[str, Any]) -> None:
    """Executa módulo de pré-processamento."""
    logger.info("=== INICIANDO PRÉ-PROCESSAMENTO ===")

    from src.data.ingestion import DataIngestion
    from src.data.preprocessing import DataPreprocessor

    use_polars = config.get('performance', {}).get('use_polars', True)
    ingestion = DataIngestion(use_polars=use_polars)
    preprocessor = DataPreprocessor(use_polars=use_polars)

    raw_data_path = config['data']['raw_data_path']
    logger.info(f"Carregando dados de: {raw_data_path}")

    try:
        schemas = ingestion.load_multiple_schemas(raw_data_path, sample_only=False)

        transactions_df = None
        stores_df = None
        products_df = None

        for schema_sig, df in schemas.items():
            columns = list(schema_sig)

            if any(col in columns for col in ['transaction_date', 'quantity', 'internal_store_id']):
                logger.info(f"Identificado como dados de TRANSAÇÃO: {df.shape}")
                transactions_df = df

            elif any(col in columns for col in ['pdv', 'premise', 'categoria_pdv']):
                logger.info(f"Identificado como dados de PDV: {df.shape}")
                stores_df = df

            elif any(col in columns for col in ['produto', 'categoria', 'marca']):
                logger.info(f"Identificado como dados de PRODUTO: {df.shape}")
                products_df = df

        if transactions_df is None:
            raise ValueError("Dados de transação não encontrados")

        logger.info("Limpando dados de transação...")
        clean_transactions = preprocessor.clean_transactions(transactions_df)

        logger.info("Agregando para vendas semanais...")
        weekly_sales = preprocessor.aggregate_weekly_sales(clean_transactions)

        logger.info("Fazendo merge com dados mestres...")
        merged_data = preprocessor.merge_master_data(
            weekly_sales,
            products=products_df,
            stores=stores_df
        )

        logger.info("Criando features temporais...")
        final_data = preprocessor.create_time_features(merged_data)

        logger.info("Validando dados processados...")
        validation = preprocessor.validate_processed_data(final_data)

        if not validation['is_valid']:
            logger.error(f"Validação falhou: {validation['errors']}")
            raise ValueError("Dados processados são inválidos")

        summary = preprocessor.get_preprocessing_summary(transactions_df, final_data)
        logger.info(f"Resumo do pré-processamento:")
        logger.info(f"  - Registros originais: {summary['original_records']:,}")
        logger.info(f"  - Registros finais: {summary['processed_records']:,}")
        logger.info(f"  - Registros removidos: {summary['records_removed']:,} ({summary['removal_percentage']:.1f}%)")
        logger.info(f"  - Colunas adicionadas: {summary['columns_added']}")
        logger.info(f"  - Uso de memória: {summary['processed_memory_mb']:.1f} MB")

        processed_data_path = Path(config['data']['processed_data_path'])
        processed_data_path.mkdir(parents=True, exist_ok=True)

        output_file = processed_data_path / "weekly_sales_processed.parquet"
        final_data.to_parquet(output_file, index=False)
        logger.info(f"Dados processados salvos em: {output_file}")

        quality_summary = {
            'preprocessing_summary': summary,
            'validation_results': validation,
            'data_shape': final_data.shape,
            'columns': list(final_data.columns),
            'processing_timestamp': datetime.now().isoformat()
        }

        import json
        summary_file = processed_data_path / "preprocessing_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(quality_summary, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Resumo de qualidade salvo em: {summary_file}")
        logger.info("Pré-processamento concluído com sucesso")

    except Exception as e:
        logger.error(f"Erro durante pré-processamento: {e}")
        raise


def run_feature_engineering(config: Dict[str, Any]) -> None:
    """Executa engenharia de features."""
    logger.info("=== INICIANDO FEATURE ENGINEERING ===")

    from src.features.engineering import FeatureEngineer
    import pandas as pd
    from pathlib import Path

    try:
        import numpy as np

        processed_data_path = Path(config['data']['processed_data_path']) / "weekly_sales_processed.parquet"
        logger.info(f"Carregando dados processados: {processed_data_path}")

        if not processed_data_path.exists():
            raise FileNotFoundError(f"Arquivo de dados processados não encontrado: {processed_data_path}")

        df = pd.read_parquet(processed_data_path)
        logger.info(f"Dados carregados: {len(df)} registros, {len(df.columns)} colunas")

        feature_engineer = FeatureEngineer()

        logger.info("Criando features temporais...")
        df_features = df.copy()

        df_features['semana_ano'] = df_features['data_semana'].dt.isocalendar().week
        df_features['mes'] = df_features['data_semana'].dt.month
        df_features['trimestre'] = df_features['data_semana'].dt.quarter
        df_features['ano'] = df_features['data_semana'].dt.year

        df_features['semana_sin'] = np.sin(2 * np.pi * df_features['semana_ano'] / 52)
        df_features['semana_cos'] = np.cos(2 * np.pi * df_features['semana_ano'] / 52)
        df_features['mes_sin'] = np.sin(2 * np.pi * df_features['mes'] / 12)
        df_features['mes_cos'] = np.cos(2 * np.pi * df_features['mes'] / 12)

        df_features['is_feriado'] = 0
        df_features['is_pre_feriado'] = 0
        df_features['is_pos_feriado'] = 0

        logger.info("Features temporais básicas criadas (otimizado)")

        logger.info("Criando features de produto...")
        df_features = feature_engineer.create_product_features(df_features)

        logger.info("Criando features de PDV...")
        df_features = feature_engineer.create_store_features(df_features)

        logger.info("Criando features de lag e rolling (simplificado)...")

        df_features = df_features.sort_values(['pdv', 'produto', 'data_semana'])

        df_features['quantidade_lag_1'] = df_features.groupby(['pdv', 'produto'])['quantidade'].shift(1)
        df_features['quantidade_lag_2'] = df_features.groupby(['pdv', 'produto'])['quantidade'].shift(2)
        df_features['quantidade_lag_4'] = df_features.groupby(['pdv', 'produto'])['quantidade'].shift(4)

        df_features['quantidade_rolling_mean_4'] = df_features.groupby(['pdv', 'produto'])['quantidade'].rolling(window=4, min_periods=1).mean().reset_index(level=['pdv', 'produto'], drop=True)
        df_features['quantidade_rolling_std_4'] = df_features.groupby(['pdv', 'produto'])['quantidade'].rolling(window=4, min_periods=1).std().reset_index(level=['pdv', 'produto'], drop=True).fillna(0)

        logger.info("Features de lag e rolling criadas (simplificado)")

        feature_summary = feature_engineer.get_feature_summary(df_features)
        logger.info(f"Features criadas: {feature_summary['total_features']} colunas totais")
        logger.info(f"  - Temporais: {feature_summary['feature_types']['temporal']}")
        logger.info(f"  - Produto: {feature_summary['feature_types']['product']}")
        logger.info(f"  - PDV: {feature_summary['feature_types']['store']}")
        logger.info(f"  - Lag: {feature_summary['feature_types']['lag']}")
        logger.info(f"  - Rolling: {feature_summary['feature_types']['rolling']}")
        logger.info(f"  - Growth: {feature_summary['feature_types']['growth']}")

        features_output_path = Path(config['data']['processed_data_path']) / "features_engineered.parquet"
        df_features.to_parquet(features_output_path, index=False)
        logger.info(f"Features salvas em: {features_output_path}")

        summary_output_path = Path(config['data']['processed_data_path']) / "feature_engineering_summary.json"
        import json
        with open(summary_output_path, 'w', encoding='utf-8') as f:
            json.dump(feature_summary, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Resumo das features salvo em: {summary_output_path}")

        logger.info("Feature engineering concluída com sucesso")

    except Exception as e:
        logger.error(f"Erro durante feature engineering: {e}")
        raise


def run_model_training(config: Dict[str, Any]) -> None:
    """Executa treinamento dos modelos."""
    logger.info("=== INICIANDO TREINAMENTO DE MODELOS ===")

    from src.models.training import XGBoostModel, LightGBMModel, ModelEvaluator
    import pandas as pd
    from pathlib import Path
    from sklearn.model_selection import train_test_split

    try:
        features_path = Path(config['data']['processed_data_path']) / "features_engineered.parquet"
        logger.info(f"Carregando features: {features_path}")

        if not features_path.exists():
            raise FileNotFoundError(f"Arquivo de features não encontrado: {features_path}")

        df = pd.read_parquet(features_path)
        logger.info(f"Features carregadas: {len(df)} registros, {len(df.columns)} colunas")

        target_column = 'quantidade'
        date_column = 'data_semana'

        exclude_columns = [target_column, 'pdv', 'produto', date_column, 'data']
        feature_columns = [col for col in df.columns if col not in exclude_columns]

        numeric_columns = []
        for col in feature_columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_columns.append(col)
            else:
                logger.warning(f"Excluindo coluna não numérica: {col}")

        logger.info(f"Usando {len(numeric_columns)} features numéricas para treinamento")

        X = df[numeric_columns].copy()
        y = df[target_column].copy()

        X = X.fillna(0)
        logger.info(f"Dados preparados: X shape {X.shape}, y shape {y.shape}")

        split_idx = int(len(X) * 0.8)
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]

        logger.info(f"Treino: {len(X_train)} registros, Teste: {len(X_test)} registros")

        models_config = {
            'xgboost': config.get('models', {}).get('xgboost', {}),
            'lightgbm': config.get('models', {}).get('lightgbm', {})
        }

        logger.info("Treinando modelo XGBoost...")
        xgb_model = XGBoostModel(config)
        xgb_model.fit(
            X_train, y_train,
            X_val=X_test, y_val=y_test,
            optimize_hyperparams=False
        )

        y_pred_xgb = xgb_model.predict(X_test)

        xgb_metrics = ModelEvaluator.evaluate_model(y_test.values, y_pred_xgb)
        logger.info(f"XGBoost - WMAPE: {xgb_metrics['wmape']:.4f}%, MAE: {xgb_metrics['mae']:.2f}")

        if config.get('models', {}).get('lightgbm', {}).get('enabled', False):
            logger.info("Treinando modelo LightGBM...")
            lgb_model = LightGBMModel(config)
            lgb_model.fit(
                X_train, y_train,
                X_val=X_test, y_val=y_test,
                optimize_hyperparams=False
            )

            y_pred_lgb = lgb_model.predict(X_test)
            lgb_metrics = ModelEvaluator.evaluate_model(y_test.values, y_pred_lgb)
            logger.info(f"LightGBM - WMAPE: {lgb_metrics['wmape']:.4f}%, MAE: {lgb_metrics['mae']:.2f}")

        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)

        model_path = models_dir / "hackathon_forecast_model.pkl"
        xgb_model.save_model(str(model_path))
        logger.info(f"Modelo salvo em: {model_path}")

        training_info = {
            'model_type': 'XGBoost',
            'training_date': pd.Timestamp.now().isoformat(),
            'data_shape': X.shape,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features_used': len(numeric_columns),
            'feature_names': numeric_columns[:10],
            'metrics': xgb_metrics,
            'model_path': str(model_path)
        }

        import json
        training_info_path = models_dir / "training_info.json"
        with open(training_info_path, 'w', encoding='utf-8') as f:
            json.dump(training_info, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Informações do treinamento salvas em: {training_info_path}")

        if hasattr(xgb_model, 'feature_importance_') and xgb_model.feature_importance_ is not None:
            top_features = xgb_model.feature_importance_.head(10)
            logger.info("Top 10 features mais importantes:")
            for _, row in top_features.iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        logger.info("Treinamento de modelos concluído com sucesso")

    except Exception as e:
        logger.error(f"Erro durante treinamento de modelos: {e}")
        raise


def run_stacking_ensemble(config: Dict[str, Any]) -> None:
    """Executa stacking ensemble como etapa separada do pipeline."""
    logger.info("=== INICIANDO STACKING ENSEMBLE ===")

    try:
        from src.models.ensemble_advanced import AdvancedEnsemble
        from pathlib import Path
        import pandas as pd
        import numpy as np
        import json
        from datetime import datetime

        features_path = Path(config['data']['processed_data_path']) / "features_engineered.parquet"
        logger.info(f"Carregando dados para stacking: {features_path}")

        if not features_path.exists():
            raise FileNotFoundError(f"Arquivo de features não encontrado: {features_path}")

        df = pd.read_parquet(features_path)
        logger.info(f"Dados carregados: {len(df)} registros, {len(df.columns)} colunas")

        target_column = 'quantidade'
        date_column = 'data_semana'

        exclude_columns = [target_column, 'pdv', 'produto', date_column, 'data']
        feature_columns = [col for col in df.columns if col not in exclude_columns and pd.api.types.is_numeric_dtype(df[col])]

        logger.info(f"Usando {len(feature_columns)} features numéricas para stacking")

        X = df[feature_columns].copy()
        y = df[target_column].copy()
        X = X.fillna(0)

        split_idx = int(len(X) * 0.8)
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]

        logger.info(f"Treino: {len(X_train)} registros, Teste: {len(X_test)} registros")

        logger.info("🏗️ Criando ensemble avançado...")
        advanced_ensemble = AdvancedEnsemble(random_state=config['general']['random_seed'])

        logger.info("🏋️ Treinando modelos base...")
        advanced_ensemble.create_base_models()

        models_trained = {}
        for model_name, model in advanced_ensemble.models.items():
            logger.info(f"Treinando {model_name}...")
            model_clone = type(model)(**model.get_params())
            model_clone.fit(X_train, y_train)
            models_trained[model_name] = model_clone

        logger.info("🔄 Gerando meta-features com validação cruzada temporal...")

        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=5)
        meta_features = np.full((len(X_train), len(models_trained)), np.nan)

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            logger.info(f"Fold {fold + 1}/5")

            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            for i, (model_name, model) in enumerate(models_trained.items()):
                model_clone = type(model)(**model.get_params())
                model_clone.fit(X_fold_train, y_fold_train)
                val_predictions = model_clone.predict(X_fold_val)
                meta_features[val_idx, i] = val_predictions

        valid_mask = ~np.isnan(meta_features).any(axis=1)
        meta_features_clean = meta_features[valid_mask]
        y_train_clean = y_train.iloc[valid_mask]

        logger.info(f"Meta-features criadas: {meta_features_clean.shape}")

        logger.info("🎯 Treinando meta-learner (Ridge Regression)...")

        from sklearn.linear_model import Ridge
        meta_learner = Ridge(alpha=1.0, random_state=config['general']['random_seed'])
        meta_learner.fit(meta_features_clean, y_train_clean)

        logger.info("🔮 Gerando previsões de teste...")

        test_meta_features = np.column_stack([
            model.predict(X_test) for model in models_trained.values()
        ])

        stacking_predictions = meta_learner.predict(test_meta_features)

        def calculate_wmape(y_true, y_pred):
            return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))

        stacking_wmape = calculate_wmape(y_test.values, stacking_predictions)
        stacking_mae = np.mean(np.abs(y_test.values - stacking_predictions))
        stacking_rmse = np.sqrt(np.mean((y_test.values - stacking_predictions) ** 2))

        logger.info("📊 STACKING ENSEMBLE - RESULTADOS:")
        logger.info(f"  WMAPE: {stacking_wmape:.4f}")
        logger.info(f"  MAE: {stacking_mae:.4f}")
        logger.info(f"  RMSE: {stacking_rmse:.4f}")

        logger.info("⚖️ Comparando com ensemble simples...")

        simple_predictions = np.mean(test_meta_features, axis=1)
        simple_wmape = calculate_wmape(y_test.values, simple_predictions)

        improvement = simple_wmape - stacking_wmape
        logger.info(f"📈 Melhoria vs Ensemble Simples: {improvement:.1%}")

        target_achieved = improvement > 0.10
        if target_achieved:
            logger.info("🎯 ✅ META ATINGIDA: Melhoria > 10% vs ensemble simples!")
        else:
            logger.info("🎯 ❌ Meta não atingida: Melhoria insuficiente")
            logger.info(f"   Atual: {improvement:.1%}, Necessário: 10.0%")

        logger.info("💾 Salvando resultados do stacking ensemble...")

        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)

        stacking_results = {
            'task': '1.3 - Stacking Ensemble',
            'timestamp': datetime.now().isoformat(),
            'meta_features_shape': meta_features_clean.shape,
            'models_used': list(models_trained.keys()),
            'meta_learner': 'Ridge Regression (alpha=1.0)',
            'metrics': {
                'stacking_wmape': stacking_wmape,
                'stacking_mae': stacking_mae,
                'stacking_rmse': stacking_rmse,
                'simple_ensemble_wmape': simple_wmape,
                'improvement_vs_simple': improvement,
                'target_achieved': target_achieved
            },
            'cross_validation': {
                'method': 'TimeSeriesSplit',
                'n_splits': 5,
                'valid_samples': len(meta_features_clean)
            }
        }

        stacking_model_data = {
            'meta_learner': meta_learner,
            'models_trained': models_trained,
            'feature_columns': feature_columns,
            'results': stacking_results
        }

        import joblib
        stacking_path = models_dir / "stacking_ensemble_tarefa_1_3.joblib"
        joblib.dump(stacking_model_data, stacking_path)

        report_path = models_dir / "stacking_ensemble_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(stacking_results, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"✅ Modelo stacking salvo em: {stacking_path}")
        logger.info(f"✅ Relatório salvo em: {report_path}")

        logger.info("🎉 STACKING ENSEMBLE CONCLUÍDO COM SUCESSO!")

    except Exception as e:
        logger.error(f"❌ Erro durante stacking ensemble: {e}")
        raise


def run_model_validation(config: Dict[str, Any]) -> None:
    """Executa validação dos modelos."""
    logger.info("=== INICIANDO VALIDAÇÃO DE MODELOS ===")

    from src.models.validation import ValidationManager
    from src.models.training import XGBoostModel
    import pandas as pd
    from pathlib import Path

    try:
        features_path = Path(config['data']['processed_data_path']) / "features_engineered.parquet"
        logger.info(f"Carregando features para validação: {features_path}")

        if not features_path.exists():
            raise FileNotFoundError(f"Arquivo de features não encontrado: {features_path}")

        df = pd.read_parquet(features_path)
        logger.info(f"Features carregadas: {len(df)} registros, {len(df.columns)} colunas")

        target_column = 'quantidade'
        date_column = 'data_semana'

        exclude_columns = [target_column, 'pdv', 'produto', date_column, 'data']
        feature_columns = [col for col in df.columns if col not in exclude_columns and pd.api.types.is_numeric_dtype(df[col])]

        logger.info(f"Usando {len(feature_columns)} features numéricas para validação")

        X = df[feature_columns].copy()
        y = df[target_column].copy()
        X = X.fillna(0)

        X_with_date = X.copy()
        X_with_date[date_column] = df[date_column]

        logger.info(f"Dados preparados para validação: X shape {X.shape}, y shape {y.shape}")

        model_path = Path("models/hackathon_forecast_model.pkl")
        if model_path.exists():
            logger.info("Carregando modelo treinado para validação...")
            model = XGBoostModel(config)
            model.load_model(str(model_path))
        else:
            logger.warning("Modelo treinado não encontrado. Treinando modelo básico para validação...")
            split_idx = int(len(X) * 0.8)
            X_train = X.iloc[:split_idx]
            y_train = y.iloc[:split_idx]

            model = XGBoostModel(config)
            model.fit(X_train, y_train, optimize_hyperparams=False)

        validation_manager = ValidationManager(config)

        logger.info("Executando validação walk-forward...")
        validation_results = validation_manager.run_complete_validation(
            model=model,
            X=X_with_date,
            y=y,
            date_column=date_column,
            segment_column='pdv',
            save_path=None
        )

        overall_metrics = validation_results['validation']['overall_metrics']
        logger.info("=== RESULTADOS DA VALIDAÇÃO ===")
        logger.info(f"WMAPE: {overall_metrics['wmape']:.4f}%")
        logger.info(f"MAE: {overall_metrics['mae']:.2f}")
        logger.info(f"RMSE: {overall_metrics['rmse']:.2f}")
        logger.info(f"MAPE: {overall_metrics['mape']:.2f}%")

        residual_stats = validation_results['residual_analysis']['basic_stats']
        logger.info("=== ANÁLISE DE RESÍDUOS ===")
        logger.info(f"Média dos resíduos: {residual_stats['mean']:.4f}")
        logger.info(f"Desvio padrão dos resíduos: {residual_stats['std']:.4f}")

        baseline_comparison = validation_results['baseline_comparison']
        logger.info("=== COMPARAÇÃO COM BASELINES ===")
        logger.info(f"Baseline usado: {baseline_comparison['best_baseline']}")
        logger.info(f"Melhoria no WMAPE: {baseline_comparison['improvements']['historical_mean']['wmape_improvement']:.2f}%")

        exec_summary = validation_results['executive_summary']
        logger.info("=== RESUMO EXECUTIVO ===")
        logger.info(f"Performance do modelo: {exec_summary['model_performance']['wmape']}")
        logger.info(f"Estabilidade dos resíduos: {exec_summary['validation_quality']['residual_bias']}")
        logger.info(f"Rank do modelo: {exec_summary['baseline_comparison']['model_rank']}")

        validation_dir = Path("validation_results")
        validation_dir.mkdir(exist_ok=True)

        import json
        validation_results_path = validation_dir / "validation_results.json"
        with open(validation_results_path, 'w', encoding='utf-8') as f:
            json.dump(validation_results, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Resultados da validação salvos em: {validation_results_path}")

        wmape_threshold = config.get('validation', {}).get('wmape_threshold', 25.0)
        if overall_metrics['wmape'] <= wmape_threshold:
            logger.info(f"✅ Modelo aprovado! WMAPE ({overall_metrics['wmape']:.2f}%) está dentro do limite de {wmape_threshold}%")
        else:
            logger.warning(f"⚠️ Modelo precisa de melhorias. WMAPE ({overall_metrics['wmape']:.2f}%) está acima do limite de {wmape_threshold}%")

        logger.info("Validação de modelos concluída com sucesso")

    except Exception as e:
        logger.error(f"Erro durante validação de modelos: {e}")
        raise


def run_prediction_generation(config: Dict[str, Any]) -> None:
    """Gera previsões finais."""
    logger.info("=== INICIANDO GERAÇÃO DE PREVISÕES ===")

    from src.models.prediction import PredictionGenerator
    from src.models.training import XGBoostModel
    import pandas as pd
    from pathlib import Path

    try:
        model_path = Path("models/hackathon_forecast_model.pkl")
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo treinado não encontrado: {model_path}")

        logger.info("Carregando modelo treinado...")
        model = XGBoostModel(config)
        model.load_model(str(model_path))

        processed_data_path = Path(config['data']['processed_data_path']) / "weekly_sales_processed.parquet"
        if processed_data_path.exists():
            historical_data = pd.read_parquet(processed_data_path)
            logger.info(f"Dados históricos carregados: {len(historical_data)} registros")
        else:
            logger.warning("Dados históricos não encontrados")
            historical_data = None

        logger.info("Criando dados de predição para janeiro/2023...")

        features_path = Path(config['data']['processed_data_path']) / "features_engineered.parquet"
        if not features_path.exists():
            raise FileNotFoundError(f"Arquivo de features processadas não encontrado: {features_path}")

        processed_data = pd.read_parquet(features_path)
        logger.info(f"Dados processados carregados: {len(processed_data)} registros, {len(processed_data.columns)} colunas")

        dec_2022_data = processed_data[processed_data['data_semana'] >= '2022-12-01'].copy()
        logger.info(f"Dados de dezembro/2022: {len(dec_2022_data)} registros")

        prediction_df = dec_2022_data.copy()

        prediction_df['ano'] = 2023
        prediction_df['mes'] = 1
        prediction_df['trimestre'] = 1
        prediction_df['data_semana'] = prediction_df['data_semana'] + pd.DateOffset(months=1)

        prediction_df['semana_ano'] = prediction_df['data_semana'].dt.isocalendar().week
        prediction_df['semana'] = prediction_df['semana_ano']

        model_features = model.model.feature_names_in_.tolist()
        available_features = [col for col in prediction_df.columns if col in model_features]

        logger.info(f"Modelo espera {len(model_features)} features")
        logger.info(f"Features disponíveis: {len(available_features)}")

        if len(available_features) != len(model_features):
            missing_features = set(model_features) - set(available_features)
            logger.warning(f"Features faltando: {missing_features}")

            for feature in missing_features:
                prediction_df[feature] = 0
                available_features.append(feature)

        prediction_df = prediction_df[available_features + ['pdv', 'produto', 'data_semana']].copy()

        logger.info(f"DataFrame final de predição: {len(prediction_df)} registros, {len(prediction_df.columns)} colunas")
        logger.info(f"Features que serão enviadas ao modelo: {sorted(available_features)}")

        prediction_generator = PredictionGenerator(config)

        logger.info("Gerando previsões...")
        predictions_df = prediction_generator.generate_predictions(
            model=model,
            features_df=prediction_df,
            historical_data=historical_data
        )

        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)

        predictions_path = output_dir / "predictions_generated.parquet"
        predictions_df.to_parquet(predictions_path, index=False)
        logger.info(f"Previsões salvas em: {predictions_path}")

        summary = prediction_generator.generate_prediction_summary(predictions_df)
        logger.info("=== RESUMO DAS PREVISÕES ===")
        logger.info(f"Total de previsões: {summary['total_predictions']}")
        logger.info(f"Semanas cobertas: {summary['prediction_period']['weeks']}")
        logger.info(f"PDVs únicos: {summary['coverage']['unique_pdvs']}")
        logger.info(f"Produtos únicos: {summary['coverage']['unique_products']}")
        logger.info(f"Quantidade total prevista: {summary['quantity_statistics']['total_quantity']}")
        logger.info(f"Quantidade média prevista: {summary['quantity_statistics']['average_quantity']:.2f}")

        import json
        summary_path = output_dir / "prediction_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Resumo das previsões salvo em: {summary_path}")
        logger.info("Geração de previsões concluída com sucesso")

    except Exception as e:
        logger.error(f"Erro durante geração de previsões: {e}")
        raise


def run_output_formatting(config: Dict[str, Any]) -> None:
    """Formata saída para submissão."""
    logger.info("=== INICIANDO FORMATAÇÃO DE SAÍDA ===")

    from src.models.output_formatter import SubmissionFormatter, SubmissionValidator
    import pandas as pd
    from pathlib import Path
    from datetime import datetime

    try:
        # Carregar previsões geradas
        predictions_path = Path("output/predictions_generated.parquet")
        if not predictions_path.exists():
            raise FileNotFoundError(f"Arquivo de previsões não encontrado: {predictions_path}")

        logger.info(f"Carregando previsões: {predictions_path}")
        predictions_df = pd.read_parquet(predictions_path)
        logger.info(f"Previsões carregadas: {len(predictions_df)} registros")

        # Inicializar formatador de submissão
        formatter = SubmissionFormatter(config)

        # Criar diretório de saída
        output_dir = Path("submissions")
        output_dir.mkdir(exist_ok=True)

        # Gerar timestamp para nome dos arquivos
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Formatar e salvar arquivo CSV para submissão
        logger.info("Formatando arquivo CSV para submissão...")
        csv_filename = f"hackathon_forecast_submission_{timestamp}.csv"
        csv_path = output_dir / csv_filename

        csv_saved_path = formatter.format_submission_csv(
            predictions_df=predictions_df,
            output_path=str(csv_path)
        )

        logger.info(f"Arquivo CSV salvo: {csv_saved_path}")

        # Formatar e salvar arquivo Parquet
        logger.info("Formatando arquivo Parquet para submissão...")
        parquet_filename = f"hackathon_forecast_submission_{timestamp}.parquet"
        parquet_path = output_dir / parquet_filename

        parquet_saved_path = formatter.format_submission_parquet(
            predictions_df=predictions_df,
            output_path=str(parquet_path)
        )

        logger.info(f"Arquivo Parquet salvo: {parquet_saved_path}")

        # Validar arquivos de submissão
        logger.info("Validando arquivos de submissão...")

        # Validar CSV
        csv_validation = SubmissionValidator.validate_csv_format(str(csv_saved_path))
        if csv_validation['is_valid']:
            logger.info("✅ Arquivo CSV validado com sucesso")
            logger.info(f"   - Total de registros: {csv_validation['data_info']['total_rows']}")
            logger.info(f"   - PDVs únicos: {csv_validation['data_info']['unique_pdvs']}")
            logger.info(f"   - Produtos únicos: {csv_validation['data_info']['unique_products']}")
        else:
            logger.error("❌ Arquivo CSV inválido:")
            for error in csv_validation['errors']:
                logger.error(f"   - {error}")

        # Validar Parquet
        parquet_validation = SubmissionValidator.validate_parquet_format(str(parquet_saved_path))
        if parquet_validation['is_valid']:
            logger.info("✅ Arquivo Parquet validado com sucesso")
            logger.info(f"   - Total de registros: {parquet_validation['data_info']['total_rows']}")
            logger.info(f"   - PDVs únicos: {parquet_validation['data_info']['unique_pdvs']}")
            logger.info(f"   - Produtos únicos: {parquet_validation['data_info']['unique_products']}")
        else:
            logger.error("❌ Arquivo Parquet inválido:")
            for error in parquet_validation['errors']:
                logger.error(f"   - {error}")

        # Comparar formatos
        logger.info("Comparando consistência entre formatos...")
        format_comparison = SubmissionValidator.compare_formats(
            csv_path=str(csv_saved_path),
            parquet_path=str(parquet_saved_path)
        )

        if format_comparison['files_consistent']:
            logger.info("✅ Arquivos CSV e Parquet são consistentes")
        else:
            logger.warning("⚠️ Diferenças encontradas entre CSV e Parquet:")
            for diff in format_comparison['differences']:
                logger.warning(f"   - {diff}")

        # Gerar resumo da submissão
        logger.info("Gerando resumo da submissão...")
        submission_summary = formatter.generate_submission_summary(predictions_df)

        logger.info("=== RESUMO DA SUBMISSÃO ===")
        logger.info(f"Total de previsões: {submission_summary['submission_info']['total_predictions']}")
        logger.info(f"Semanas cobertas: {submission_summary['data_coverage']['weeks']}")
        logger.info(f"PDVs únicos: {submission_summary['data_coverage']['unique_pdvs']}")
        logger.info(f"Produtos únicos: {submission_summary['data_coverage']['unique_products']}")
        logger.info(f"Quantidade total prevista: {submission_summary['quantity_summary']['total_quantity']:,}")
        logger.info(f"Quantidade média: {submission_summary['quantity_summary']['average_quantity']:.2f}")
        logger.info(f"Previsões zero: {submission_summary['quantity_summary']['zero_predictions']} ({submission_summary['quantity_summary']['zero_percentage']:.1f}%)")

        # Salvar resumo
        import json
        summary_filename = f"submission_summary_{timestamp}.json"
        summary_path = output_dir / summary_filename

        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(submission_summary, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Resumo da submissão salvo: {summary_path}")

        # Log caminhos dos arquivos finais
        logger.info("=== ARQUIVOS DE SUBMISSÃO GERADOS ===")
        logger.info(f"CSV: {csv_saved_path}")
        logger.info(f"Parquet: {parquet_saved_path}")
        logger.info(f"Resumo: {summary_path}")

        # Verificar tamanho dos arquivos
        csv_size = Path(csv_saved_path).stat().st_size / (1024 * 1024)  # MB
        parquet_size = Path(parquet_saved_path).stat().st_size / (1024 * 1024)  # MB

        logger.info("=== TAMANHO DOS ARQUIVOS ===")
        logger.info(f"CSV: {csv_size:.2f} MB")
        logger.info(f"Parquet: {parquet_size:.2f} MB")
        logger.info(f"Compressão: {((csv_size - parquet_size) / csv_size * 100):.1f}%")

        # Verificação final de integridade
        if csv_validation['is_valid'] and parquet_validation['is_valid']:
            logger.info("✅ TODOS OS ARQUIVOS DE SUBMISSÃO SÃO VÁLIDOS!")
            logger.info("🎉 Submissão pronta para envio ao hackathon!")
        else:
            logger.warning("⚠️ Alguns arquivos podem ter problemas. Verifique os logs acima.")

        logger.info("Formatação de saída concluída com sucesso")

    except Exception as e:
        logger.error(f"Erro durante formatação de saída: {e}")
        raise


def run_multiple_submissions(config: Dict[str, Any]) -> None:
    """Executa sistema de múltiplas submissões."""
    logger.info("=== INICIANDO SISTEMA DE MÚLTIPLAS SUBMISSÕES ===")
    
    try:
        from src.utils.fast_submission_pipeline import FastSubmissionPipeline
        
        # Inicializar pipeline de submissões
        pipeline = FastSubmissionPipeline()
        
        # Gerar todas as submissões
        logger.info("Gerando submissões para todas as estratégias...")
        submissions = pipeline.generate_all_submissions()
        
        logger.info(f"✓ {len(submissions)} submissões geradas com sucesso!")
        
        # Mostrar resumo das submissões
        logger.info("=== RESUMO DAS SUBMISSÕES ===")
        for submission in submissions:
            logger.info(f"• {submission.strategy_name} {submission.version}")
            if submission.performance_metrics and 'wmape' in submission.performance_metrics:
                wmape = submission.performance_metrics['wmape']
                logger.info(f"  WMAPE: {wmape:.6f}")
        
        # Gerar relatório de comparação
        logger.info("Gerando relatório de comparação...")
        report_path = pipeline.submission_manager.generate_performance_report()
        logger.info(f"Relatório gerado: {report_path}")
        
        # Mostrar melhores submissões
        best_submissions = pipeline.submission_manager.get_best_submission_by_strategy()
        if best_submissions:
            logger.info("=== MELHORES SUBMISSÕES POR ESTRATÉGIA ===")
            for strategy, submission in best_submissions.items():
                if submission.performance_metrics and 'wmape' in submission.performance_metrics:
                    wmape = submission.performance_metrics['wmape']
                    logger.info(f"• {strategy}: WMAPE {wmape:.6f} ({submission.version})")
        
        # Mostrar estatísticas do cache
        cache_stats = pipeline.get_cache_stats()
        total_cache_size = cache_stats['features_cache']['size_mb'] + cache_stats['models_cache']['size_mb']
        logger.info(f"Cache utilizado: {total_cache_size:.2f} MB")
        
        logger.info("Sistema de múltiplas submissões concluído com sucesso")
        
    except Exception as e:
        logger.error(f"Erro no sistema de múltiplas submissões: {e}")
        raise


def run_experiment_analysis(config: Dict[str, Any]) -> None:
    """Analisa resultados dos experimentos."""
    logger.info("=== ANALISANDO EXPERIMENTOS ===")
    
    try:
        from src.utils.mlflow_integration import MLflowModelTracker, create_experiment_report
        
        # Initialize tracker
        tracker = MLflowModelTracker()
        
        # Get experiment runs
        runs_df = tracker.tracker.get_experiment_runs()
        
        if len(runs_df) == 0:
            logger.info("Nenhum experimento encontrado")
            return
        
        logger.info(f"Encontrados {len(runs_df)} experimentos")
        
        # Show leaderboard
        leaderboard = tracker.get_model_leaderboard(metric='wmape', top_k=10)
        if not leaderboard.empty:
            logger.info("\n=== TOP 10 MODELOS (WMAPE) ===")
            for i, (_, row) in enumerate(leaderboard.iterrows(), 1):
                run_id = row['run_id'][:8]
                wmape = row.get('metrics.wmape', 'N/A')
                model_type = row.get('params.model_type', 'Unknown')
                logger.info(f"{i:2d}. {run_id} - WMAPE: {wmape} - Tipo: {model_type}")
        
        # Export results
        output_path = "experiment_results.csv"
        tracker.export_experiment_results(output_path)
        logger.info(f"Resultados exportados para: {output_path}")
        
        # Create HTML report
        report_path = "experiment_report.html"
        create_experiment_report(tracker.tracker, report_path)
        logger.info(f"Relatório HTML criado: {report_path}")
        
        # Show experiment summary
        logger.info("\n=== RESUMO DOS EXPERIMENTOS ===")
        if 'metrics.wmape' in runs_df.columns:
            wmape_stats = runs_df['metrics.wmape'].describe()
            logger.info(f"WMAPE - Média: {wmape_stats['mean']:.4f}, Melhor: {wmape_stats['min']:.4f}")
        
        if 'metrics.mae' in runs_df.columns:
            mae_stats = runs_df['metrics.mae'].describe()
            logger.info(f"MAE - Média: {mae_stats['mean']:.4f}, Melhor: {mae_stats['min']:.4f}")
        
        logger.info("\nPara visualizar no MLflow UI:")
        logger.info("1. Execute: mlflow ui")
        logger.info("2. Abra: http://localhost:5000")
        
    except Exception as e:
        logger.error(f"Erro durante análise de experimentos: {e}")
        raise


class PipelineExecutionTracker:
    """Rastreia execução detalhada do pipeline."""
    
    def __init__(self):
        self.steps = []
        self.start_time = None
        self.end_time = None
        self.current_step = None
        
    def start_pipeline(self):
        """Inicia rastreamento do pipeline."""
        self.start_time = datetime.now()
        logger.info(f"Pipeline iniciado em: {self.start_time}")
        
    def start_step(self, step_name: str):
        """Inicia rastreamento de uma etapa."""
        self.current_step = {
            'name': step_name,
            'start_time': datetime.now(),
            'end_time': None,
            'duration': None,
            'success': False,
            'error': None
        }
        logger.info(f"Iniciando etapa: {step_name}")
        
    def end_step(self, success: bool = True, error: str = None):
        """Finaliza rastreamento de uma etapa."""
        if self.current_step:
            self.current_step['end_time'] = datetime.now()
            self.current_step['duration'] = (
                self.current_step['end_time'] - self.current_step['start_time']
            ).total_seconds()
            self.current_step['success'] = success
            self.current_step['error'] = error
            
            self.steps.append(self.current_step.copy())
            
            status = "SUCESSO" if success else "ERRO"
            logger.info(f"Etapa '{self.current_step['name']}' finalizada: {status} "
                       f"(Duração: {self.current_step['duration']:.2f}s)")
            
            if error:
                logger.error(f"Erro na etapa '{self.current_step['name']}': {error}")
                
            self.current_step = None
            
    def end_pipeline(self):
        """Finaliza rastreamento do pipeline."""
        self.end_time = datetime.now()
        total_duration = (self.end_time - self.start_time).total_seconds()
        
        logger.info(f"Pipeline finalizado em: {self.end_time}")
        logger.info(f"Duração total: {total_duration:.2f}s ({total_duration/60:.2f}min)")
        
        # Resumo das etapas
        logger.info("=== RESUMO DAS ETAPAS ===")
        successful_steps = 0
        for step in self.steps:
            status = "✓" if step['success'] else "✗"
            logger.info(f"{status} {step['name']}: {step['duration']:.2f}s")
            if step['success']:
                successful_steps += 1
                
        logger.info(f"Etapas concluídas com sucesso: {successful_steps}/{len(self.steps)}")
        
    def get_summary(self) -> Dict[str, Any]:
        """Retorna resumo da execução."""
        if not self.start_time:
            return {}
            
        total_duration = 0
        if self.end_time:
            total_duration = (self.end_time - self.start_time).total_seconds()
            
        successful_steps = sum(1 for step in self.steps if step['success'])
        
        return {
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'total_duration_seconds': total_duration,
            'total_steps': len(self.steps),
            'successful_steps': successful_steps,
            'success_rate': successful_steps / len(self.steps) if self.steps else 0,
            'steps': self.steps
        }


def run_full_pipeline(config: Dict[str, Any]) -> None:
    """Executa pipeline completo com experiment tracking."""
    # Inicializar rastreador de execução
    execution_tracker = PipelineExecutionTracker()
    execution_tracker.start_pipeline()
    
    # Configurar ambiente e experiment tracking
    experiment_tracker = setup_environment(config)
    
    # Iniciar run do MLflow se disponível
    run_id = None
    if experiment_tracker:
        run_name = f"full_pipeline_{execution_tracker.start_time.strftime('%Y%m%d_%H%M%S')}"
        run_id = experiment_tracker.start_run(run_name)
        
        # Log pipeline parameters
        pipeline_params = {
            'pipeline_type': 'full',
            'config_file': 'model_config.yaml',
            'random_seed': config['general']['random_seed'],
            'start_time': execution_tracker.start_time.isoformat(),
            'python_version': sys.version,
            'platform': sys.platform
        }
        experiment_tracker.log_params(pipeline_params)
    
    # Lista de etapas do pipeline
    pipeline_steps = [
        ("Ingestão de Dados", run_data_ingestion),
        ("Pré-processamento", run_data_preprocessing),
        ("Feature Engineering", run_feature_engineering),
        ("Treinamento de Modelos", run_model_training),
        ("Stacking Ensemble", run_stacking_ensemble),
        ("Validação de Modelos", run_model_validation),
        ("Geração de Previsões", run_prediction_generation),
        ("Formatação de Saída", run_output_formatting)
    ]
    
    pipeline_success = True
    
    try:
        for step_name, step_function in pipeline_steps:
            execution_tracker.start_step(step_name)
            
            try:
                step_function(config)
                execution_tracker.end_step(success=True)
                
            except Exception as step_error:
                execution_tracker.end_step(success=False, error=str(step_error))
                pipeline_success = False
                raise step_error
        
        execution_tracker.end_pipeline()
        
        # Log pipeline metrics
        if experiment_tracker:
            summary = execution_tracker.get_summary()
            pipeline_metrics = {
                'pipeline_duration_seconds': summary['total_duration_seconds'],
                'pipeline_duration_minutes': summary['total_duration_seconds'] / 60,
                'total_steps': summary['total_steps'],
                'successful_steps': summary['successful_steps'],
                'success_rate': summary['success_rate'],
                'success': 1.0 if pipeline_success else 0.0
            }
            experiment_tracker.log_metrics(pipeline_metrics)
            
            # Log step-level metrics
            for i, step in enumerate(summary['steps']):
                step_metrics = {
                    f'step_{i+1}_duration': step['duration'],
                    f'step_{i+1}_success': 1.0 if step['success'] else 0.0
                }
                experiment_tracker.log_metrics(step_metrics)
        
        logger.info("=== PIPELINE CONCLUÍDO COM SUCESSO ===")
        
    except Exception as e:
        execution_tracker.end_pipeline()
        pipeline_success = False
        
        # Log error metrics
        if experiment_tracker:
            summary = execution_tracker.get_summary()
            experiment_tracker.log_metrics({
                'success': 0.0,
                'pipeline_duration_seconds': summary.get('total_duration_seconds', 0),
                'successful_steps': summary.get('successful_steps', 0),
                'total_steps': summary.get('total_steps', 0)
            })
            experiment_tracker.log_params({'error': str(e)})
        
        logger.error(f"Pipeline falhou: {e}")
        raise
    finally:
        # Salvar resumo da execução
        try:
            summary = execution_tracker.get_summary()
            summary_file = f"pipeline_execution_summary_{execution_tracker.start_time.strftime('%Y%m%d_%H%M%S')}.json"
            
            import json
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Resumo da execução salvo em: {summary_file}")
        except Exception as summary_error:
            logger.warning(f"Erro ao salvar resumo da execução: {summary_error}")
        
        # End MLflow run
        if experiment_tracker:
            experiment_tracker.end_run()


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description="Pipeline de Previsão de Vendas - Hackathon 2025",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python main.py --step full                    # Executa pipeline completo
  python main.py --step ingestion --verbose    # Executa apenas ingestão com logs detalhados
  python main.py --config custom_config.yaml   # Usa configuração personalizada
  python main.py --step experiments            # Analisa experimentos existentes
        """
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_config.yaml",
        help="Caminho para arquivo de configuração (padrão: configs/model_config.yaml)"
    )
    parser.add_argument(
        "--step",
        type=str,
        choices=[
            "ingestion", "preprocessing", "features",
            "training", "stacking", "validation", "prediction", "output", "full", "experiments", "submissions"
        ],
        default="full",
        help="Etapa específica do pipeline para executar (padrão: full)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Habilitar logging detalhado (DEBUG level)"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Arquivo de log personalizado (padrão: pipeline_TIMESTAMP.log)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execução de teste (valida configuração sem executar pipeline)"
    )

    args = parser.parse_args()

    global logger
    logger = setup_logging(verbose=args.verbose, log_file=args.log_file)

    logger.info("=== HACKATHON FORECAST MODEL 2025 ===")
    logger.info(f"Argumentos: {vars(args)}")

    try:
        logger.info(f"Carregando configuração de: {args.config}")
        config = load_config(args.config)

        validate_config(config)

        if args.dry_run:
            logger.info("=== MODO DRY RUN - VALIDAÇÃO APENAS ===")
            logger.info("Configuração válida. Pipeline não será executado.")
            return

        step_functions = {
            "ingestion": run_data_ingestion,
            "preprocessing": run_data_preprocessing,
            "features": run_feature_engineering,
            "training": run_model_training,
            "stacking": run_stacking_ensemble,
            "validation": run_model_validation,
            "prediction": run_prediction_generation,
            "output": run_output_formatting,
            "full": run_full_pipeline,
            "experiments": run_experiment_analysis,
            "submissions": run_multiple_submissions
        }

        if args.step in step_functions:
            logger.info(f"Executando etapa: {args.step}")
            step_functions[args.step](config)
            logger.info(f"Etapa '{args.step}' concluída com sucesso")
        else:
            logger.error(f"Etapa inválida: {args.step}")
            logger.info(f"Etapas disponíveis: {list(step_functions.keys())}")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.warning("Execução interrompida pelo usuário")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Erro durante execução: {e}")
        logger.debug("Detalhes do erro:", exc_info=True)
        sys.exit(1)


def validate_config(config: Dict[str, Any]) -> None:
    """Valida configuração do pipeline."""
    logger.info("Validando configuração...")

    required_sections = ['general', 'data', 'models']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Seção obrigatória '{section}' não encontrada na configuração")

    general = config['general']
    if 'random_seed' not in general:
        raise ValueError("random_seed não configurado em 'general'")

    data = config['data']
    required_paths = ['raw_data_path', 'processed_data_path']
    for path_key in required_paths:
        if path_key not in data:
            raise ValueError(f"Caminho '{path_key}' não configurado em 'data'")

        path = Path(data[path_key])
        if path_key == 'raw_data_path' and not path.exists():
            logger.warning(f"Caminho de dados brutos não existe: {path}")

    models = config['models']
    if not models:
        raise ValueError("Nenhum modelo configurado")

    logger.info("Configuração validada com sucesso")
    logger.debug(f"Seções encontradas: {list(config.keys())}")
    logger.debug(f"Modelos configurados: {list(models.keys())}")


if __name__ == "__main__":
    main()