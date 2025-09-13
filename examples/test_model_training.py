"""
Exemplo de uso dos modelos de treinamento.
Demonstra como treinar e avaliar modelos XGBoost, LightGBM e Prophet.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import yaml
from datetime import datetime, timedelta
import logging

from src.models.training import XGBoostModel, LightGBMModel, ProphetModel, ModelEvaluator

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config():
    """Carrega configuração do modelo."""
    with open('configs/model_config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def generate_sample_data():
    """Gera dados de exemplo para demonstração."""
    logger.info("Gerando dados de exemplo...")
    
    np.random.seed(42)
    
    # Gerar dados de 52 semanas (1 ano)
    dates = pd.date_range('2022-01-01', '2022-12-31', freq='W')
    n_weeks = len(dates)
    
    # Simular múltiplos PDVs e produtos
    n_pdvs = 10
    n_produtos = 5
    
    data = []
    
    for pdv in range(1, n_pdvs + 1):
        for produto in range(1, n_produtos + 1):
            for i, date in enumerate(dates):
                # Simular padrões de vendas realistas
                base_sales = 100 + pdv * 10 + produto * 5
                
                # Tendência temporal
                trend = i * 0.5
                
                # Sazonalidade
                seasonal = 20 * np.sin(2 * np.pi * i / 52) + 10 * np.sin(2 * np.pi * i / 13)
                
                # Ruído
                noise = np.random.randn() * 15
                
                # Vendas finais
                vendas = max(0, base_sales + trend + seasonal + noise)
                
                data.append({
                    'data': date,
                    'semana': i + 1,
                    'pdv': pdv,
                    'produto': produto,
                    'vendas': vendas,
                    'categoria_produto': f'cat_{produto % 3 + 1}',
                    'tipo_pdv': 'c-store' if pdv <= 5 else 'g-store'
                })
    
    df = pd.DataFrame(data)
    logger.info(f"Dados gerados: {len(df)} registros")
    
    return df


def create_features(df):
    """Cria features para treinamento dos modelos."""
    logger.info("Criando features...")
    
    # Ordenar por PDV, produto e data
    df = df.sort_values(['pdv', 'produto', 'data']).reset_index(drop=True)
    
    features_list = []
    
    for (pdv, produto), group in df.groupby(['pdv', 'produto']):
        group = group.sort_values('data').reset_index(drop=True)
        
        # Features temporais
        group['semana_ano'] = group['semana']
        group['mes'] = group['data'].dt.month
        group['trimestre'] = group['data'].dt.quarter
        
        # Features de lag
        group['lag_1'] = group['vendas'].shift(1)
        group['lag_2'] = group['vendas'].shift(2)
        group['lag_4'] = group['vendas'].shift(4)
        
        # Médias móveis
        group['media_movel_4'] = group['vendas'].rolling(window=4, min_periods=1).mean()
        group['media_movel_8'] = group['vendas'].rolling(window=8, min_periods=1).mean()
        
        # Crescimento
        group['crescimento_1sem'] = group['vendas'].pct_change(1)
        
        # Features de produto e PDV
        group['media_historica_produto'] = group['vendas'].expanding().mean()
        group['volatilidade_produto'] = group['vendas'].rolling(window=8, min_periods=1).std()
        
        features_list.append(group)
    
    df_features = pd.concat(features_list, ignore_index=True)
    
    # Preencher valores faltantes
    df_features = df_features.fillna(method='ffill').fillna(0)
    
    # Encoding de variáveis categóricas
    df_features['categoria_produto_encoded'] = pd.Categorical(df_features['categoria_produto']).codes
    df_features['tipo_pdv_encoded'] = pd.Categorical(df_features['tipo_pdv']).codes
    
    logger.info(f"Features criadas: {df_features.shape[1]} colunas")
    
    return df_features


def prepare_model_data(df_features):
    """Prepara dados para treinamento dos modelos."""
    # Selecionar features para modelos de ML
    feature_columns = [
        'semana_ano', 'mes', 'trimestre',
        'lag_1', 'lag_2', 'lag_4',
        'media_movel_4', 'media_movel_8',
        'crescimento_1sem',
        'media_historica_produto', 'volatilidade_produto',
        'categoria_produto_encoded', 'tipo_pdv_encoded'
    ]
    
    X = df_features[feature_columns].copy()
    y = df_features['vendas'].copy()
    
    # Remover primeiras semanas com muitos NaN
    valid_idx = X.notna().all(axis=1) & (X != 0).any(axis=1)
    X = X[valid_idx]
    y = y[valid_idx]
    
    return X, y


def prepare_prophet_data(df_features):
    """Prepara dados específicos para Prophet."""
    # Prophet precisa de dados agregados por data
    prophet_data = df_features.groupby('data').agg({
        'vendas': 'sum',
        'semana_ano': 'first',
        'mes': 'first'
    }).reset_index()
    
    # Adicionar features como regressores
    prophet_data['semana_ano_norm'] = prophet_data['semana_ano'] / 52
    prophet_data['mes_norm'] = prophet_data['mes'] / 12
    
    X_prophet = prophet_data[['data', 'semana_ano_norm', 'mes_norm']].copy()
    X_prophet.columns = ['ds', 'semana_regressor', 'mes_regressor']
    
    y_prophet = prophet_data['vendas']
    
    return X_prophet, y_prophet


def test_xgboost_model(config, X, y):
    """Testa o modelo XGBoost."""
    logger.info("=== Testando XGBoost ===")
    
    # Dividir dados
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Treinar modelo
    model = XGBoostModel(config)
    model.fit(X_train, y_train, X_test, y_test, optimize_hyperparams=False)
    
    # Gerar previsões
    y_pred = model.predict(X_test)
    
    # Avaliar modelo
    metrics = ModelEvaluator.evaluate_model(y_test.values, y_pred)
    
    logger.info("Métricas XGBoost:")
    for metric, value in metrics.items():
        logger.info(f"  {metric.upper()}: {value:.4f}")
    
    # Mostrar importância das features
    logger.info("Top 5 features mais importantes:")
    top_features = model.feature_importance_.head()
    for _, row in top_features.iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    return model, metrics


def test_lightgbm_model(config, X, y):
    """Testa o modelo LightGBM."""
    logger.info("=== Testando LightGBM ===")
    
    # Dividir dados
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Treinar modelo
    model = LightGBMModel(config)
    model.fit(X_train, y_train, X_test, y_test, optimize_hyperparams=False)
    
    # Gerar previsões
    y_pred = model.predict(X_test)
    
    # Avaliar modelo
    metrics = ModelEvaluator.evaluate_model(y_test.values, y_pred)
    
    logger.info("Métricas LightGBM:")
    for metric, value in metrics.items():
        logger.info(f"  {metric.upper()}: {value:.4f}")
    
    return model, metrics


def test_prophet_model(config, X_prophet, y_prophet):
    """Testa o modelo Prophet."""
    logger.info("=== Testando Prophet ===")
    
    # Dividir dados
    split_idx = int(len(X_prophet) * 0.8)
    X_train, X_test = X_prophet[:split_idx], X_prophet[split_idx:]
    y_train, y_test = y_prophet[:split_idx], y_prophet[split_idx:]
    
    # Treinar modelo
    model = ProphetModel(config)
    model.fit(X_train, y_train)
    
    # Gerar previsões
    y_pred = model.predict(X_test)
    
    # Avaliar modelo
    metrics = ModelEvaluator.evaluate_model(y_test.values, y_pred)
    
    logger.info("Métricas Prophet:")
    for metric, value in metrics.items():
        logger.info(f"  {metric.upper()}: {value:.4f}")
    
    return model, metrics


def compare_models(xgb_metrics, lgb_metrics, prophet_metrics):
    """Compara performance dos modelos."""
    logger.info("=== Comparação de Modelos ===")
    
    models_comparison = pd.DataFrame({
        'XGBoost': xgb_metrics,
        'LightGBM': lgb_metrics,
        'Prophet': prophet_metrics
    })
    
    logger.info("Comparação de métricas:")
    logger.info(f"\n{models_comparison}")
    
    # Identificar melhor modelo por métrica
    logger.info("\nMelhor modelo por métrica:")
    for metric in models_comparison.index:
        best_model = models_comparison.loc[metric].idxmin()
        best_value = models_comparison.loc[metric].min()
        logger.info(f"  {metric.upper()}: {best_model} ({best_value:.4f})")


def main():
    """Função principal."""
    logger.info("Iniciando teste dos modelos de treinamento")
    
    try:
        # Carregar configuração
        config = load_config()
        
        # Gerar dados de exemplo
        df = generate_sample_data()
        
        # Criar features
        df_features = create_features(df)
        
        # Preparar dados para modelos ML
        X, y = prepare_model_data(df_features)
        logger.info(f"Dados preparados: {X.shape[0]} amostras, {X.shape[1]} features")
        
        # Preparar dados para Prophet
        X_prophet, y_prophet = prepare_prophet_data(df_features)
        
        # Testar XGBoost
        xgb_model, xgb_metrics = test_xgboost_model(config, X, y)
        
        # Testar LightGBM
        lgb_model, lgb_metrics = test_lightgbm_model(config, X, y)
        
        # Testar Prophet
        prophet_model, prophet_metrics = test_prophet_model(config, X_prophet, y_prophet)
        
        # Comparar modelos
        compare_models(xgb_metrics, lgb_metrics, prophet_metrics)
        
        logger.info("Teste concluído com sucesso!")
        
    except Exception as e:
        logger.error(f"Erro durante execução: {str(e)}")
        raise


if __name__ == "__main__":
    main()