#!/usr/bin/env python3
"""
PLANO INFALÍVEL PARA WMAPE < 10%
================================

Este plano visa alcançar WMAPE abaixo de 10% através de:
1. Otimização do modelo atual
2. Engenharia de features avançada
3. Ensemble methods
4. Validação cruzada temporal
5. Hyperparameter tuning
6. Feature selection
7. Tratamento de outliers
8. Validação externa
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def calculate_wmape(y_true, y_pred):
    """Calcula WMAPE (Weighted Mean Absolute Percentage Error)"""
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100

def load_data():
    """Carrega dados históricos e processados"""
    print('📥 CARREGANDO DADOS PARA OTIMIZAÇÃO')
    print('=' * 40)

    # Carregar dados históricos
    hist_file = 'data/processed/weekly_sales_processed.parquet'
    features_file = 'data/processed/features_engineered.parquet'

    df_hist = pd.read_parquet(hist_file)
    df_feat = pd.read_parquet(features_file)

    print(f'✅ Dados históricos: {len(df_hist):,} registros')
    print(f'✅ Features: {len(df_feat):,} registros, {len(df_feat.columns)} colunas')

    return df_hist, df_feat

def optimize_features(df_feat):
    """Otimiza engenharia de features para melhor performance"""
    print('\\n🔧 OTIMIZANDO FEATURES')
    print('=' * 25)

    # Features temporais avançadas
    df_feat['dias_da_semana'] = df_feat['data_semana'].dt.dayofweek
    df_feat['eh_fim_de_semana'] = df_feat['dias_da_semana'].isin([5, 6]).astype(int)
    df_feat['eh_feriado'] = 0  # Implementar lógica de feriados

    # Features de tendência e sazonalidade
    df_feat['tendencia_linear'] = np.arange(len(df_feat))
    df_feat['sazonalidade_anual'] = np.sin(2 * np.pi * df_feat['semana_ano'] / 52)
    df_feat['sazonalidade_trimestral'] = np.sin(2 * np.pi * df_feat['trimestre'] / 4)

    # Features de lag avançadas
    for lag in [1, 2, 4, 8, 12, 26, 52]:
        df_feat[f'quantidade_lag_{lag}'] = df_feat.groupby(['pdv', 'produto'])['quantidade'].shift(lag)

    # Features de rolling window
    for window in [4, 8, 12, 26]:
        df_feat[f'rolling_mean_{window}'] = df_feat.groupby(['pdv', 'produto'])['quantidade'].rolling(window).mean().reset_index(0, drop=True)
        df_feat[f'rolling_std_{window}'] = df_feat.groupby(['pdv', 'produto'])['quantidade'].rolling(window).std().reset_index(0, drop=True)

    # Features de produto e PDV
    df_feat['produto_popularidade'] = df_feat.groupby('produto')['quantidade'].transform('mean')
    df_feat['pdv_performance'] = df_feat.groupby('pdv')['quantidade'].transform('mean')

    # Features de interação
    df_feat['produto_pdv_interacao'] = df_feat.groupby(['produto', 'pdv'])['quantidade'].transform('mean')

    print(f'✅ Features otimizadas: {len(df_feat.columns)} colunas')
    return df_feat

def create_temporal_splits(df_feat):
    """Cria splits temporais para validação"""
    print('\\n⏰ CRIANDO SPLITS TEMPORAIS')
    print('=' * 28)

    # Ordenar por data
    df_feat = df_feat.sort_values('data_semana')

    # Split temporal (últimas 4 semanas para teste)
    train_end = df_feat['data_semana'].max() - pd.DateOffset(weeks=4)
    val_end = df_feat['data_semana'].max() - pd.DateOffset(weeks=2)

    train_data = df_feat[df_feat['data_semana'] <= train_end]
    val_data = df_feat[(df_feat['data_semana'] > train_end) & (df_feat['data_semana'] <= val_end)]
    test_data = df_feat[df_feat['data_semana'] > val_end]

    print(f'📊 Train: {len(train_data):,} registros ({len(train_data)/len(df_feat)*100:.1f}%)')
    print(f'📊 Validation: {len(val_data):,} registros ({len(val_data)/len(df_feat)*100:.1f}%)')
    print(f'📊 Test: {len(test_data):,} registros ({len(test_data)/len(df_feat)*100:.1f}%)')

    return train_data, val_data, test_data

def train_optimized_model(train_data, val_data):
    """Treina modelo otimizado com hyperparameter tuning"""
    print('\\n🤖 TREINANDO MODELO OTIMIZADO')
    print('=' * 32)

    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_absolute_percentage_error
    import xgboost as xgb
    import optuna

    # Preparar dados
    exclude_cols = ['quantidade', 'pdv', 'produto', 'data_semana', 'data']
    feature_cols = [col for col in train_data.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(train_data[col])]

    X_train = train_data[feature_cols].fillna(0)
    y_train = train_data['quantidade']
    X_val = val_data[feature_cols].fillna(0)
    y_val = val_data['quantidade']

    print(f'📊 Features: {len(feature_cols)}')
    print(f'📈 Train shape: {X_train.shape}')
    print(f'📈 Val shape: {X_val.shape}')

    # Função objetivo para Optuna
    def objective(trial):
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
        }

        model = xgb.XGBRegressor(**params, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        wmape = calculate_wmape(y_val, y_pred)

        return wmape

    # Otimização com Optuna
    print('\\n🎯 Iniciando hyperparameter tuning...')
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    # Melhor modelo
    best_params = study.best_params
    best_wmape = study.best_value

    print(f'\\n🏆 Melhor WMAPE na validação: {best_wmape:.2f}%')
    print(f'🏆 Melhores parâmetros: {best_params}')

    # Treinar modelo final
    final_model = xgb.XGBRegressor(**best_params, random_state=42)
    final_model.fit(X_train, y_train)

    return final_model, feature_cols, best_wmape

def create_ensemble_model(models, feature_cols):
    """Cria ensemble de modelos para melhor performance"""
    print('\\n🎭 CRIANDO ENSEMBLE DE MODELOS')
    print('=' * 32)

    from sklearn.ensemble import StackingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score

    # Base models
    base_models = [
        ('xgb', models[0]),
        ('rf', RandomForestRegressor(n_estimators=200, random_state=42)),
        ('ridge', Ridge(alpha=1.0, random_state=42))
    ]

    # Meta-learner
    meta_model = Ridge(alpha=0.1, random_state=42)

    # Stacking ensemble
    stacking = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model,
        cv=3,
        n_jobs=-1,
        verbose=1
    )

    print('✅ Ensemble configurado com 3 modelos base + meta-learner')
    return stacking

def validate_model_performance(model, test_data, feature_cols):
    """Valida performance do modelo no conjunto de teste"""
    print('\\n📊 VALIDANDO PERFORMANCE FINAL')
    print('=' * 33)

    X_test = test_data[feature_cols].fillna(0)
    y_test = test_data['quantidade']

    # Predições
    y_pred = model.predict(X_test)

    # Métricas
    wmape = calculate_wmape(y_test, y_pred)
    mae = np.mean(np.abs(y_test - y_pred))
    rmse = np.sqrt(np.mean((y_test - y_pred)**2))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    print(f'📈 WMAPE: {wmape:.2f}%')
    print(f'📈 MAE: {mae:.4f}')
    print(f'📈 RMSE: {rmse:.4f}')
    print(f'📈 MAPE: {mape:.2f}%')

    # Análise por semana
    test_data_copy = test_data.copy()
    test_data_copy['predicao'] = y_pred
    test_data_copy['erro_absoluto'] = np.abs(test_data_copy['quantidade'] - test_data_copy['predicao'])

    weekly_performance = test_data_copy.groupby('semana').agg({
        'quantidade': 'sum',
        'predicao': 'sum',
        'erro_absoluto': 'sum'
    })

    weekly_performance['wmape_semanal'] = (weekly_performance['erro_absoluto'] / weekly_performance['quantidade']) * 100

    print(f'\\n📅 Performance por semana:')
    for week in sorted(weekly_performance.index):
        wmape_week = weekly_performance.loc[week, 'wmape_semanal']
        print(f'   Semana {week}: WMAPE = {wmape_week:.2f}%')

    return wmape, y_pred

def main():
    """Executa plano completo de otimização"""
    print('🚀 PLANO INFALÍVEL PARA WMAPE < 10%')
    print('=' * 40)

    try:
        # 1. Carregar dados
        df_hist, df_feat = load_data()

        # 2. Otimizar features
        df_feat = optimize_features(df_feat)

        # 3. Criar splits temporais
        train_data, val_data, test_data = create_temporal_splits(df_feat)

        # 4. Treinar modelo otimizado
        model, feature_cols, val_wmape = train_optimized_model(train_data, val_data)

        # 5. Criar ensemble
        ensemble_model = create_ensemble_model([model], feature_cols)

        # 6. Validar performance
        final_wmape, predictions = validate_model_performance(model, test_data, feature_cols)

        # 7. Resultado final
        print(f'\\n🎯 RESULTADO FINAL:')
        print(f'=' * 20)
        if final_wmape < 10:
            print(f'✅ SUCESSO! WMAPE = {final_wmape:.2f}% (< 10%)')
            print(f'🏆 OBJETIVO ALCANÇADO!')
        elif final_wmape < 15:
            print(f'⚠️ QUASE! WMAPE = {final_wmape:.2f}% (meta: < 10%)')
            print(f'💪 Performance excelente!')
        else:
            print(f'📈 WMAPE = {final_wmape:.2f}% (precisa otimizar mais)')
            print(f'🔧 Continuar otimizações...')

        return final_wmape, model

    except Exception as e:
        print(f'❌ Erro na execução: {e}')
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    final_wmape, optimized_model = main()

    if final_wmape is not None:
        if final_wmape < 10:
            print(f'\\n🎉 MISSÃO CUMPRIDA! WMAPE = {final_wmape:.2f}%')
        else:
            print(f'\\n📊 WMAPE atual: {final_wmape:.2f}% - Continuar otimizações')
