#!/usr/bin/env python3
"""
Script para verificar qualidade dos dados históricos
"""

import pandas as pd
from pathlib import Path

def check_data_quality():
    """Verifica qualidade dos dados históricos e features"""

    print('🔍 VERIFICANDO DADOS HISTÓRICOS')
    print('=' * 35)

    try:
        # Carregar dados históricos
        historical_file = 'data/processed/weekly_sales_processed.parquet'
        if Path(historical_file).exists():
            df_hist = pd.read_parquet(historical_file)
            print(f'✅ Dados históricos: {len(df_hist):,} registros')

            # Estatísticas
            zero_hist = (df_hist['quantidade'] == 0).sum()
            non_zero_hist = (df_hist['quantidade'] != 0).sum()
            zero_hist_pct = zero_hist / len(df_hist) * 100

            print(f'Quantidade - Média: {df_hist["quantidade"].mean():.2f}')
            print(f'Quantidade - Mediana: {df_hist["quantidade"].median():.2f}')
            print(f'Quantidade - Máximo: {df_hist["quantidade"].max()}')

            print(f'Valores zero: {zero_hist:,} ({zero_hist_pct:.1f}%)')
            print(f'Valores não-zero: {non_zero_hist:,} ({100-zero_hist_pct:.1f}%)')

            if zero_hist_pct > 90:
                print('❌ CRÍTICO: Quase todos os valores históricos são zero!')
                print('Isso explica por que as previsões são zero.')
            elif zero_hist_pct > 50:
                print('⚠️ GRAVE: Muitos valores zero nos dados históricos!')
            else:
                print('✅ Dados históricos parecem OK.')
        else:
            print(f'❌ Dados históricos não encontrados: {historical_file}')

    except Exception as e:
        print(f'❌ Erro dados históricos: {e}')

    print('\n🔍 VERIFICANDO FEATURES')
    print('=' * 25)

    try:
        features_file = 'data/processed/features_engineered.parquet'
        if Path(features_file).exists():
            df_feat = pd.read_parquet(features_file)
            print(f'✅ Features: {len(df_feat):,} registros, {len(df_feat.columns)} colunas')

            if 'quantidade' in df_feat.columns:
                print('✅ Target presente')
            else:
                print('❌ Target ausente')

            # Contar features
            exclude_cols = ['pdv', 'produto', 'data_semana', 'data', 'quantidade']
            feature_cols = [col for col in df_feat.columns if col not in exclude_cols]
            print(f'Features numéricas: {len(feature_cols)}')
        else:
            print(f'❌ Features não encontradas: {features_file}')

    except Exception as e:
        print(f'❌ Erro features: {e}')

    print('\n🔍 VERIFICANDO MODELO')
    print('=' * 20)

    try:
        model_file = 'models/hackathon_forecast_model.pkl'
        if Path(model_file).exists():
            print('✅ Modelo treinado encontrado')
            import joblib
            model_data = joblib.load(model_file)
            print('✅ Modelo carregado com sucesso')

            # Verificar se é um modelo válido
            if hasattr(model_data, 'predict'):
                print('✅ Modelo tem método predict')
            else:
                print('❌ Modelo não tem método predict')
        else:
            print('❌ Modelo não encontrado')

    except Exception as e:
        print(f'❌ Erro modelo: {e}')

if __name__ == "__main__":
    check_data_quality()

