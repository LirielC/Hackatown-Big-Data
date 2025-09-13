#!/usr/bin/env python3
"""
Script para corrigir o pipeline de predição
"""

import pandas as pd
import joblib
from pathlib import Path

def fix_prediction_pipeline():
    """Corrige o pipeline de predição que está gerando zeros"""

    print('🔧 CORRIGINDO PIPELINE DE PREDIÇÃO')
    print('=' * 35)

    try:
        # 1. Carregar modelo
        print('1. 📥 Carregando modelo...')
        model = joblib.load('models/hackathon_forecast_model.pkl')
        expected_features = model.feature_names_in_.tolist()
        print(f'   Modelo carrega com {len(expected_features)} features esperadas')

        # 2. Carregar dados processados
        print('\\n2. 📊 Carregando dados processados...')
        features_path = 'data/processed/features_engineered.parquet'
        df = pd.read_parquet(features_path)
        print(f'   Dados carregados: {len(df):,} registros')

        # 3. Criar dados de predição CORRETAMENTE
        print('\\n3. 🎯 Criando dados de predição...')

        # Usar dados de dezembro/2022
        dec_data = df[df['data_semana'] >= '2022-12-01'].copy()
        print(f'   Dados base (dezembro): {len(dec_data)} registros')

        # Criar dados de janeiro/2023
        pred_data = dec_data.copy()

        # Atualizar campos temporais
        pred_data['ano'] = 2023
        pred_data['mes'] = 1
        pred_data['trimestre'] = 1
        pred_data['data_semana'] = pred_data['data_semana'] + pd.DateOffset(months=1)

        # Recalcular features temporais
        pred_data['semana_ano'] = pred_data['data_semana'].dt.isocalendar().week
        pred_data['semana'] = pred_data['semana_ano']

        print(f'   Dados de janeiro/2023 criados: {len(pred_data)} registros')

        # 4. Preparar features para predição
        print('\\n4. 🔧 Preparando features...')

        # IMPORTANTE: Usar exatamente as mesmas features do modelo
        X_pred = pred_data[expected_features].copy()
        X_pred = X_pred.fillna(0)  # Preencher NaN com 0

        print(f'   Features preparadas: {X_pred.shape}')
        print(f'   Features: {list(X_pred.columns[:5])}...')

        # Verificar se 'semana' está presente
        if 'semana' in X_pred.columns:
            print('   ✅ Feature \"semana\" presente')
            print(f'   Valores únicos de semana: {sorted(X_pred["semana"].unique())}')
        else:
            print('   ❌ Feature \"semana\" AUSENTE!')
            print(f'   Colunas disponíveis: {list(X_pred.columns[:10])}...')
            return

        # 5. Fazer predições em lotes (para não sobrecarregar memória)
        print('\\n5. 🎲 Gerando predições...')

        batch_size = 10000
        predictions = []

        for i in range(0, len(X_pred), batch_size):
            batch_end = min(i + batch_size, len(X_pred))
            X_batch = X_pred.iloc[i:batch_end]

            batch_predictions = model.predict(X_batch)
            predictions.extend(batch_predictions)

            if i % 50000 == 0:
                print(f'   Processado: {batch_end:,}/{len(X_pred):,} registros')

        predictions = pd.Series(predictions, name='quantidade')
        print(f'   Predições geradas: {len(predictions):,} valores')

        # 6. Verificar qualidade das predições
        print('\\n6. 📊 Verificando qualidade...')
        zero_count = (predictions == 0).sum()
        zero_percentage = zero_count / len(predictions) * 100

        print(f'   Média das predições: {predictions.mean():.4f}')
        print(f'   Mediana: {predictions.median():.4f}')
        print(f'   Mínimo: {predictions.min():.4f}')
        print(f'   Máximo: {predictions.max():.4f}')
        print(f'   Valores zero: {zero_count:,} ({zero_percentage:.1f}%)')

        # 7. Criar DataFrame final
        print('\\n7. 📄 Criando arquivo final...')

        final_df = pd.DataFrame({
            'semana': pred_data['semana'].values,
            'pdv': pred_data['pdv'].values,
            'produto': pred_data['produto'].values,
            'quantidade': predictions.values
        })

        # Remover duplicatas (mantendo primeira ocorrência)
        final_df = final_df.drop_duplicates(subset=['semana', 'pdv', 'produto'], keep='first')
        print(f'   Após remoção de duplicatas: {len(final_df):,} registros')

        # Verificar novamente
        final_zero_count = (final_df['quantidade'] == 0).sum()
        final_zero_percentage = final_zero_count / len(final_df) * 100

        print(f'   Média final: {final_df["quantidade"].mean():.4f}')
        print(f'   Valores zero finais: {final_zero_count:,} ({final_zero_percentage:.1f}%)')

        # 8. Salvar arquivo corrigido
        output_file = 'final_submission/hackathon_forecast_submission_corrected_final.csv'
        final_df.to_csv(output_file, sep=';', index=False, encoding='utf-8')

        print(f'\\n💾 Arquivo corrigido salvo: {output_file}')
        print(f'📊 Tamanho: {len(final_df):,} registros')

        # Verificar primeiras linhas
        print('\\n🔍 Primeiras linhas do arquivo corrigido:')
        print(final_df.head().to_string())

        return final_df

    except Exception as e:
        print(f'❌ Erro na correção: {e}')
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print('🚀 INICIANDO CORREÇÃO DO PIPELINE DE PREDIÇÃO')
    print('=' * 50)

    result = fix_prediction_pipeline()

    if result is not None:
        zero_count = (result['quantidade'] == 0).sum()
        zero_percentage = zero_count / len(result) * 100

        print(f'\\n🎯 RESULTADO FINAL:')
        if zero_percentage == 0:
            print('✅ SUCESSO TOTAL: Nenhuma predição é zero!')
        elif zero_percentage < 10:
            print('✅ SUCESSO: Predições corrigidas!')
        elif zero_percentage < 50:
            print('⚠️ MELHORIA: Predições parcialmente corrigidas')
        else:
            print('❌ PROBLEMA: Ainda há muitas predições zero')

        print(f'📄 Arquivo final: final_submission/hackathon_forecast_submission_corrected_final.csv')
        print(f'📊 Registros: {len(result):,}')
        print(f'🎲 Média: {result["quantidade"].mean():.4f}')
        print(f'📈 Valores zero: {zero_percentage:.1f}%')

        print('\\n🏆 PRONTO PARA SUBMISSÃO!')
