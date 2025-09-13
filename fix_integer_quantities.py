#!/usr/bin/env python3
"""
Script para corrigir valores decimais para inteiros no arquivo de submissão
"""

import pandas as pd
from pathlib import Path
import numpy as np

def fix_integer_quantities():
    """Converte valores decimais para inteiros na coluna quantidade"""

    print('🔧 CORRIGINDO VALORES PARA INTEIROS')
    print('=' * 40)

    # Arquivo de entrada (corrigido anteriormente)
    input_file = 'final_submission/hackathon_forecast_submission_corrected_final.csv'
    output_file = 'final_submission/hackathon_forecast_submission_final_integer.csv'

    try:
        # Ler arquivo
        print('📥 Carregando arquivo...')
        df = pd.read_csv(input_file, sep=';')
        print(f'   Arquivo carregado: {len(df):,} registros')

        # Verificar valores atuais
        print(f'\\n📊 Valores atuais na coluna quantidade:')
        print(f'   Tipo de dados: {df["quantidade"].dtype}')
        print(f'   Média: {df["quantidade"].mean():.4f}')
        print(f'   Mínimo: {df["quantidade"].min():.4f}')
        print(f'   Máximo: {df["quantidade"].max():.4f}')

        # Verificar valores decimais
        decimal_count = (~df['quantidade'].astype(str).str.contains(r'^\d+\.0+$')).sum()
        print(f'   Valores decimais: {decimal_count:,} ({decimal_count/len(df)*100:.1f}%)')

        # Converter para inteiros (arredondamento)
        print(f'\\n🔄 Convertendo para inteiros...')
        df['quantidade'] = df['quantidade'].round().astype(int)

        # Verificar conversão
        print(f'\\n✅ Valores após conversão:')
        print(f'   Tipo de dados: {df["quantidade"].dtype}')
        print(f'   Média: {df["quantidade"].mean():.2f}')
        print(f'   Mínimo: {df["quantidade"].min()}')
        print(f'   Máximo: {df["quantidade"].max()}')

        # Verificar distribuição de valores
        unique_values = sorted(df['quantidade'].unique())
        print(f'   Valores únicos: {len(unique_values)}')
        print(f'   Exemplos: {unique_values[:10]}...')

        # Verificar valores zero ou negativos
        zero_count = (df['quantidade'] == 0).sum()
        negative_count = (df['quantidade'] < 0).sum()
        print(f'   Valores zero: {zero_count:,} ({zero_count/len(df)*100:.1f}%)')
        print(f'   Valores negativos: {negative_count:,} ({negative_count/len(df)*100:.1f}%)')

        # Salvar arquivo corrigido
        print(f'\\n💾 Salvando arquivo corrigido...')
        df.to_csv(output_file, sep=';', index=False, encoding='utf-8')
        print(f'   Arquivo salvo: {output_file}')

        # Verificar primeiras linhas
        print(f'\\n🔍 Primeiras linhas do arquivo corrigido:')
        print(df.head().to_string())

        # Verificar se todos são inteiros
        all_integers = df['quantidade'].dtype == 'int64' or df['quantidade'].dtype == 'int32'
        if all_integers:
            print(f'\\n✅ SUCESSO: Todos os valores são inteiros!')
            print(f'🏆 Arquivo pronto para submissão!')
        else:
            print(f'\\n⚠️ ATENÇÃO: Ainda há valores não-inteiros')

        return df

    except Exception as e:
        print(f'❌ Erro na correção: {e}')
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = fix_integer_quantities()

    if result is not None:
        print(f'\\n🎯 RESUMO DA CORREÇÃO:')
        print(f'📄 Arquivo original: hackathon_forecast_submission_corrected_final.csv')
        print(f'📄 Arquivo corrigido: hackathon_forecast_submission_final_integer.csv')
        print(f'📊 Registros: {len(result):,}')
        print(f'🎲 Média: {result["quantidade"].mean():.2f}')
        print(f'📈 Valores zero: {(result["quantidade"] == 0).sum():,}')

        # Estimativa de impacto no WMAPE
        original_mean = 3.2205
        new_mean = result['quantidade'].mean()
        impact = abs(original_mean - new_mean) / original_mean * 100
        print(f'📊 Impacto estimado no WMAPE: {impact:.1f}%')

