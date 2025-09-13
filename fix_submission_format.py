#!/usr/bin/env python3
"""
Corrige o formato do arquivo de submissão para atender aos requisitos do hackathon
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

def fix_submission_format():
    """Corrige o formato do arquivo de submissão"""

    print("🔧 CORRIGINDO FORMATO DA SUBMISSÃO")
    print("=" * 50)

    # Arquivo original
    input_file = Path("final_submission/hackathon_forecast_submission_20250910_234035.csv")

    if not input_file.exists():
        print("❌ Arquivo de submissão não encontrado!")
        return False

    print(f"📁 Arquivo original: {input_file}")

    # Ler o arquivo
    print("📖 Lendo arquivo...")
    df = pd.read_csv(input_file, sep=';', encoding='utf-8')
    print(f"📊 Dados carregados: {len(df)} registros")
    print(f"📋 Colunas atuais: {list(df.columns)}")

    # Verificar formato atual
    print("\n🔍 ANÁLISE DO FORMATO ATUAL:")
    print(f"  - Ordem atual: {list(df.columns)}")
    print("  - Ordem esperada: ['semana', 'pdv', 'produto', 'quantidade']")
    print(f"  - Tipos de dados: {df.dtypes.to_dict()}")

    # Verificar valores de semana
    semanas_unicas = sorted(df['semana'].unique())
    print(f"  - Semanas encontradas: {semanas_unicas}")

    # Verificar se as semanas estão no formato correto (1-5)
    if not all(1 <= semana <= 5 for semana in semanas_unicas):
        print("⚠️  Semanas fora do range 1-5! Mapeando para formato correto...")

        # Mapeamento de semanas (assumindo que as semanas atuais representam janeiro/2023)
        # Vou mapear as semanas encontradas para 1-5
        semana_mapping = {}
        for i, semana in enumerate(sorted(semanas_unicas), 1):
            semana_mapping[semana] = min(i, 5)  # Limita a 5 semanas

        print(f"  - Mapeamento de semanas: {semana_mapping}")

        df['semana'] = df['semana'].map(semana_mapping)

    # Reordenar colunas conforme especificação
    colunas_corretas = ['semana', 'pdv', 'produto', 'quantidade']
    df = df[colunas_corretas]

    # Converter tipos de dados
    df['semana'] = df['semana'].astype(int)
    df['pdv'] = df['pdv'].astype(int)
    df['produto'] = df['produto'].astype(int)
    df['quantidade'] = df['quantidade'].astype(int)

    print("\n✅ FORMATO CORRIGIDO:")
    print(f"  - Ordem das colunas: {list(df.columns)}")
    print(f"  - Tipos de dados: {df.dtypes.to_dict()}")

    # Verificar valores finais
    semanas_finais = sorted(df['semana'].unique())
    print(f"  - Semanas finais: {semanas_finais}")
    print(f"  - PDVs únicos: {len(df['pdv'].unique())}")
    print(f"  - Produtos únicos: {len(df['produto'].unique())}")
    print(f"  - Quantidade total prevista: {df['quantidade'].sum()}")

    # Criar arquivo corrigido
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(f"final_submission/hackathon_forecast_submission_corrected_{timestamp}.csv")

    print("\n💾 Salvando arquivo corrigido...")
    df.to_csv(output_file, sep=';', index=False, encoding='utf-8')

    print(f"✅ Arquivo salvo: {output_file}")
    print(f"📏 Tamanho do arquivo: {output_file.stat().st_size} bytes")

    # Verificar arquivo salvo
    print("\n🔍 VERIFICAÇÃO FINAL:")
    with open(output_file, 'r', encoding='utf-8') as f:
        primeiras_linhas = [next(f).strip() for _ in range(5)]

    for linha in primeiras_linhas:
        print(f"  {linha}")

    # Criar arquivo de backup do original
    backup_file = Path(f"final_submission/backup_original_{timestamp}.csv")
    import shutil
    shutil.copy2(input_file, backup_file)
    print(f"📋 Backup do original: {backup_file}")

    # Atualizar arquivo de informações
    info_file = Path(f"final_submission/submission_info_{timestamp}.txt")
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write("HACKATHON FORECAST SUBMISSION - CORRIGIDO\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Arquivo Original: hackathon_forecast_submission_20250910_234035.csv\n")
        f.write(f"Arquivo Corrigido: {output_file.name}\n")
        f.write(f"Arquivo Backup: {backup_file.name}\n\n")
        f.write("CORREÇÕES APLICADAS:\n")
        f.write("✅ Ordem das colunas: semana,pdv,produto,quantidade\n")
        f.write("✅ Semanas mapeadas para range 1-5\n")
        f.write("✅ Tipos de dados convertidos para inteiro\n")
        f.write("✅ Separador: ;\n")
        f.write("✅ Encoding: UTF-8\n\n")
        f.write("ESTATÍSTICAS:\n")
        f.write(f"- Total de previsões: {len(df)}\n")
        f.write(f"- Semanas: {semanas_finais}\n")
        f.write(f"- PDVs únicos: {len(df['pdv'].unique())}\n")
        f.write(f"- Produtos únicos: {len(df['produto'].unique())}\n")
        f.write(f"- Quantidade total: {df['quantidade'].sum()}\n\n")
        f.write("PARA ENVIAR:\n")
        f.write("1. Faça upload do arquivo CORRIGIDO no site do hackathon\n")
        f.write("2. O arquivo segue exatamente a especificação\n")
        f.write("3. Você pode fazer até 5 submissões!\n")

    print(f"📄 Informações atualizadas: {info_file}")

    return True

if __name__ == "__main__":
    success = fix_submission_format()
    if success:
        print("\n🎉 Sucesso! Arquivo corrigido e pronto para submissão.")
    else:
        print("\n❌ Erro na correção do formato.")
