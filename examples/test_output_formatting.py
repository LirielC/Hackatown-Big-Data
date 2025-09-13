"""
Exemplo de uso do módulo de formatação de saída.

Este script demonstra como usar o SubmissionFormatter para gerar
arquivos de submissão em formato CSV e Parquet.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.output_formatter import SubmissionFormatter, SubmissionValidator
from src.models.prediction import PredictionGenerator


def create_sample_predictions():
    """Cria dados de exemplo para demonstração."""
    print("Criando dados de exemplo...")
    
    # Simular previsões para 5 semanas de janeiro
    np.random.seed(42)
    
    # Definir PDVs e produtos
    pdvs = [f"{i:03d}" for i in range(1, 21)]  # 20 PDVs
    produtos = [f"PROD_{chr(65+i)}" for i in range(10)]  # 10 produtos
    semanas = [1, 2, 3, 4, 5]
    
    # Gerar todas as combinações
    data = []
    for semana in semanas:
        for pdv in pdvs:
            for produto in produtos:
                # Simular quantidade com alguma variabilidade
                base_quantity = np.random.poisson(20)
                seasonal_factor = 1.2 if semana in [1, 5] else 1.0  # Maior demanda nas semanas 1 e 5
                quantity = int(base_quantity * seasonal_factor)
                
                data.append({
                    'semana': semana,
                    'pdv': pdv,
                    'produto': produto,
                    'quantidade': quantity
                })
    
    df = pd.DataFrame(data)
    print(f"Criadas {len(df)} previsões para {len(semanas)} semanas, {len(pdvs)} PDVs e {len(produtos)} produtos")
    
    return df


def demonstrate_csv_formatting():
    """Demonstra formatação CSV."""
    print("\n=== Demonstração de Formatação CSV ===")
    
    # Configuração
    config = {
        'output': {
            'csv_separator': ';',
            'csv_encoding': 'utf-8',
            'csv_decimal': '.',
            'validate_format': True,
            'add_timestamp': True
        }
    }
    
    # Criar dados de exemplo
    predictions_df = create_sample_predictions()
    
    # Inicializar formatador
    formatter = SubmissionFormatter(config)
    
    # Gerar resumo antes da formatação
    summary = formatter.generate_submission_summary(predictions_df)
    print(f"Resumo das previsões:")
    print(f"  - Total de previsões: {summary['submission_info']['total_predictions']}")
    print(f"  - Semanas cobertas: {summary['data_coverage']['weeks']}")
    print(f"  - PDVs únicos: {summary['data_coverage']['unique_pdvs']}")
    print(f"  - Produtos únicos: {summary['data_coverage']['unique_products']}")
    print(f"  - Quantidade total: {summary['quantity_summary']['total_quantity']}")
    
    # Formatar e salvar CSV
    output_path = "predictions_csv_example.csv"
    try:
        saved_path = formatter.format_submission_csv(predictions_df, output_path)
        print(f"Arquivo CSV salvo: {saved_path}")
        
        # Validar arquivo salvo
        validation = SubmissionValidator.validate_csv_format(saved_path)
        if validation['is_valid']:
            print("✓ Arquivo CSV validado com sucesso")
            print(f"  - Tamanho: {validation['file_info']['size_mb']} MB")
            print(f"  - Linhas: {validation['data_info']['total_rows']}")
        else:
            print("✗ Problemas na validação:")
            for error in validation['errors']:
                print(f"    - {error}")
                
    except Exception as e:
        print(f"Erro na formatação CSV: {e}")


def demonstrate_parquet_formatting():
    """Demonstra formatação Parquet."""
    print("\n=== Demonstração de Formatação Parquet ===")
    
    # Configuração
    config = {
        'output': {
            'parquet_compression': 'snappy',
            'validate_format': True,
            'add_timestamp': True
        }
    }
    
    # Criar dados de exemplo
    predictions_df = create_sample_predictions()
    
    # Inicializar formatador
    formatter = SubmissionFormatter(config)
    
    # Formatar e salvar Parquet
    output_path = "predictions_parquet_example.parquet"
    try:
        saved_path = formatter.format_submission_parquet(predictions_df, output_path)
        print(f"Arquivo Parquet salvo: {saved_path}")
        
        # Validar arquivo salvo
        validation = SubmissionValidator.validate_parquet_format(saved_path)
        if validation['is_valid']:
            print("✓ Arquivo Parquet validado com sucesso")
            print(f"  - Tamanho: {validation['file_info']['size_mb']} MB")
            print(f"  - Linhas: {validation['data_info']['total_rows']}")
            print(f"  - Tipos de dados: {validation['data_info']['data_types']}")
        else:
            print("✗ Problemas na validação:")
            for error in validation['errors']:
                print(f"    - {error}")
                
    except Exception as e:
        print(f"Erro na formatação Parquet: {e}")


def demonstrate_format_comparison():
    """Demonstra comparação entre formatos."""
    print("\n=== Demonstração de Comparação entre Formatos ===")
    
    # Configuração
    config = {
        'output': {
            'csv_separator': ';',
            'csv_encoding': 'utf-8',
            'parquet_compression': 'snappy',
            'validate_format': True,
            'add_timestamp': False  # Desabilitar para comparação
        }
    }
    
    # Criar dados menores para comparação
    predictions_df = pd.DataFrame({
        'semana': [1, 1, 2, 2, 3, 3],
        'pdv': ['001', '002', '001', '002', '001', '002'],
        'produto': ['A', 'A', 'B', 'B', 'C', 'C'],
        'quantidade': [10, 15, 20, 25, 12, 18]
    })
    
    formatter = SubmissionFormatter(config)
    
    # Salvar em ambos os formatos
    csv_path = "comparison_test.csv"
    parquet_path = "comparison_test.parquet"
    
    try:
        formatter.format_submission_csv(predictions_df, csv_path)
        formatter.format_submission_parquet(predictions_df, parquet_path)
        
        # Comparar formatos
        comparison = SubmissionValidator.compare_formats(csv_path, parquet_path)
        
        if comparison['files_consistent']:
            print("✓ Formatos são consistentes")
            print(f"  - CSV: {comparison['csv_info']['rows']} linhas, {comparison['csv_info']['file_size']} bytes")
            print(f"  - Parquet: {comparison['parquet_info']['rows']} linhas, {comparison['parquet_info']['file_size']} bytes")
        else:
            print("✗ Inconsistências encontradas:")
            for diff in comparison['differences']:
                print(f"    - {diff}")
                
    except Exception as e:
        print(f"Erro na comparação: {e}")


def demonstrate_validation_errors():
    """Demonstra validação com dados inválidos."""
    print("\n=== Demonstração de Validação com Erros ===")
    
    config = {
        'output': {
            'validate_format': True,
            'add_timestamp': False
        }
    }
    
    # Criar dados inválidos
    invalid_data = pd.DataFrame({
        'semana': [1, 2, 0, 6],  # Semanas inválidas: 0 e 6
        'pdv': ['001', '002', '003', None],  # Valor nulo
        'produto': ['A', 'B', 'C', 'D'],
        'quantidade': [10, -5, 20, 15]  # Quantidade negativa
    })
    
    formatter = SubmissionFormatter(config)
    
    try:
        formatter.format_submission_csv(invalid_data, "invalid_test.csv")
        print("✗ Validação deveria ter falhado")
    except Exception as e:
        print(f"✓ Validação capturou erro corretamente: {e}")


def demonstrate_integration_with_prediction():
    """Demonstra integração com módulo de predição."""
    print("\n=== Demonstração de Integração com Predição ===")
    
    # Configuração completa
    config = {
        'prediction': {
            'target_weeks': [1, 2, 3, 4, 5],
            'post_processing': {
                'ensure_positive': True,
                'apply_bounds': True
            }
        },
        'output': {
            'csv_separator': ';',
            'csv_encoding': 'utf-8',
            'validate_format': True,
            'add_timestamp': True
        }
    }
    
    # Criar dados de exemplo
    predictions_df = create_sample_predictions()
    
    # Usar PredictionGenerator para salvar
    try:
        generator = PredictionGenerator(config)
        
        # Salvar em CSV
        csv_path = generator.save_predictions(predictions_df, "integrated_example.csv", "csv")
        print(f"✓ CSV salvo via PredictionGenerator: {csv_path}")
        
        # Salvar em Parquet
        parquet_path = generator.save_predictions(predictions_df, "integrated_example.parquet", "parquet")
        print(f"✓ Parquet salvo via PredictionGenerator: {parquet_path}")
        
        # Gerar resumo
        summary = generator.generate_prediction_summary(predictions_df)
        print(f"✓ Resumo gerado: {summary['total_predictions']} previsões")
        
    except Exception as e:
        print(f"✗ Erro na integração: {e}")


def main():
    """Função principal do exemplo."""
    print("=== Exemplo de Formatação de Saída - Hackathon Forecast Model ===")
    
    # Criar diretório de saída se não existir
    Path("output_examples").mkdir(exist_ok=True)
    os.chdir("output_examples")
    
    try:
        # Executar demonstrações
        demonstrate_csv_formatting()
        demonstrate_parquet_formatting()
        demonstrate_format_comparison()
        demonstrate_validation_errors()
        demonstrate_integration_with_prediction()
        
        print("\n=== Exemplo concluído com sucesso! ===")
        print("Arquivos gerados no diretório 'output_examples'")
        
    except Exception as e:
        print(f"\nErro durante execução: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Voltar ao diretório original
        os.chdir("..")


if __name__ == "__main__":
    main()