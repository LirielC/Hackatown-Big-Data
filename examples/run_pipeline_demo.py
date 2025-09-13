#!/usr/bin/env python3
"""
Exemplo de execução do pipeline principal.
Demonstra diferentes formas de usar o pipeline de previsão.
"""

import sys
import os
from pathlib import Path

# Adicionar diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import (
    load_config, validate_config, setup_environment,
    run_data_ingestion, run_data_preprocessing,
    PipelineExecutionTracker, run_full_pipeline
)


def demo_pipeline_steps():
    """Demonstra execução de etapas individuais do pipeline."""
    print("=== DEMO: Execução de Etapas Individuais ===")
    
    # Carregar configuração
    config = load_config("configs/model_config.yaml")
    
    # Validar configuração
    validate_config(config)
    print("✓ Configuração validada")
    
    # Configurar ambiente
    tracker = setup_environment(config)
    print("✓ Ambiente configurado para reprodutibilidade")
    
    # Executar etapas individuais
    try:
        print("\n1. Executando ingestão de dados...")
        run_data_ingestion(config)
        print("✓ Ingestão concluída")
        
        print("\n2. Executando pré-processamento...")
        run_data_preprocessing(config)
        print("✓ Pré-processamento concluído")
        
        print("\n✓ Demo de etapas individuais concluído com sucesso!")
        
    except Exception as e:
        print(f"✗ Erro durante execução: {e}")


def demo_execution_tracker():
    """Demonstra uso do rastreador de execução."""
    print("\n=== DEMO: Rastreador de Execução ===")
    
    tracker = PipelineExecutionTracker()
    
    # Simular execução de pipeline
    tracker.start_pipeline()
    
    # Simular etapas
    steps = [
        ("Carregamento de Dados", True),
        ("Pré-processamento", True),
        ("Feature Engineering", False),  # Simular falha
        ("Treinamento", True)
    ]
    
    for step_name, success in steps:
        tracker.start_step(step_name)
        
        # Simular processamento
        import time
        time.sleep(0.1)
        
        if success:
            tracker.end_step(success=True)
        else:
            tracker.end_step(success=False, error="Erro simulado")
    
    tracker.end_pipeline()
    
    # Mostrar resumo
    summary = tracker.get_summary()
    print(f"\nResumo da Execução:")
    print(f"  - Duração total: {summary['total_duration_seconds']:.2f}s")
    print(f"  - Etapas executadas: {summary['total_steps']}")
    print(f"  - Etapas bem-sucedidas: {summary['successful_steps']}")
    print(f"  - Taxa de sucesso: {summary['success_rate']:.1%}")


def demo_reproducibility():
    """Demonstra reprodutibilidade do pipeline."""
    print("\n=== DEMO: Reprodutibilidade ===")
    
    config = load_config("configs/model_config.yaml")
    
    results = []
    
    for run in range(3):
        print(f"\nExecução {run + 1}:")
        
        # Configurar ambiente com mesmo seed
        setup_environment(config)
        
        # Gerar números aleatórios
        import numpy as np
        random_data = np.random.random(5)
        results.append(random_data)
        
        print(f"  Números gerados: {random_data[:3]}")
    
    # Verificar se são idênticos
    all_equal = all(np.array_equal(results[0], result) for result in results[1:])
    
    if all_equal:
        print("\n✓ Reprodutibilidade confirmada - todos os resultados são idênticos")
    else:
        print("\n✗ Problema de reprodutibilidade - resultados diferentes")


def demo_config_validation():
    """Demonstra validação de configuração."""
    print("\n=== DEMO: Validação de Configuração ===")
    
    # Configuração válida
    try:
        config = load_config("configs/model_config.yaml")
        validate_config(config)
        print("✓ Configuração padrão é válida")
    except Exception as e:
        print(f"✗ Erro na configuração padrão: {e}")
    
    # Configuração inválida (sem random_seed)
    try:
        config = load_config("configs/model_config.yaml")
        del config['general']['random_seed']
        validate_config(config)
        print("✗ Validação deveria ter falhado")
    except ValueError as e:
        print(f"✓ Validação detectou erro corretamente: {e}")


def main():
    """Executa todas as demonstrações."""
    print("PIPELINE DE PREVISÃO DE VENDAS - HACKATHON 2025")
    print("=" * 50)
    
    try:
        # Demo 1: Validação de configuração
        demo_config_validation()
        
        # Demo 2: Reprodutibilidade
        demo_reproducibility()
        
        # Demo 3: Rastreador de execução
        demo_execution_tracker()
        
        # Demo 4: Etapas individuais (apenas se dados existirem)
        if Path("data/raw").exists() and any(Path("data/raw").iterdir()):
            demo_pipeline_steps()
        else:
            print("\n=== DEMO: Etapas Individuais ===")
            print("⚠️  Dados não encontrados em data/raw/")
            print("   Para executar este demo, coloque os arquivos Parquet em data/raw/")
        
        print("\n" + "=" * 50)
        print("✓ Todas as demonstrações concluídas com sucesso!")
        print("\nPara executar o pipeline completo:")
        print("  python main.py --step full")
        print("\nPara executar etapas específicas:")
        print("  python main.py --step ingestion")
        print("  python main.py --step preprocessing")
        print("\nPara mais opções:")
        print("  python main.py --help")
        
    except Exception as e:
        print(f"\n✗ Erro durante demonstração: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()