#!/usr/bin/env python3
"""
Interface de Linha de Comando para Sistema de Submissões - Hackathon 2025
Ferramenta para gerenciar múltiplas submissões de forma eficiente.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import List, Optional
import pandas as pd

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.submission_manager import SubmissionManager
from src.utils.fast_submission_pipeline import FastSubmissionPipeline

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def list_strategies(args):
    """Lista estratégias disponíveis."""
    manager = SubmissionManager(args.config)
    strategies = manager.list_strategies()
    
    print(f"\n=== ESTRATÉGIAS DISPONÍVEIS ({len(strategies)}) ===")
    for i, strategy_name in enumerate(strategies, 1):
        strategy_config = manager.get_strategy_config(strategy_name)
        print(f"{i:2d}. {strategy_name}")
        print(f"    Nome: {strategy_config['name']}")
        print(f"    Descrição: {strategy_config['description']}")
        
        # Mostrar modelos habilitados
        enabled_models = [
            model_type for model_type, config in strategy_config['models'].items()
            if config.get('enabled', True)
        ]
        print(f"    Modelos: {', '.join(enabled_models)}")
        print()


def generate_single(args):
    """Gera submissão para uma estratégia específica."""
    pipeline = FastSubmissionPipeline(args.config)
    
    print(f"\n=== GERANDO SUBMISSÃO: {args.strategy} ===")
    
    try:
        submission = pipeline.generate_single_submission(
            strategy_name=args.strategy,
            data_path=args.data,
            version_type=args.version_type
        )
        
        print(f"✓ Submissão gerada com sucesso!")
        print(f"  Estratégia: {submission.strategy_name}")
        print(f"  Versão: {submission.version}")
        print(f"  Arquivo: {submission.file_path}")
        
        if submission.performance_metrics:
            print(f"  Métricas:")
            for metric, value in submission.performance_metrics.items():
                print(f"    {metric}: {value:.6f}")
        
    except Exception as e:
        print(f"✗ Erro ao gerar submissão: {e}")
        sys.exit(1)


def generate_all(args):
    """Gera submissões para todas as estratégias."""
    pipeline = FastSubmissionPipeline(args.config)
    
    print(f"\n=== GERANDO TODAS AS SUBMISSÕES ===")
    
    try:
        submissions = pipeline.generate_all_submissions(data_path=args.data)
        
        print(f"\n✓ {len(submissions)} submissões geradas!")
        
        # Mostrar resumo
        print(f"\n=== RESUMO DAS SUBMISSÕES ===")
        for submission in submissions:
            print(f"• {submission.strategy_name} {submission.version}")
            if submission.performance_metrics and 'wmape' in submission.performance_metrics:
                wmape = submission.performance_metrics['wmape']
                print(f"  WMAPE: {wmape:.6f}")
        
        # Mostrar melhor submissão
        best_submissions = pipeline.submission_manager.get_best_submission_by_strategy()
        if best_submissions:
            print(f"\n=== MELHORES SUBMISSÕES POR ESTRATÉGIA ===")
            for strategy, submission in best_submissions.items():
                if submission.performance_metrics and 'wmape' in submission.performance_metrics:
                    wmape = submission.performance_metrics['wmape']
                    print(f"• {strategy}: WMAPE {wmape:.6f} ({submission.version})")
        
    except Exception as e:
        print(f"✗ Erro ao gerar submissões: {e}")
        sys.exit(1)


def list_submissions(args):
    """Lista submissões existentes."""
    manager = SubmissionManager(args.config)
    
    submissions = manager.version_manager.list_submissions(args.strategy)
    
    if not submissions:
        print("Nenhuma submissão encontrada.")
        return
    
    strategy_filter = f" para {args.strategy}" if args.strategy else ""
    print(f"\n=== SUBMISSÕES{strategy_filter.upper()} ({len(submissions)}) ===")
    
    for submission in submissions:
        print(f"• {submission.strategy_name} {submission.version}")
        print(f"  Timestamp: {submission.timestamp}")
        print(f"  Config Hash: {submission.config_hash[:16]}...")
        
        if submission.performance_metrics:
            metrics_str = ", ".join([
                f"{k}: {v:.6f}" for k, v in submission.performance_metrics.items()
            ])
            print(f"  Métricas: {metrics_str}")
        
        if submission.file_path:
            print(f"  Arquivo: {submission.file_path}")
        print()


def compare_submissions(args):
    """Compara performance das submissões."""
    manager = SubmissionManager(args.config)
    
    print(f"\n=== COMPARAÇÃO DE PERFORMANCE ===")
    
    # Gerar comparação
    comparison_df = manager.compare_all_submissions(args.strategy)
    
    if comparison_df.empty:
        print("Nenhuma submissão com métricas para comparar.")
        return
    
    # Mostrar tabela de comparação
    print(comparison_df.to_string(index=False))
    
    # Gerar relatório HTML se solicitado
    if args.report:
        report_path = manager.generate_performance_report(args.report)
        print(f"\nRelatório HTML gerado: {report_path}")


def show_best(args):
    """Mostra melhores submissões."""
    manager = SubmissionManager(args.config)
    
    print(f"\n=== MELHORES SUBMISSÕES ===")
    
    best_submissions = manager.get_best_submission_by_strategy()
    
    if not best_submissions:
        print("Nenhuma submissão com métricas encontrada.")
        return
    
    for strategy, submission in best_submissions.items():
        print(f"\n• {strategy}:")
        print(f"  Versão: {submission.version}")
        print(f"  Timestamp: {submission.timestamp}")
        
        if submission.performance_metrics:
            for metric, value in submission.performance_metrics.items():
                print(f"  {metric}: {value:.6f}")
        
        if submission.file_path:
            print(f"  Arquivo: {submission.file_path}")


def cleanup_submissions(args):
    """Remove submissões antigas."""
    manager = SubmissionManager(args.config)
    
    print(f"\n=== LIMPEZA DE SUBMISSÕES ===")
    print(f"Mantendo {args.keep} melhores versões por estratégia...")
    
    if not args.force:
        response = input("Confirma a remoção? (s/N): ")
        if response.lower() != 's':
            print("Operação cancelada.")
            return
    
    try:
        manager.cleanup_old_submissions(keep_versions=args.keep)
        print("✓ Limpeza concluída!")
    except Exception as e:
        print(f"✗ Erro durante limpeza: {e}")
        sys.exit(1)


def cache_stats(args):
    """Mostra estatísticas do cache."""
    pipeline = FastSubmissionPipeline(args.config)
    
    print(f"\n=== ESTATÍSTICAS DO CACHE ===")
    
    stats = pipeline.get_cache_stats()
    
    print(f"Cache de Features:")
    print(f"  Arquivos: {stats['features_cache']['files']}")
    print(f"  Tamanho: {stats['features_cache']['size_mb']:.2f} MB")
    
    print(f"\nCache de Modelos:")
    print(f"  Arquivos: {stats['models_cache']['files']}")
    print(f"  Tamanho: {stats['models_cache']['size_mb']:.2f} MB")
    
    total_size = stats['features_cache']['size_mb'] + stats['models_cache']['size_mb']
    print(f"\nTamanho Total: {total_size:.2f} MB")


def clear_cache(args):
    """Limpa cache."""
    pipeline = FastSubmissionPipeline(args.config)
    
    print(f"\n=== LIMPEZA DE CACHE ===")
    print(f"Tipo: {args.cache_type}")
    
    if not args.force:
        response = input("Confirma a limpeza? (s/N): ")
        if response.lower() != 's':
            print("Operação cancelada.")
            return
    
    try:
        pipeline.clear_cache(args.cache_type)
        print("✓ Cache limpo!")
    except Exception as e:
        print(f"✗ Erro ao limpar cache: {e}")
        sys.exit(1)


def validate_submission_file(args):
    """Valida arquivo de submissão."""
    manager = SubmissionManager(args.config)
    
    print(f"\n=== VALIDAÇÃO DE ARQUIVO ===")
    print(f"Arquivo: {args.file}")
    
    try:
        # Carregar arquivo
        if args.file.endswith('.csv'):
            df = pd.read_csv(args.file, sep=';')
        elif args.file.endswith('.parquet'):
            df = pd.read_parquet(args.file)
        else:
            print("✗ Formato de arquivo não suportado. Use .csv ou .parquet")
            sys.exit(1)
        
        # Validar
        validation = manager.validator.validate_submission(df)
        
        if validation['is_valid']:
            print("✓ Arquivo válido!")
        else:
            print("✗ Arquivo inválido!")
            for error in validation['errors']:
                print(f"  Erro: {error}")
        
        if validation['warnings']:
            print("Avisos:")
            for warning in validation['warnings']:
                print(f"  Aviso: {warning}")
        
        # Mostrar estatísticas
        if validation['stats']:
            print(f"\nEstatísticas:")
            for key, value in validation['stats'].items():
                print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"✗ Erro ao validar arquivo: {e}")
        sys.exit(1)


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description="Sistema de Gerenciamento de Submissões - Hackathon 2025",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python submission_cli.py list-strategies                    # Lista estratégias
  python submission_cli.py generate-single conservative       # Gera submissão específica
  python submission_cli.py generate-all                       # Gera todas as submissões
  python submission_cli.py list-submissions                   # Lista submissões
  python submission_cli.py compare --report report.html       # Compara e gera relatório
  python submission_cli.py show-best                          # Mostra melhores submissões
  python submission_cli.py cleanup --keep 5 --force           # Remove submissões antigas
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/submission_strategies.yaml",
        help="Arquivo de configuração (padrão: configs/submission_strategies.yaml)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Logging detalhado"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Comandos disponíveis")
    
    # Comando: list-strategies
    list_parser = subparsers.add_parser("list-strategies", help="Lista estratégias disponíveis")
    list_parser.set_defaults(func=list_strategies)
    
    # Comando: generate-single
    single_parser = subparsers.add_parser("generate-single", help="Gera submissão para estratégia específica")
    single_parser.add_argument("strategy", help="Nome da estratégia")
    single_parser.add_argument("--data", default="data/processed/weekly_sales_processed.parquet", help="Arquivo de dados")
    single_parser.add_argument("--version-type", choices=["patch", "minor", "major"], default="patch", help="Tipo de versão")
    single_parser.set_defaults(func=generate_single)
    
    # Comando: generate-all
    all_parser = subparsers.add_parser("generate-all", help="Gera submissões para todas as estratégias")
    all_parser.add_argument("--data", default="data/processed/weekly_sales_processed.parquet", help="Arquivo de dados")
    all_parser.set_defaults(func=generate_all)
    
    # Comando: list-submissions
    list_sub_parser = subparsers.add_parser("list-submissions", help="Lista submissões existentes")
    list_sub_parser.add_argument("--strategy", help="Filtrar por estratégia específica")
    list_sub_parser.set_defaults(func=list_submissions)
    
    # Comando: compare
    compare_parser = subparsers.add_parser("compare", help="Compara performance das submissões")
    compare_parser.add_argument("--strategy", help="Filtrar por estratégia específica")
    compare_parser.add_argument("--report", help="Gerar relatório HTML no arquivo especificado")
    compare_parser.set_defaults(func=compare_submissions)
    
    # Comando: show-best
    best_parser = subparsers.add_parser("show-best", help="Mostra melhores submissões por estratégia")
    best_parser.set_defaults(func=show_best)
    
    # Comando: cleanup
    cleanup_parser = subparsers.add_parser("cleanup", help="Remove submissões antigas")
    cleanup_parser.add_argument("--keep", type=int, default=10, help="Número de versões a manter (padrão: 10)")
    cleanup_parser.add_argument("--force", action="store_true", help="Não pedir confirmação")
    cleanup_parser.set_defaults(func=cleanup_submissions)
    
    # Comando: cache-stats
    cache_stats_parser = subparsers.add_parser("cache-stats", help="Mostra estatísticas do cache")
    cache_stats_parser.set_defaults(func=cache_stats)
    
    # Comando: clear-cache
    clear_cache_parser = subparsers.add_parser("clear-cache", help="Limpa cache")
    clear_cache_parser.add_argument("--type", dest="cache_type", choices=["all", "features", "models"], default="all", help="Tipo de cache a limpar")
    clear_cache_parser.add_argument("--force", action="store_true", help="Não pedir confirmação")
    clear_cache_parser.set_defaults(func=clear_cache)
    
    # Comando: validate
    validate_parser = subparsers.add_parser("validate", help="Valida arquivo de submissão")
    validate_parser.add_argument("file", help="Arquivo a validar")
    validate_parser.set_defaults(func=validate_submission_file)
    
    args = parser.parse_args()
    
    # Configurar logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Executar comando
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()