"""
Exemplo de uso do módulo de validação e avaliação de modelos.
Demonstra validação walk-forward, análise de resíduos e comparação com baselines.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import yaml
import logging
from datetime import datetime
import matplotlib.pyplot as plt

from src.models.training import XGBoostModel, ModelEvaluator
from src.models.validation import ValidationManager

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_data():
    """Cria dados de exemplo para demonstração."""
    logger.info("Criando dados de exemplo")
    
    np.random.seed(42)
    n_samples = 200
    
    # Criar série temporal com tendência e sazonalidade
    dates = pd.date_range('2022-01-01', periods=n_samples, freq='W')
    
    # Componentes da série
    trend = np.linspace(100, 200, n_samples)
    seasonal = 20 * np.sin(2 * np.pi * np.arange(n_samples) / 52)  # Sazonalidade anual
    weekly_seasonal = 5 * np.sin(2 * np.pi * np.arange(n_samples) / 4)  # Sazonalidade mensal
    
    # Features exógenas
    promo = np.random.binomial(1, 0.2, n_samples)  # 20% de promoções
    temperatura = 25 + 10 * np.sin(2 * np.pi * np.arange(n_samples) / 52) + np.random.normal(0, 2, n_samples)
    concorrencia = np.random.normal(50, 10, n_samples)
    
    # Segmentos
    segmentos = np.random.choice(['Eletrônicos', 'Roupas', 'Casa'], n_samples, p=[0.4, 0.35, 0.25])
    pdvs = np.random.choice(['PDV_A', 'PDV_B', 'PDV_C', 'PDV_D'], n_samples)
    
    # Target com dependências das features
    noise = np.random.normal(0, 8, n_samples)
    vendas = (trend + seasonal + weekly_seasonal + 
              promo * 15 +  # Efeito promoção
              (temperatura - 25) * 0.5 +  # Efeito temperatura
              concorrencia * 0.1 +  # Efeito concorrência
              noise)
    
    # Adicionar efeitos por segmento
    segment_effects = {'Eletrônicos': 1.2, 'Roupas': 1.0, 'Casa': 0.8}
    for i, seg in enumerate(segmentos):
        vendas[i] *= segment_effects[seg]
    
    # Criar DataFrame
    data = pd.DataFrame({
        'data': dates,
        'vendas': vendas,
        'promo': promo,
        'temperatura': temperatura,
        'concorrencia': concorrencia,
        'segmento': segmentos,
        'pdv': pdvs,
        'semana': dates.isocalendar().week,
        'mes': dates.month,
        'trimestre': dates.quarter
    })
    
    logger.info(f"Dados criados: {len(data)} observações, {len(data.columns)} colunas")
    return data


def load_config():
    """Carrega configuração para o exemplo."""
    config = {
        'general': {
            'n_jobs': -1,
            'random_state': 42
        },
        'models': {
            'xgboost': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'early_stopping_rounds': 20
            }
        },
        'validation': {
            'walk_forward': {
                'initial_train_size': 0.6,
                'step_size': 8,  # 8 semanas
                'min_train_size': 52  # 1 ano mínimo
            },
            'n_splits': 5,
            'test_size': 8
        },
        'baseline': {
            'moving_average_window': 4,
            'seasonal_period': 52
        },
        'preprocessing': {
            'handle_missing': True,
            'create_lags': True,
            'lag_periods': [1, 2, 4, 8, 12],
            'create_rolling_features': True,
            'rolling_windows': [4, 8, 12],
            'create_date_features': True
        },
        'visualization': {
            'save_plots': True,
            'plot_format': 'png',
            'dpi': 300
        }
    }
    
    return config


def prepare_features(data, config):
    """Prepara features para o modelo."""
    logger.info("Preparando features")
    
    # Separar features e target
    feature_columns = ['promo', 'temperatura', 'concorrencia', 'semana', 'mes', 'trimestre']
    X_base = data[['data'] + feature_columns + ['segmento', 'pdv']].copy()
    y = data['vendas'].copy()
    
    # Encoding simples de variáveis categóricas
    X_processed = pd.get_dummies(X_base, columns=['segmento', 'pdv'], prefix=['seg', 'pdv'])
    
    # Remover colunas não numéricas para o modelo (exceto data e segmento original)
    numeric_columns = X_processed.select_dtypes(include=[np.number]).columns
    X_final = X_processed[numeric_columns].copy()
    
    # Manter coluna de data e segmento original para análises
    X_final['data'] = X_base['data']
    X_final['segmento'] = X_base['segmento']
    
    logger.info(f"Features preparadas: {X_final.shape[1]} colunas")
    return X_final, y


def train_model(X, y, config):
    """Treina modelo XGBoost."""
    logger.info("Treinando modelo XGBoost")
    
    # Separar features numéricas para treino
    feature_columns = X.select_dtypes(include=[np.number]).columns
    feature_columns = [col for col in feature_columns if col not in ['data']]
    
    X_train = X[feature_columns]
    
    # Dividir em treino e validação
    split_idx = int(len(X_train) * 0.8)
    X_train_split = X_train.iloc[:split_idx]
    X_val_split = X_train.iloc[split_idx:]
    y_train_split = y.iloc[:split_idx]
    y_val_split = y.iloc[split_idx:]
    
    # Treinar modelo
    model = XGBoostModel(config)
    model.fit(X_train_split, y_train_split, X_val_split, y_val_split, optimize_hyperparams=False)
    
    logger.info("Modelo treinado com sucesso")
    return model


def run_validation_example():
    """Executa exemplo completo de validação."""
    logger.info("=== INICIANDO EXEMPLO DE VALIDAÇÃO ===")
    
    # 1. Preparar dados
    data = create_sample_data()
    config = load_config()
    
    # 2. Preparar features
    X, y = prepare_features(data, config)
    
    # 3. Treinar modelo
    model = train_model(X, y, config)
    
    # 4. Executar validação completa
    logger.info("Executando validação completa")
    
    validation_manager = ValidationManager(config)
    
    # Criar diretório para salvar resultados
    results_dir = "validation_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Executar validação
    validation_results = validation_manager.run_complete_validation(
        model=model,
        X=X,
        y=y,
        date_column='data',
        segment_column='segmento',
        save_path=results_dir
    )
    
    # 5. Exibir resultados
    print("\n" + "="*60)
    print("RESUMO DOS RESULTADOS DE VALIDAÇÃO")
    print("="*60)
    
    # Performance do modelo
    performance = validation_results['executive_summary']['model_performance']
    print(f"\n📊 PERFORMANCE DO MODELO:")
    print(f"   WMAPE: {performance['wmape']}")
    print(f"   MAE: {performance['mae']}")
    print(f"   RMSE: {performance['rmse']}")
    print(f"   Estabilidade: {performance['stability']}")
    
    # Qualidade da validação
    quality = validation_results['executive_summary']['validation_quality']
    print(f"\n🔍 QUALIDADE DA VALIDAÇÃO:")
    print(f"   Viés dos resíduos: {quality['residual_bias']}")
    print(f"   Taxa de outliers: {quality['outlier_rate']}")
    print(f"   Resíduos normais: {'✓' if quality['normality_ok'] else '✗'}")
    
    # Comparação com baselines
    baseline = validation_results['executive_summary']['baseline_comparison']
    print(f"\n📈 COMPARAÇÃO COM BASELINES:")
    print(f"   Melhor baseline: {baseline['best_baseline']}")
    print(f"   Ranking do modelo: {baseline['model_rank']}")
    print(f"   Melhor melhoria: {baseline['best_improvement']}")
    
    # Recomendações
    recommendations = validation_results['executive_summary']['recommendations']
    print(f"\n💡 RECOMENDAÇÕES:")
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
    else:
        print("   ✓ Modelo apresenta boa performance geral")
    
    # Detalhes da validação walk-forward
    print(f"\n📅 DETALHES DA VALIDAÇÃO WALK-FORWARD:")
    fold_results = validation_results['validation']['metrics_by_fold']
    print(f"   Número de folds: {len(fold_results)}")
    
    wmape_values = [fold['wmape'] for fold in fold_results]
    print(f"   WMAPE médio: {np.mean(wmape_values):.2f}%")
    print(f"   WMAPE desvio: {np.std(wmape_values):.2f}%")
    print(f"   WMAPE min/max: {np.min(wmape_values):.2f}% / {np.max(wmape_values):.2f}%")
    
    # Análise por segmento
    if 'segment_analysis' in validation_results['residual_analysis']:
        print(f"\n🎯 ANÁLISE POR SEGMENTO:")
        segment_analysis = validation_results['residual_analysis']['segment_analysis']
        for segment, metrics in segment_analysis.items():
            wmape_seg = metrics['metrics']['wmape']
            count_seg = metrics['count']
            print(f"   {segment}: WMAPE {wmape_seg:.2f}% ({count_seg} obs)")
    
    print(f"\n📁 Resultados salvos em: {results_dir}/")
    if 'dashboard' in validation_results:
        print(f"   Gráficos criados: {', '.join(validation_results['dashboard']['plots_created'])}")
    else:
        print("   Dashboard de visualização não implementado na versão simplificada")
    
    print("\n" + "="*60)
    print("VALIDAÇÃO CONCLUÍDA COM SUCESSO!")
    print("="*60)
    
    return validation_results


def demonstrate_individual_components():
    """Demonstra uso individual dos componentes de validação."""
    logger.info("=== DEMONSTRANDO COMPONENTES INDIVIDUAIS ===")
    
    # Preparar dados
    data = create_sample_data()
    config = load_config()
    X, y = prepare_features(data, config)
    model = train_model(X, y, config)
    
    print("\n1. VALIDAÇÃO WALK-FORWARD")
    print("-" * 40)
    
    from src.models.validation import WalkForwardValidator
    
    validator = WalkForwardValidator(config)
    wf_results = validator.validate_model(model, X, y, 'data')
    
    print(f"Folds executados: {len(wf_results.get('fold_results', wf_results.get('metrics_by_fold', [])))}")
    print(f"WMAPE geral: {wf_results['overall_metrics']['wmape']:.2f}%")
    print(f"MAE geral: {wf_results['overall_metrics']['mae']:.2f}")
    
    print("\n2. ANÁLISE DE RESÍDUOS")
    print("-" * 40)
    
    from src.models.validation import ResidualAnalyzer
    
    # Gerar previsões para análise
    feature_cols = X.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in feature_cols if col not in ['data']]
    y_pred = model.predict(X[feature_cols])
    
    analyzer = ResidualAnalyzer(config)
    residual_results = analyzer.analyze_residuals(y.values, y_pred, X, 'segmento')
    
    print(f"Média dos resíduos: {residual_results['basic_stats']['mean']:.4f}")
    print(f"Desvio dos resíduos: {residual_results['basic_stats']['std']:.2f}")
    print(f"Outliers detectados: {residual_results['outliers']['z_score_outliers']['percentage']:.1f}%")
    
    print("\n3. COMPARAÇÃO COM BASELINES")
    print("-" * 40)
    
    from src.models.validation import BaselineComparator
    
    comparator = BaselineComparator(config)
    baselines = comparator.create_baselines(X, y, 'segmento')
    
    print("Baselines criados:")
    for baseline_name, baseline_info in baselines.items():
        if baseline_name != 'segment_baselines':
            print(f"  - {baseline_name}: {baseline_info.get('description', 'N/A')}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    try:
        # Executar exemplo completo
        validation_results = run_validation_example()
        
        # Demonstrar componentes individuais
        demonstrate_individual_components()
        
        print("\n✅ Exemplo executado com sucesso!")
        
    except Exception as e:
        logger.error(f"Erro durante execução: {str(e)}")
        raise