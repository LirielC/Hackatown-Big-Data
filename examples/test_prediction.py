"""
Exemplo de uso do módulo de geração de previsões.

Este script demonstra como usar o PredictionGenerator para gerar
previsões de vendas para janeiro/2023.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import yaml
from datetime import datetime
import logging

from src.models.prediction import PredictionGenerator, PredictionValidator
from src.models.training import XGBoostModel
from src.models.ensemble import WeightedEnsemble

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config():
    """Carrega configuração do modelo."""
    try:
        with open('configs/model_config.yaml', 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning("Arquivo de configuração não encontrado, usando configuração padrão")
        return {
            'prediction': {
                'target_weeks': [1, 2, 3, 4, 5],
                'post_processing': {
                    'ensure_positive': True,
                    'apply_bounds': True,
                    'max_multiplier': 3.0
                },
                'output': {
                    'format': 'csv',
                    'separator': ';',
                    'encoding': 'utf-8'
                }
            }
        }


def create_sample_features():
    """Cria DataFrame de features de exemplo para janeiro/2023."""
    logger.info("Criando features de exemplo para janeiro/2023")
    
    np.random.seed(42)
    
    # Criar combinações PDV/Produto para 5 semanas de janeiro
    pdvs = [f"{i:03d}" for i in range(1, 21)]  # 20 PDVs
    produtos = [f"PROD_{i:03d}" for i in range(1, 51)]  # 50 produtos
    semanas = [1, 2, 3, 4, 5]  # 5 semanas de janeiro/2023
    
    # Criar todas as combinações
    data = []
    for semana in semanas:
        for pdv in pdvs:
            for produto in produtos[:10]:  # Usar apenas 10 produtos por PDV para exemplo
                data.append({
                    'semana': semana,
                    'pdv': pdv,
                    'produto': produto,
                    # Features de lag (simuladas)
                    'lag_1_semana': np.random.normal(15, 5),
                    'lag_2_semanas': np.random.normal(12, 4),
                    'lag_4_semanas': np.random.normal(10, 3),
                    'lag_8_semanas': np.random.normal(8, 2),
                    # Features de média móvel
                    'media_movel_4sem': np.random.normal(12, 3),
                    'media_movel_8sem': np.random.normal(11, 2),
                    'media_movel_12sem': np.random.normal(10, 2),
                    # Features temporais
                    'mes': 1,  # Janeiro
                    'trimestre': 1,
                    'semana_ano': semana,
                    'is_inicio_mes': 1 if semana <= 2 else 0,
                    # Features de sazonalidade
                    'sazonalidade_semanal': np.random.uniform(0.8, 1.2),
                    'sazonalidade_mensal': np.random.uniform(0.9, 1.1),
                    # Features de tendência
                    'tendencia_crescimento': np.random.normal(0.02, 0.01),
                    'volatilidade': np.random.uniform(0.1, 0.3),
                    # Features de produto (simuladas)
                    'categoria_produto_encoded': np.random.randint(1, 6),
                    'preco_medio': np.random.uniform(5.0, 50.0),
                    # Features de PDV (simuladas)
                    'tipo_pdv_encoded': np.random.randint(1, 4),
                    'regiao_encoded': np.random.randint(1, 6),
                    'performance_pdv': np.random.uniform(0.5, 2.0)
                })
    
    df = pd.DataFrame(data)
    logger.info(f"Features criadas: {len(df)} registros, {len(df.columns)} colunas")
    
    return df


def create_sample_historical_data():
    """Cria dados históricos de exemplo para referência."""
    logger.info("Criando dados históricos de exemplo")
    
    np.random.seed(123)
    
    # Usar os mesmos PDVs e produtos das features
    pdvs = [f"{i:03d}" for i in range(1, 21)]
    produtos = [f"PROD_{i:03d}" for i in range(1, 51)]
    
    data = []
    for pdv in pdvs:
        for produto in produtos[:10]:
            # Simular 52 semanas de dados históricos
            for semana in range(1, 53):
                quantidade = max(0, int(np.random.normal(15, 8)))
                data.append({
                    'pdv': pdv,
                    'produto': produto,
                    'semana': semana,
                    'quantidade': quantidade
                })
    
    df = pd.DataFrame(data)
    logger.info(f"Dados históricos criados: {len(df)} registros")
    
    return df


def create_mock_model():
    """Cria um modelo mock para demonstração."""
    logger.info("Criando modelo mock para demonstração")
    
    class MockModel:
        def __init__(self):
            self.is_fitted = True
            np.random.seed(456)
        
        def predict(self, X):
            """Gera previsões simuladas baseadas nas features."""
            # Simular previsões baseadas em algumas features
            base_prediction = 10
            
            if 'lag_1_semana' in X.columns:
                base_prediction += X['lag_1_semana'] * 0.3
            
            if 'media_movel_4sem' in X.columns:
                base_prediction += X['media_movel_4sem'] * 0.2
            
            if 'sazonalidade_semanal' in X.columns:
                base_prediction *= X['sazonalidade_semanal']
            
            # Adicionar ruído
            noise = np.random.normal(0, 2, len(X))
            predictions = base_prediction + noise
            
            return predictions.values
    
    return MockModel()


def demonstrate_prediction_generation():
    """Demonstra o processo completo de geração de previsões."""
    logger.info("=== Demonstração de Geração de Previsões ===")
    
    # 1. Carregar configuração
    config = load_config()
    logger.info("Configuração carregada")
    
    # 2. Criar dados de exemplo
    features_df = create_sample_features()
    historical_data = create_sample_historical_data()
    
    # 3. Criar modelo mock
    model = create_mock_model()
    
    # 4. Inicializar gerador de previsões
    generator = PredictionGenerator(config)
    
    # 5. Gerar previsões
    logger.info("Gerando previsões...")
    predictions = generator.generate_predictions(
        model=model,
        features_df=features_df,
        historical_data=historical_data
    )
    
    # 6. Exibir resultados
    logger.info(f"Previsões geradas com sucesso!")
    logger.info(f"Total de registros: {len(predictions)}")
    logger.info(f"Semanas cobertas: {sorted(predictions['semana'].unique())}")
    logger.info(f"PDVs únicos: {predictions['pdv'].nunique()}")
    logger.info(f"Produtos únicos: {predictions['produto'].nunique()}")
    
    # Estatísticas das quantidades
    qty_stats = predictions['quantidade'].describe()
    logger.info(f"Estatísticas de quantidade:\n{qty_stats}")
    
    # 7. Validar formato
    logger.info("Validando formato das previsões...")
    validation_result = PredictionValidator.validate_prediction_format(predictions)
    
    if validation_result['is_valid']:
        logger.info("✓ Formato das previsões válido")
    else:
        logger.error("✗ Formato das previsões inválido:")
        for error in validation_result['errors']:
            logger.error(f"  - {error}")
    
    # 8. Validar regras de negócio
    logger.info("Validando regras de negócio...")
    business_validation = PredictionValidator.validate_business_rules(predictions)
    
    if business_validation['is_valid']:
        logger.info("✓ Regras de negócio atendidas")
    else:
        logger.warning("⚠ Algumas regras de negócio violadas:")
        for violation in business_validation['violations']:
            logger.warning(f"  - {violation}")
    
    # 9. Gerar resumo
    logger.info("Gerando resumo das previsões...")
    summary = generator.generate_prediction_summary(predictions)
    
    logger.info("=== Resumo das Previsões ===")
    logger.info(f"Total de previsões: {summary['total_predictions']}")
    logger.info(f"Quantidade total prevista: {summary['quantity_statistics']['total_quantity']:,}")
    logger.info(f"Quantidade média por registro: {summary['quantity_statistics']['average_quantity']:.2f}")
    logger.info(f"Previsões zero: {summary['distribution']['zero_predictions']}")
    
    # Top 5 PDVs por volume
    top_pdvs = list(summary['top_pdvs_by_volume'].items())[:5]
    logger.info("Top 5 PDVs por volume previsto:")
    for pdv, volume in top_pdvs:
        logger.info(f"  {pdv}: {volume:,}")
    
    # 10. Salvar previsões
    output_path = f"predictions_example_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    logger.info(f"Salvando previsões em: {output_path}")
    
    saved_path = generator.save_predictions(predictions, output_path, 'csv')
    logger.info(f"Previsões salvas com sucesso: {saved_path}")
    
    return predictions, summary


def demonstrate_comparison_with_baseline():
    """Demonstra comparação com baseline."""
    logger.info("=== Demonstração de Comparação com Baseline ===")
    
    # Criar previsões de exemplo
    predictions = pd.DataFrame({
        'semana': [1, 1, 2, 2, 3, 3],
        'pdv': ['001', '002', '001', '002', '001', '002'],
        'produto': ['A', 'A', 'B', 'B', 'A', 'A'],
        'quantidade': [15, 20, 25, 12, 18, 22]
    })
    
    # Criar baseline (média histórica simples)
    baseline = pd.DataFrame({
        'semana': [1, 1, 2, 2, 3, 3],
        'pdv': ['001', '002', '001', '002', '001', '002'],
        'produto': ['A', 'A', 'B', 'B', 'A', 'A'],
        'quantidade': [12, 18, 20, 10, 15, 20]  # Valores ligeiramente menores
    })
    
    # Comparar
    comparison = PredictionValidator.compare_with_baseline(predictions, baseline)
    
    logger.info("Comparação com baseline:")
    logger.info(f"Total previsto: {comparison['prediction_total']}")
    logger.info(f"Total baseline: {comparison['baseline_total']}")
    logger.info(f"Diferença absoluta: {comparison['difference_absolute']}")
    logger.info(f"Diferença percentual: {comparison['difference_percentage']:.2f}%")


def main():
    """Função principal do exemplo."""
    try:
        # Demonstrar geração de previsões
        predictions, summary = demonstrate_prediction_generation()
        
        print("\n" + "="*60)
        
        # Demonstrar comparação com baseline
        demonstrate_comparison_with_baseline()
        
        print("\n" + "="*60)
        logger.info("Demonstração concluída com sucesso!")
        
        # Exibir amostra das previsões
        print("\nAmostra das previsões geradas:")
        print(predictions.head(10).to_string(index=False))
        
    except Exception as e:
        logger.error(f"Erro durante a demonstração: {str(e)}")
        raise


if __name__ == "__main__":
    main()