#!/usr/bin/env python3
"""
Script Rápido para Stacking Ensemble em Produção
Versão otimizada para execução em minutos, não horas.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import joblib
from pathlib import Path
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_wmape(y_true, y_pred):
    """Calcula WMAPE."""
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))

def run_fast_stacking_production():
    """
    Executa stacking ensemble otimizado para produção.
    """
    print("🚀 STACKING ENSEMBLE - VERSÃO PRODUÇÃO OTIMIZADA")
    print("=" * 60)

    start_time = time.time()

    try:
        print("📊 Carregando dados de produção...")

        data_path = "data/processed/features_engineered.parquet"

        if not Path(data_path).exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {data_path}")

        print("🔄 Carregando amostra de dados (teste rápido)...")
        df_full = pd.read_parquet(data_path)

        sample_size = int(len(df_full) * 0.2)
        df = df_full.sample(n=sample_size, random_state=42)

        print(f"✅ Dados carregados: {len(df):,} registros (amostra de {len(df_full):,} total)")

        target_col = 'quantidade'
        exclude_cols = [target_col, 'pdv', 'produto', 'data_semana', 'data']

        feature_cols = [col for col in df.columns
                       if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]

        X = df[feature_cols].fillna(0)
        y = df[target_col]

        print(f"📈 Features: {len(feature_cols)}, Target: {target_col}")

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        print(f"🎯 Treino: {len(X_train):,}, Teste: {len(X_test):,}")

        print("🏗️ Criando modelos base otimizados...")

        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression
        import xgboost as xgb
        import lightgbm as lgb

        base_models = [
            ('rf', RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42)),
            ('lr', LinearRegression()),
            ('xgb', xgb.XGBRegressor(n_estimators=50, max_depth=4, learning_rate=0.1,
                                   random_state=42, n_jobs=-1)),
            ('lgb', lgb.LGBMRegressor(n_estimators=50, max_depth=4, learning_rate=0.1,
                                   random_state=42, verbose=-1))
        ]

        print("🎯 Treinando Stacking Ensemble...")

        stacking = StackingRegressor(
            estimators=base_models,
            final_estimator=Ridge(alpha=1.0, random_state=42),
            cv=3,
            n_jobs=-1
        )

        print("🔄 Treinando modelos base individualmente...")
        trained_models = {}
        for name, model in base_models:
            print(f"  Treinando {name}...")
            model.fit(X_train, y_train)
            trained_models[name] = model

        print("🔄 Treinando Stacking Ensemble...")
        base_models_for_stacking = [(name, model) for name, model in trained_models.items()]
        stacking = StackingRegressor(
            estimators=base_models_for_stacking,
            final_estimator=Ridge(alpha=1.0, random_state=42),
            cv=3,
            n_jobs=-1,
            verbose=1
        )
        stacking.fit(X_train, y_train)

        print("🔮 Gerando previsões...")

        y_pred_stacking = stacking.predict(X_test)

        y_pred_individual = {}
        for name, model in trained_models.items():
            y_pred_individual[name] = model.predict(X_test)

        y_pred_simple = np.mean([y_pred_individual[name] for name, _ in base_models], axis=0)

        print("📊 Calculando métricas...")

        results = {}

        # Stacking
        stacking_wmape = calculate_wmape(y_test, y_pred_stacking)
        stacking_mae = mean_absolute_error(y_test, y_pred_stacking)

        results['stacking'] = {
            'wmape': stacking_wmape,
            'mae': stacking_mae
        }

        # Ensemble simples
        simple_wmape = calculate_wmape(y_test, y_pred_simple)
        simple_mae = mean_absolute_error(y_test, y_pred_simple)

        results['simple_ensemble'] = {
            'wmape': simple_wmape,
            'mae': simple_mae
        }

        # Modelos individuais
        for name, y_pred in y_pred_individual.items():
            wmape = calculate_wmape(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            results[name] = {'wmape': wmape, 'mae': mae}

        # 7. Exibir resultados
        print("\n" + "=" * 60)
        print("📊 RESULTADOS FINAIS - STACKING ENSEMBLE PRODUÇÃO")
        print("=" * 60)

        print("🏆 Comparação de Performance:")
        print("-" * 40)

        for model_name, metrics in results.items():
            status = "🏆" if model_name == 'stacking' else "  "
            print(f"{model_name:15s} WMAPE: {metrics['wmape']:.4f}")

        # Análise da melhoria
        improvement = simple_wmape - stacking_wmape
        baseline_wmape = 0.681  # Baseline do projeto

        print("\n🎯 ANÁLISE DA TAREFA 1.3:")
        print(f"📊 Baseline WMAPE: {baseline_wmape:.1%}")
        print(f"⚖️  Ensemble Simples: {simple_wmape:.4f}")
        print(f"🏗️  Stacking: {stacking_wmape:.4f}")
        print(f"📈 Melhoria vs Simples: {improvement:.1%}")
        print(f"🎯 Meta (>10%): {'✅ ATINGIDA' if improvement > 0.10 else '❌ NÃO ATINGIDA'}")

        # 8. Salvar modelo
        print("\n💾 Salvando modelo de produção...")
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)

        model_data = {
            'stacking_model': stacking,
            'feature_columns': feature_cols,
            'results': results,
            'improvement': improvement,
            'timestamp': pd.Timestamp.now().isoformat(),
            'version': 'production_v1.0'
        }

        model_path = models_dir / "stacking_production_model.joblib"
        joblib.dump(model_data, model_path)

        print(f"✅ Modelo salvo: {model_path}")

        # 9. Tempo total
        total_time = time.time() - start_time
        print(f"⏱️  Tempo total: {total_time:.2f}s")
        # 10. Conclusão
        if improvement > 0.10:
            print("\n🎉 SUCESSO! Stacking Ensemble superou a meta da Tarefa 1.3!")
            print("🏆 Implementação pronta para produção!")
        else:
            print("\n⚠️  Meta não atingida, mas stacking implementado com sucesso!")

        return results

    except Exception as e:
        logger.error(f"❌ Erro: {e}")
        raise

if __name__ == "__main__":
    results = run_fast_stacking_production()
