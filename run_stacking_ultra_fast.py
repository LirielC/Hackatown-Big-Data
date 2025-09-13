#!/usr/bin/env python3
"""
Script ULTRA RÁPIDO para Stacking Ensemble
Versão otimizada para execução em SEGUNDOS.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.tree import DecisionTreeRegressor
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

def run_ultra_fast_stacking():
    """
    Executa stacking ensemble ultra-rápido para demonstração.
    """
    print("⚡ STACKING ENSEMBLE - VERSÃO ULTRA RÁPIDA")
    print("=" * 50)

    start_time = time.time()

    try:
        print("📊 Carregando dados mínimos...")

        data_path = "data/processed/features_engineered.parquet"

        if not Path(data_path).exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {data_path}")

        print("🔄 Carregando 1% dos dados (ultra-rápido)...")
        df_full = pd.read_parquet(data_path)

        sample_size = int(len(df_full) * 0.01)
        df = df_full.sample(n=sample_size, random_state=42)

        print(f"✅ Dados carregados: {len(df):,} registros (1% de {len(df_full):,} total)")

        target_col = 'quantidade'
        exclude_cols = [target_col, 'pdv', 'produto', 'data_semana', 'data']

        all_feature_cols = [col for col in df.columns
                           if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]

        feature_cols = all_feature_cols[:10]

        X = df[feature_cols].fillna(0)
        y = df[target_col]

        print(f"📈 Features: {len(feature_cols)}, Target: {target_col}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        print(f"🎯 Treino: {len(X_train):,}, Teste: {len(X_test):,}")

        print("🏗️ Criando modelos base ultra-simples...")

        dt1 = DecisionTreeRegressor(max_depth=3, random_state=42)
        dt2 = DecisionTreeRegressor(max_depth=4, random_state=43)
        lr = LinearRegression()

        print("🔄 Treinando modelos base...")
        dt1.fit(X_train, y_train)
        dt2.fit(X_train, y_train)
        lr.fit(X_train, y_train)

        base_models = [
            ('dt1', dt1),
            ('dt2', dt2),
            ('lr', lr)
        ]

        print("🎯 Treinando Stacking Ensemble...")

        stacking = StackingRegressor(
            estimators=base_models,
            final_estimator=Ridge(alpha=1.0, random_state=42),
            cv=2,
            n_jobs=1
        )

        print("🔄 Ajustando modelo...")
        stacking.fit(X_train, y_train)

        print("🔮 Gerando previsões...")

        y_pred_stacking = stacking.predict(X_test)

        y_pred_dt1 = base_models[0][1].predict(X_test)
        y_pred_dt2 = base_models[1][1].predict(X_test)
        y_pred_lr = base_models[2][1].predict(X_test)

        y_pred_simple = (y_pred_dt1 + y_pred_dt2 + y_pred_lr) / 3

        print("📊 Calculando métricas...")

        stacking_wmape = calculate_wmape(y_test, y_pred_stacking)
        simple_wmape = calculate_wmape(y_test, y_pred_simple)

        improvement = simple_wmape - stacking_wmape

        # 7. Exibir resultados
        print("\n" + "=" * 50)
        print("📊 RESULTADOS ULTRA-RÁPIDOS")
        print("=" * 50)

        print("🏆 Modelos:")
        print(f"  Decision Tree 1:  WMAPE {calculate_wmape(y_test, y_pred_dt1):.4f}")
        print(f"  Decision Tree 2:  WMAPE {calculate_wmape(y_test, y_pred_dt2):.4f}")
        print(f"  Linear Regression: WMAPE {calculate_wmape(y_test, y_pred_lr):.4f}")
        print(f"  Simple Ensemble:  WMAPE {simple_wmape:.4f}")
        print(f"  🏆 STACKING:       WMAPE {stacking_wmape:.4f}")

        print("\n🎯 ANÁLISE:")
        print(f"⚖️  Ensemble Simples: {simple_wmape:.4f}")
        print(f"🏗️  Stacking: {stacking_wmape:.4f}")
        print(f"📈 Melhoria: {improvement:.1%}")
        print(f"🎯 Meta (>10%): {'✅ ATINGIDA' if improvement > 0.10 else '❌ NÃO ATINGIDA'}")

        # Tempo total
        total_time = time.time() - start_time
        print(f"⏱️  Tempo total: {total_time:.2f}s")
        # Salvar modelo simples
        print("\n💾 Salvando modelo...")
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)

        model_data = {
            'stacking_model': stacking,
            'feature_columns': feature_cols,
            'wmape': stacking_wmape,
            'improvement': improvement,
            'timestamp': pd.Timestamp.now().isoformat()
        }

        model_path = models_dir / "stacking_ultra_fast_model.joblib"
        joblib.dump(model_data, model_path)

        print(f"✅ Modelo salvo: {model_path}")

        # Conclusão
        if improvement > 0.10:
            print("\n🎉 SUCESSO! Stacking Ensemble demonstrou conceito avançado!")
        else:
            print("\n⚡ Conceito demonstrado - Stacking implementado com sucesso!")

        return {
            'stacking_wmape': stacking_wmape,
            'simple_wmape': simple_wmape,
            'improvement': improvement,
            'time_seconds': total_time
        }

    except Exception as e:
        logger.error(f"❌ Erro: {e}")
        raise

if __name__ == "__main__":
    results = run_ultra_fast_stacking()
    print(f"\n📈 Resumo: Stacking WMAPE {results['stacking_wmape']:.4f}, "
          f"Melhoria {results['improvement']:.1%}, "
          f"Tempo {results['time_seconds']:.1f}s")
