"""
Configuração global para testes pytest.
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

@pytest.fixture(scope="session")
def test_data_dir():
    """Diretório para dados de teste."""
    return Path(__file__).parent / "test_data"

@pytest.fixture(scope="session")
def sample_transactions():
    """Dados de transações de exemplo para testes."""
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', '2022-12-31', freq='D')
    
    data = []
    for i, date in enumerate(dates):
        for pdv in range(1, 6):  # 5 PDVs
            for produto in range(1, 11):  # 10 produtos
                if np.random.random() > 0.3:  # 70% chance de ter venda
                    quantidade = np.random.poisson(10)
                    faturamento = quantidade * np.random.uniform(5, 50)
                    data.append({
                        'data': date,
                        'pdv': pdv,
                        'produto': produto,
                        'quantidade': quantidade,
                        'faturamento': faturamento
                    })
    
    return pd.DataFrame(data)

@pytest.fixture(scope="session")
def sample_products():
    """Dados de produtos de exemplo para testes."""
    return pd.DataFrame({
        'produto': range(1, 11),
        'categoria': ['A', 'B', 'C'] * 3 + ['A'],
        'preco_medio': np.random.uniform(5, 50, 10)
    })

@pytest.fixture(scope="session")
def sample_stores():
    """Dados de PDVs de exemplo para testes."""
    return pd.DataFrame({
        'pdv': range(1, 6),
        'tipo': ['c-store', 'g-store', 'liquor', 'c-store', 'g-store'],
        'zipcode': [12345, 23456, 34567, 45678, 56789],
        'regiao': ['Norte', 'Sul', 'Centro', 'Norte', 'Sul']
    })

@pytest.fixture
def temp_config_file(tmp_path):
    """Arquivo de configuração temporário para testes."""
    config_content = """
model:
  name: "test_model"
  random_state: 42
  
data:
  test_size: 0.2
  validation_weeks: 4
  
features:
  lag_weeks: [1, 2, 4, 8]
  rolling_windows: [4, 8, 12]
  
xgboost:
  n_estimators: 10
  max_depth: 3
  learning_rate: 0.1
  
lightgbm:
  n_estimators: 10
  max_depth: 3
  learning_rate: 0.1
  
ensemble:
  method: "weighted_average"
  weights: [0.4, 0.4, 0.2]
"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config_content)
    return str(config_file)

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup automático para todos os testes."""
    # Configurar seeds para reprodutibilidade
    np.random.seed(42)
    
    # Configurar warnings
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    yield
    
    # Cleanup após testes se necessário
    pass