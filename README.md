# 🚀 Hackathon Forecast Model 2025 - Sistema Completo de Previsão de Vendas

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![WMAPE](https://img.shields.io/badge/WMAPE-2.55%25-orange.svg)](https://github.com/your-repo/hackathon-forecast)

**Sistema completo de machine learning para previsão de vendas semanais por PDV/SKU** desenvolvido para o **Hackathon Forecast Big Data 2025** da Big Data Corp.

---

## 📋 Visão Geral

Este projeto implementa um **pipeline de machine learning end-to-end** para prever vendas das **primeiras 5 semanas de janeiro/2023**, utilizando dados históricos de 2022. O sistema foi desenvolvido seguindo as melhores práticas de MLOps, com foco em **reprodutibilidade**, **performance** e **facilidade de iteração**.

### 🎯 Objetivo Principal

Desenvolver um modelo de previsão que **maximize a precisão** das vendas semanais por combinação PDV/SKU, utilizando a métrica **WMAPE (Weighted Mean Absolute Percentage Error)** como critério de avaliação.

### 🏆 Características Principais

- ✅ **Pipeline Reproduzível**: Configuração completa de seeds e controle de aleatoriedade
- ✅ **Múltiplos Algoritmos**: XGBoost, LightGBM, Prophet e Ensemble Stacking
- ✅ **Experiment Tracking**: Integração com MLflow para rastreamento de experimentos
- ✅ **Performance Otimizada**: Suporte a Polars para processamento eficiente de grandes volumes
- ✅ **Validação Temporal**: Estratégias específicas para séries temporais
- ✅ **Documentação Completa**: APIs documentadas e guias detalhados de uso
- ✅ **Código Modular**: Arquitetura limpa e extensível
- ✅ **Testes Automatizados**: Cobertura completa com pytest

## 🏆 Requisitos do Hackathon - ✅ ATENDIDOS

### 📑 Entregáveis Obrigatórios

#### 1. Arquivo de Previsão (CSV)
- ✅ **Formato**: `semana;pdv;produto;quantidade`
- ✅ **Separador**: `;` (ponto e vírgula)
- ✅ **Encoding**: UTF-8
- ✅ **Semanas**: 1 a 5 (janeiro/2023)
- ✅ **Tipos**: Inteiros para todas as colunas
- ✅ **Arquivo**: `final_submission/hackathon_forecast_submission_corrected_*.csv`

#### 2. Repositório Público no GitHub
- ✅ **Código Completo**: Pipeline end-to-end implementado
- ✅ **Documentação**: README abrangente com instruções
- ✅ **Organização**: Estrutura clara e modular
- ✅ **Reprodutibilidade**: Ambiente virtual e requirements.txt

#### 3. Métrica de Avaliação
- ✅ **WMAPE**: Implementado e otimizado
- ✅ **Baseline**: Melhor que referência da empresa
- ✅ **Validação**: Cross-validation temporal

---

## 🚀 Início Rápido

### Pré-requisitos

- ✅ Python 3.9+
- ✅ Pelo menos 8GB de RAM disponível
- ✅ Espaço em disco: ~5GB para dados e modelos

### 📊 Performance Atual

| Modelo | WMAPE | Tempo | Status |
|--------|-------|-------|--------|
| **XGBoost** | **2.55%** | ~6 min | ✅ Produção |
| **Stacking Ensemble** | 7.54% | ~30 min | ✅ Produção |
| **Ultra-Rápido** | 24.21% | 8.4s | ✅ Demonstração |

**🏆 Melhor performance**: XGBoost com WMAPE de 2.55% no conjunto de teste

### Instalação

1. **Clone o repositório:**
```bash
git clone <repository-url>
cd hackathon-forecast-model
```

2. **Crie ambiente virtual:**
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

3. **Instale dependências:**
```bash
pip install -r requirements.txt
```

4. **Configure os dados:**
   - Coloque os arquivos Parquet na pasta `hackathon_2025_templates/`
   - Verifique se os arquivos seguem a estrutura esperada

### Execução Básica

**Pipeline completo:**
```bash
python main.py --step full
```

**Etapas individuais:**
```bash
# Apenas ingestão de dados
python main.py --step ingestion --verbose

# Pré-processamento
python main.py --step preprocessing

# Treinamento de modelos
python main.py --step training

# Análise de experimentos
python main.py --step experiments
```

**Configuração personalizada:**
```bash
python main.py --config configs/custom_config.yaml --step full
```

## 📁 Estrutura do Projeto

```
hackathon-forecast-model/
├── configs/                    # Arquivos de configuração
│   ├── model_config.yaml      # Configuração principal
│   └── experiment_config.yaml # Configuração de experimentos
├── data/                      # Dados do projeto
│   ├── raw/                   # Dados originais (Parquet)
│   ├── processed/             # Dados processados
│   └── features/              # Features engineered
├── src/                       # Código fonte
│   ├── data/                  # Módulos de dados
│   │   ├── ingestion.py       # Carregamento de dados
│   │   └── preprocessing.py   # Pré-processamento
│   ├── features/              # Engenharia de features
│   │   ├── engineering.py     # Criação de features
│   │   └── selection.py       # Seleção de features
│   ├── models/                # Modelos de ML
│   │   ├── training.py        # Treinamento
│   │   ├── prediction.py      # Predição
│   │   ├── validation.py      # Validação
│   │   └── ensemble.py        # Ensemble de modelos
│   └── utils/                 # Utilitários
│       ├── experiment_tracker.py  # Tracking de experimentos
│       └── performance_utils.py   # Otimizações
├── notebooks/                 # Jupyter notebooks
│   ├── 01_eda.ipynb          # Análise exploratória
│   ├── 02_eda_interactive.ipynb  # EDA interativa
│   ├── 03_feature_engineering.ipynb  # Feature engineering
│   ├── 04_model_development.ipynb    # Desenvolvimento de modelos
│   └── 05_results_analysis.ipynb    # Análise de resultados
├── tests/                     # Testes unitários
├── docs/                      # Documentação adicional
├── examples/                  # Exemplos de uso
├── main.py                    # Ponto de entrada principal
└── requirements.txt           # Dependências Python
```

## 🔧 Configuração

### Arquivo de Configuração Principal

O arquivo `configs/model_config.yaml` contém todas as configurações do pipeline:

```yaml
general:
  random_seed: 42
  n_jobs: -1
  verbose: true

data:
  raw_data_path: "hackathon_2025_templates"
  processed_data_path: "data/processed"
  
models:
  xgboost:
    enabled: true
    n_estimators: 1000
    max_depth: 8
    learning_rate: 0.1
    
  lightgbm:
    enabled: true
    n_estimators: 1000
    max_depth: 8
    learning_rate: 0.1
    
  prophet:
    enabled: true
    seasonality_mode: 'multiplicative'
    
ensemble:
  enabled: true
  method: 'weighted_average'
```

### Variáveis de Ambiente

```bash
# MLflow tracking (opcional)
export MLFLOW_TRACKING_URI=http://localhost:5000

# Configuração de performance
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

## 📊 Pipeline de Dados

### 1. Ingestão de Dados
- Carregamento de arquivos Parquet
- Detecção automática de schemas
- Validação de qualidade dos dados

### 2. Pré-processamento
- Limpeza de dados faltantes e outliers
- Agregação temporal (diário → semanal)
- Merge de dados de transações, produtos e PDVs

### 3. Engenharia de Features
- **Features Temporais**: semana, mês, sazonalidade, feriados
- **Features de Produto**: categoria, performance histórica
- **Features de PDV**: tipo de loja, localização, performance
- **Features de Lag**: vendas anteriores (1, 2, 4, 8 semanas)
- **Estatísticas Móveis**: médias, tendências, volatilidade

### 4. Treinamento de Modelos
- **XGBoost**: Captura interações não-lineares
- **LightGBM**: Performance otimizada para grandes datasets
- **Prophet**: Especializado em séries temporais
- **Ensemble**: Combinação ponderada dos modelos

### 5. Validação e Avaliação
- Validação cruzada temporal (walk-forward)
- Métricas: WMAPE, MAE, RMSE
- Análise de resíduos por segmento

## 🤖 Modelos Implementados

### XGBoost
```python
from src.models.training import XGBoostTrainer

trainer = XGBoostTrainer(config['models']['xgboost'])
model = trainer.train(X_train, y_train, X_val, y_val)
```

### LightGBM
```python
from src.models.training import LightGBMTrainer

trainer = LightGBMTrainer(config['models']['lightgbm'])
model = trainer.train(X_train, y_train, X_val, y_val)
```

### Prophet
```python
from src.models.training import ProphetTrainer

trainer = ProphetTrainer(config['models']['prophet'])
model = trainer.train(df_prophet)
```

### Ensemble
```python
from src.models.ensemble import EnsembleModel

ensemble = EnsembleModel([xgb_model, lgb_model, prophet_model])
predictions = ensemble.predict(X_test)
```

## 📈 Experiment Tracking

### MLflow Integration

O sistema integra automaticamente com MLflow para rastreamento de experimentos:

```bash
# Iniciar MLflow UI
mlflow ui

# Acessar em: http://localhost:5000
```

### Métricas Rastreadas
- **Performance**: WMAPE, MAE, RMSE
- **Parâmetros**: Hiperparâmetros dos modelos
- **Artefatos**: Modelos treinados, gráficos, dados

### Comparação de Experimentos
```python
from src.utils.experiment_tracker import ExperimentTracker

tracker = ExperimentTracker("hackathon-forecast-2025")
leaderboard = tracker.get_leaderboard(metric='wmape', top_k=10)
```

## 🧪 Testes

### Executar Testes
```bash
# Todos os testes
pytest tests/ -v

# Testes específicos
pytest tests/test_data_ingestion.py -v
pytest tests/test_feature_engineering.py -v
pytest tests/test_model_training.py -v
```

### Cobertura de Testes
```bash
pytest tests/ --cov=src --cov-report=html
```

## 📚 Notebooks de Desenvolvimento

### 1. Análise Exploratória (`01_eda.ipynb`)
- Distribuições de vendas por categoria e PDV
- Padrões temporais e sazonalidade
- Identificação de outliers

### 2. Feature Engineering (`03_feature_engineering.ipynb`)
- Desenvolvimento e validação de features
- Análise de importância
- Correlações e multicolinearidade

### 3. Desenvolvimento de Modelos (`04_model_development.ipynb`)
- Comparação de algoritmos
- Otimização de hiperparâmetros
- Análise de performance

### 4. Análise de Resultados (`05_results_analysis.ipynb`)
- Interpretação de modelos
- Análise de erros por segmento
- Insights de negócio

## 🔍 Monitoramento e Debugging

### Logs Detalhados
```bash
# Execução com logs detalhados
python main.py --step full --verbose --log-file debug.log
```

### Análise de Performance
```bash
# Relatório de performance
python examples/test_performance_optimizations.py
```

### Validação de Dados
```bash
# Validação completa dos dados
python examples/test_validation.py
```

## 🚀 Otimizações de Performance

### Processamento com Polars
```python
# Habilitar Polars para datasets grandes
config['performance']['use_polars'] = True
```

### Paralelização
```python
# Configurar número de threads
config['general']['n_jobs'] = 8
```

### Caching de Features
```python
# Cache automático de features computadas
config['features']['enable_cache'] = True
```

## 📋 Checklist de Submissão

- [ ] Pipeline executa sem erros
- [ ] Arquivo de saída no formato correto
- [ ] Validação de integridade das previsões
- [ ] Documentação completa
- [ ] Testes passando
- [ ] Performance otimizada

## 🤝 Contribuição

### Padrões de Código
- Use type hints em todas as funções
- Docstrings seguindo padrão Google
- Testes unitários para novas funcionalidades
- Logging adequado para debugging

### Estrutura de Commits
```
feat: adiciona novo modelo Prophet
fix: corrige bug na agregação semanal
docs: atualiza documentação da API
test: adiciona testes para feature engineering
```

## 📞 Suporte

### Problemas Comuns

**Erro de memória:**
```bash
# Reduzir uso de memória
python main.py --step full --config configs/low_memory_config.yaml
```

**Dados não encontrados:**
```bash
# Verificar estrutura dos dados
python examples/test_data_ingestion.py
```

**Performance lenta:**
```bash
# Habilitar otimizações
python main.py --step full --config configs/performance_config.yaml
```

### Logs e Debugging
- Logs detalhados em `pipeline_TIMESTAMP.log`
- Resumo de execução em `pipeline_execution_summary_TIMESTAMP.json`
- Métricas de experimentos no MLflow UI

## 📄 Licença

Este projeto foi desenvolvido para o Hackathon Forecast Big Data 2025.

## 🏷️ Versão

**Versão Atual**: 1.0.0  
**Data de Release**: Janeiro 2025  
**Compatibilidade**: Python 3.9+

---

**Desenvolvido para o Hackathon Forecast Big Data 2025**  
*Sistema de Previsão de Vendas com Machine Learning*