# Makefile para automação de testes e validações

.PHONY: help test test-unit test-integration test-performance test-quality test-coverage lint format check-format install-dev clean

# Configurações
PYTHON := python
PIP := pip
SRC_DIR := src
TEST_DIR := tests

help: ## Mostra esta mensagem de ajuda
	@echo "Comandos disponíveis:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install-dev: ## Instala dependências de desenvolvimento
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-cov black flake8 isort bandit

test: ## Executa todos os testes
	$(PYTHON) run_tests.py --all

test-fast: ## Executa apenas testes rápidos
	$(PYTHON) run_tests.py --fast

test-unit: ## Executa apenas testes unitários
	$(PYTHON) run_tests.py --unit

test-integration: ## Executa apenas testes de integração
	$(PYTHON) run_tests.py --integration

test-performance: ## Executa apenas testes de performance
	$(PYTHON) run_tests.py --performance

test-quality: ## Executa apenas testes de qualidade de código
	$(PYTHON) run_tests.py --quality

test-coverage: ## Executa análise de cobertura
	$(PYTHON) run_tests.py --coverage

lint: ## Executa linting
	$(PYTHON) run_tests.py --lint

format: ## Formata código com black e isort
	$(PYTHON) -m black $(SRC_DIR) $(TEST_DIR) main.py
	$(PYTHON) -m isort $(SRC_DIR) $(TEST_DIR) main.py

check-format: ## Verifica formatação sem modificar
	$(PYTHON) run_tests.py --format

security: ## Executa verificação de segurança
	$(PYTHON) run_tests.py --security

benchmark: ## Executa benchmarks de performance
	$(PYTHON) -m pytest $(TEST_DIR)/test_benchmarks.py -v -m performance

clean: ## Remove arquivos temporários e cache
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf coverage.json
	rm -rf test_report.json

setup-pre-commit: ## Configura pre-commit hooks
	@echo "#!/bin/bash" > .git/hooks/pre-commit
	@echo "make test-fast" >> .git/hooks/pre-commit
	@chmod +x .git/hooks/pre-commit
	@echo "Pre-commit hook configurado!"

ci: ## Executa pipeline de CI (para integração contínua)
	$(PYTHON) run_tests.py --all

validate: ## Validação completa antes de commit
	make clean
	make format
	make test

# Comandos específicos do pytest
pytest-unit: ## Executa testes unitários com pytest diretamente
	$(PYTHON) -m pytest $(TEST_DIR) -v -m "not integration and not performance and not slow"

pytest-integration: ## Executa testes de integração com pytest diretamente
	$(PYTHON) -m pytest $(TEST_DIR)/test_pipeline_integration.py -v -m integration

pytest-performance: ## Executa testes de performance com pytest diretamente
	$(PYTHON) -m pytest $(TEST_DIR)/test_benchmarks.py -v -m performance

pytest-coverage: ## Executa cobertura com pytest diretamente
	$(PYTHON) -m pytest $(TEST_DIR) --cov=$(SRC_DIR) --cov-report=html --cov-report=term-missing

# Comandos de desenvolvimento
dev-setup: install-dev setup-pre-commit ## Configuração completa para desenvolvimento
	@echo "Ambiente de desenvolvimento configurado!"

dev-test: ## Testes rápidos para desenvolvimento
	$(PYTHON) -m pytest $(TEST_DIR) -v -x --tb=short -m "not slow"

# Relatórios
report: ## Gera relatórios de teste e cobertura
	$(PYTHON) run_tests.py --coverage
	@echo "Relatórios gerados:"
	@echo "  - Cobertura HTML: htmlcov/index.html"
	@echo "  - Cobertura JSON: coverage.json"

# Verificações específicas
check-imports: ## Verifica imports não utilizados
	$(PYTHON) -c "import ast, sys; [print(f'Checking {f}') for f in sys.argv[1:]]" $(SRC_DIR)/*.py

check-complexity: ## Verifica complexidade do código
	@echo "Verificando complexidade do código..."
	@find $(SRC_DIR) -name "*.py" -exec wc -l {} + | sort -n

# Docker (se aplicável)
docker-test: ## Executa testes em container Docker
	docker build -t hackathon-forecast-test .
	docker run --rm hackathon-forecast-test make test

# Documentação
docs: ## Gera documentação
	@echo "Gerando documentação..."
	@find $(SRC_DIR) -name "*.py" -exec $(PYTHON) -c "import ast, inspect; print('Documentação para:', '{}')".format('{}') \;

# Métricas
metrics: ## Coleta métricas do código
	@echo "Coletando métricas do código..."
	@echo "Linhas de código:"
	@find $(SRC_DIR) -name "*.py" -exec wc -l {} + | tail -1
	@echo "Número de arquivos Python:"
	@find $(SRC_DIR) -name "*.py" | wc -l
	@echo "Número de testes:"
	@find $(TEST_DIR) -name "test_*.py" | wc -l

# Pipeline de execução
run-pipeline: ## Executa pipeline completo
	$(PYTHON) main.py --step full --verbose

run-ingestion: ## Executa apenas ingestão de dados
	$(PYTHON) main.py --step ingestion --verbose

run-preprocessing: ## Executa apenas pré-processamento
	$(PYTHON) main.py --step preprocessing --verbose

run-experiments: ## Executa análise de experimentos
	$(PYTHON) main.py --step experiments --verbose

run-submissions: ## Executa sistema de múltiplas submissões
	$(PYTHON) main.py --step submissions --verbose

# Sistema de submissões
generate-all-submissions: ## Gera todas as submissões via CLI
	$(PYTHON) scripts/submission_cli.py generate-all --verbose

generate-submission: ## Gera submissão específica (uso: make generate-submission STRATEGY=nome)
	@echo "Uso: make generate-submission STRATEGY=nome_da_estrategia"
	@if [ -z "$(STRATEGY)" ]; then \
		echo "Erro: Especifique STRATEGY=nome_da_estrategia"; \
		$(PYTHON) scripts/submission_cli.py list-strategies; \
	else \
		$(PYTHON) scripts/submission_cli.py generate-single $(STRATEGY) --verbose; \
	fi

list-strategies: ## Lista estratégias disponíveis
	$(PYTHON) scripts/submission_cli.py list-strategies

list-submissions: ## Lista submissões existentes
	$(PYTHON) scripts/submission_cli.py list-submissions

compare-submissions: ## Compara performance das submissões
	$(PYTHON) scripts/submission_cli.py compare --report submission_comparison_report.html

show-best-submissions: ## Mostra melhores submissões
	$(PYTHON) scripts/submission_cli.py show-best

cleanup-submissions: ## Limpa submissões antigas (manter 5 melhores)
	$(PYTHON) scripts/submission_cli.py cleanup --keep 5

cache-stats: ## Ver estatísticas do cache
	$(PYTHON) scripts/submission_cli.py cache-stats

clear-cache: ## Limpar cache
	$(PYTHON) scripts/submission_cli.py clear-cache --type all

validate-submission: ## Valida arquivo de submissão (uso: make validate-submission FILE=arquivo)
	@echo "Uso: make validate-submission FILE=caminho_do_arquivo"
	@if [ -z "$(FILE)" ]; then \
		echo "Erro: Especifique FILE=caminho_do_arquivo"; \
	else \
		$(PYTHON) scripts/submission_cli.py validate $(FILE); \
	fi

test-submission-system: ## Testa sistema de submissões
	$(PYTHON) -m pytest $(TEST_DIR)/test_submission_system.py -v