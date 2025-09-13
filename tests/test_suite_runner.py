"""
Runner completo da suite de testes com relatórios detalhados.
"""
import pytest
import sys
import time
import psutil
import os
from pathlib import Path
from typing import Dict, List, Any
import json

class TestSuiteRunner:
    """Executor da suite completa de testes com métricas."""
    
    def __init__(self):
        self.results = {}
        self.start_time = None
        self.end_time = None
        
    def run_unit_tests(self) -> Dict[str, Any]:
        """Executa testes unitários."""
        print("🧪 Executando testes unitários...")
        
        unit_tests = [
            "test_data_ingestion.py",
            "test_data_preprocessing.py", 
            "test_feature_engineering.py",
            "test_feature_selection.py",
            "test_model_training.py",
            "test_ensemble.py",
            "test_prediction.py",
            "test_validation.py",
            "test_output_formatter.py",
            "test_eda.py"
        ]
        
        results = {}
        for test_file in unit_tests:
            print(f"  Executando {test_file}...")
            start = time.time()
            
            exit_code = pytest.main([
                f"tests/{test_file}",
                "-v",
                "-m", "not slow",
                "--tb=short"
            ])
            
            duration = time.time() - start
            results[test_file] = {
                "status": "PASSED" if exit_code == 0 else "FAILED",
                "duration": duration,
                "exit_code": exit_code
            }
            
        return results
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """Executa testes de integração."""
        print("🔗 Executando testes de integração...")
        
        integration_tests = [
            "test_pipeline_integration.py",
            "test_experiment_tracking.py"
        ]
        
        results = {}
        for test_file in integration_tests:
            print(f"  Executando {test_file}...")
            start = time.time()
            
            exit_code = pytest.main([
                f"tests/{test_file}",
                "-v",
                "-m", "integration",
                "--tb=short"
            ])
            
            duration = time.time() - start
            results[test_file] = {
                "status": "PASSED" if exit_code == 0 else "FAILED", 
                "duration": duration,
                "exit_code": exit_code
            }
            
        return results
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Executa testes de performance."""
        print("⚡ Executando testes de performance...")
        
        start = time.time()
        exit_code = pytest.main([
            "tests/test_performance_optimizations.py",
            "-v",
            "-m", "performance",
            "--tb=short"
        ])
        
        duration = time.time() - start
        return {
            "test_performance_optimizations.py": {
                "status": "PASSED" if exit_code == 0 else "FAILED",
                "duration": duration,
                "exit_code": exit_code
            }
        }
    
    def run_coverage_analysis(self) -> Dict[str, Any]:
        """Executa análise de cobertura de código."""
        print("📊 Executando análise de cobertura...")
        
        start = time.time()
        exit_code = pytest.main([
            "tests/",
            "--cov=src",
            "--cov-report=html:htmlcov",
            "--cov-report=term-missing",
            "--cov-report=json:coverage.json",
            "-m", "not slow"
        ])
        
        duration = time.time() - start
        
        # Ler resultados de cobertura se disponível
        coverage_data = {}
        if os.path.exists("coverage.json"):
            with open("coverage.json", "r") as f:
                coverage_data = json.load(f)
        
        return {
            "status": "PASSED" if exit_code == 0 else "FAILED",
            "duration": duration,
            "coverage_data": coverage_data
        }
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Coleta métricas do sistema durante os testes."""
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "python_version": sys.version,
            "platform": sys.platform
        }
    
    def run_full_suite(self) -> Dict[str, Any]:
        """Executa a suite completa de testes."""
        print("🚀 Iniciando suite completa de testes...")
        self.start_time = time.time()
        
        # Métricas do sistema
        system_metrics = self.get_system_metrics()
        
        # Executar diferentes tipos de teste
        unit_results = self.run_unit_tests()
        integration_results = self.run_integration_tests()
        performance_results = self.run_performance_tests()
        coverage_results = self.run_coverage_analysis()
        
        self.end_time = time.time()
        total_duration = self.end_time - self.start_time
        
        # Compilar resultados
        self.results = {
            "summary": {
                "total_duration": total_duration,
                "start_time": self.start_time,
                "end_time": self.end_time,
                "system_metrics": system_metrics
            },
            "unit_tests": unit_results,
            "integration_tests": integration_results,
            "performance_tests": performance_results,
            "coverage": coverage_results
        }
        
        self.generate_report()
        return self.results
    
    def generate_report(self):
        """Gera relatório detalhado dos testes."""
        print("\n" + "="*60)
        print("📋 RELATÓRIO DA SUITE DE TESTES")
        print("="*60)
        
        # Resumo geral
        total_tests = (len(self.results["unit_tests"]) + 
                      len(self.results["integration_tests"]) + 
                      len(self.results["performance_tests"]))
        
        passed_tests = sum(1 for category in ["unit_tests", "integration_tests", "performance_tests"]
                          for test, result in self.results[category].items()
                          if result["status"] == "PASSED")
        
        print(f"Total de testes: {total_tests}")
        print(f"Testes aprovados: {passed_tests}")
        print(f"Taxa de sucesso: {passed_tests/total_tests*100:.1f}%")
        print(f"Duração total: {self.results['summary']['total_duration']:.2f}s")
        
        # Detalhes por categoria
        for category in ["unit_tests", "integration_tests", "performance_tests"]:
            print(f"\n{category.upper().replace('_', ' ')}:")
            for test, result in self.results[category].items():
                status_icon = "✅" if result["status"] == "PASSED" else "❌"
                print(f"  {status_icon} {test}: {result['duration']:.2f}s")
        
        # Cobertura
        if "coverage_data" in self.results["coverage"]:
            coverage = self.results["coverage"]["coverage_data"]
            if "totals" in coverage:
                total_coverage = coverage["totals"]["percent_covered"]
                print(f"\n📊 Cobertura de código: {total_coverage:.1f}%")
        
        # Salvar relatório em arquivo
        with open("test_report.json", "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n📄 Relatório detalhado salvo em: test_report.json")
        print("📊 Relatório de cobertura HTML em: htmlcov/index.html")

def main():
    """Função principal para executar a suite."""
    runner = TestSuiteRunner()
    results = runner.run_full_suite()
    
    # Exit code baseado nos resultados
    failed_tests = sum(1 for category in ["unit_tests", "integration_tests", "performance_tests"]
                      for test, result in results[category].items()
                      if result["status"] == "FAILED")
    
    sys.exit(1 if failed_tests > 0 else 0)

if __name__ == "__main__":
    main()