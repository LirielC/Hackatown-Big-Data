#!/usr/bin/env python3
"""
Teste Rápido do Sistema de Submissões - Hackathon 2025
Verifica se o sistema está funcionando sem precisar de dados reais.
"""

import sys
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent.parent))

def create_mock_data():
    """Cria dados mock para teste."""
    print("📊 Criando dados mock...")
    
    np.random.seed(42)
    data = []
    
    # Gerar dados para 5 semanas, 5 PDVs, 10 produtos
    for semana in range(1, 6):
        for pdv in range(1, 6):
            for produto in range(1, 11):
                quantidade = np.random.poisson(25)
                data.append({
                    'semana': semana,
                    'pdv': pdv,
                    'produto': produto,
                    'quantidade': quantidade
                })
    
    df = pd.DataFrame(data)
    print(f"✓ Dados mock criados: {df.shape}")
    return df

def test_submission_validator():
    """Testa o validador de submissões."""
    print("\n🔍 Testando validador de submissões...")
    
    try:
        from src.utils.submission_manager import SubmissionValidator
        
        validator = SubmissionValidator()
        mock_data = create_mock_data()
        
        result = validator.validate_submission(mock_data)
        
        if result['is_valid']:
            print("✓ Validador funcionando corretamente!")
            print(f"  - Total de registros: {result['stats']['total_records']}")
            print(f"  - PDVs únicos: {result['stats']['unique_pdvs']}")
            print(f"  - Produtos únicos: {result['stats']['unique_products']}")
        else:
            print("✗ Validação falhou:")
            for error in result['errors']:
                print(f"    - {error}")
        
        return result['is_valid']
        
    except Exception as e:
        print(f"✗ Erro no validador: {e}")
        return False

def test_version_manager():
    """Testa o gerenciador de versões."""
    print("\n📝 Testando gerenciador de versões...")
    
    try:
        from src.utils.submission_manager import SubmissionVersionManager
        
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SubmissionVersionManager(temp_dir)
            
            # Testar incremento de versões
            v1 = manager.get_next_version('test_strategy')
            v2 = manager.get_next_version('test_strategy')
            v3 = manager.increment_minor_version('test_strategy')
            
            expected = ['v1.0.0', 'v1.0.1', 'v1.1.0']
            actual = [v1, v2, v3]
            
            if actual == expected:
                print("✓ Versionamento funcionando corretamente!")
                print(f"  - Versões geradas: {actual}")
            else:
                print(f"✗ Versionamento incorreto. Esperado: {expected}, Atual: {actual}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Erro no versionamento: {e}")
        return False

def test_config_loading():
    """Testa carregamento de configurações."""
    print("\n⚙️ Testando carregamento de configurações...")
    
    try:
        from src.utils.submission_manager import SubmissionManager
        
        manager = SubmissionManager()
        strategies = manager.list_strategies()
        
        print(f"✓ Configurações carregadas com sucesso!")
        print(f"  - Estratégias encontradas: {len(strategies)}")
        
        for strategy in strategies:
            config = manager.get_strategy_config(strategy)
            print(f"    - {strategy}: {config['name']}")
        
        return len(strategies) > 0
        
    except Exception as e:
        print(f"✗ Erro ao carregar configurações: {e}")
        return False

def test_cache_system():
    """Testa sistema de cache."""
    print("\n💾 Testando sistema de cache...")
    
    try:
        from src.utils.fast_submission_pipeline import FeatureCache, ModelCache
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Testar cache de features
            feature_cache = FeatureCache(temp_dir + "/features")
            
            test_df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
            cache_key = 'test_key'
            
            feature_cache.save_features(cache_key, test_df, {'test': True})
            
            if feature_cache.is_cache_valid(cache_key, 24):
                loaded_df, metadata = feature_cache.load_features(cache_key)
                
                if test_df.equals(loaded_df) and metadata['test']:
                    print("✓ Cache de features funcionando!")
                else:
                    print("✗ Dados do cache não coincidem")
                    return False
            else:
                print("✗ Cache não é válido")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Erro no sistema de cache: {e}")
        return False

def test_cli_commands():
    """Testa comandos da CLI."""
    print("\n🖥️ Testando comandos da CLI...")
    
    try:
        import subprocess
        
        # Testar comando list-strategies
        result = subprocess.run([
            sys.executable, 'scripts/submission_cli.py', 'list-strategies'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✓ CLI funcionando corretamente!")
            print("  - Comando list-strategies executado com sucesso")
            return True
        else:
            print(f"✗ CLI falhou: {result.stderr}")
            return False
        
    except Exception as e:
        print(f"✗ Erro ao testar CLI: {e}")
        return False

def test_makefile_targets():
    """Testa targets do Makefile."""
    print("\n🔨 Testando targets do Makefile...")
    
    try:
        import subprocess
        
        # Testar se o target existe
        result = subprocess.run([
            'make', 'list-strategies'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✓ Targets do Makefile funcionando!")
            return True
        else:
            print(f"✗ Makefile falhou: {result.stderr}")
            return False
        
    except Exception as e:
        print(f"✗ Erro ao testar Makefile: {e}")
        return False

def main():
    """Executa todos os testes."""
    print("🚀 TESTE RÁPIDO DO SISTEMA DE SUBMISSÕES")
    print("=" * 50)
    
    tests = [
        ("Carregamento de Configurações", test_config_loading),
        ("Validador de Submissões", test_submission_validator),
        ("Gerenciador de Versões", test_version_manager),
        ("Sistema de Cache", test_cache_system),
        ("Comandos CLI", test_cli_commands),
        ("Targets Makefile", test_makefile_targets),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ Erro inesperado em {test_name}: {e}")
            results.append((test_name, False))
    
    # Resumo
    print("\n" + "=" * 50)
    print("📋 RESUMO DOS TESTES")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASSOU" if success else "✗ FALHOU"
        print(f"{status:10} | {test_name}")
        if success:
            passed += 1
    
    print("-" * 50)
    print(f"Total: {passed}/{total} testes passaram ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 TODOS OS TESTES PASSARAM!")
        print("O sistema de submissões está funcionando corretamente!")
    else:
        print(f"\n⚠️ {total-passed} teste(s) falharam.")
        print("Verifique os erros acima e corrija antes de usar o sistema.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)