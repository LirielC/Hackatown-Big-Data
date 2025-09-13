"""
Testes de qualidade de código e validações estáticas.
"""
import pytest
import subprocess
import sys
import os
from pathlib import Path
import ast
import re

class TestCodeQuality:
    """Testes de qualidade e padrões de código."""
    
    def test_black_formatting(self):
        """Verifica formatação com Black."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "black", "--check", "src/", "tests/"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent
            )
            
            if result.returncode != 0:
                pytest.fail(f"Código não formatado com Black:\n{result.stdout}\n{result.stderr}")
                
        except FileNotFoundError:
            pytest.skip("Black não instalado")
    
    def test_isort_imports(self):
        """Verifica ordenação de imports com isort."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "isort", "--check-only", "src/", "tests/"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent
            )
            
            if result.returncode != 0:
                pytest.fail(f"Imports não ordenados:\n{result.stdout}\n{result.stderr}")
                
        except FileNotFoundError:
            pytest.skip("isort não instalado")
    
    def test_flake8_linting(self):
        """Verifica linting com flake8."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "flake8", "src/", "tests/", "--max-line-length=88"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent
            )
            
            if result.returncode != 0:
                pytest.fail(f"Problemas de linting:\n{result.stdout}\n{result.stderr}")
                
        except FileNotFoundError:
            pytest.skip("flake8 não instalado")
    
    def test_no_print_statements(self):
        """Verifica ausência de print statements no código de produção."""
        src_path = Path(__file__).parent.parent / "src"
        
        for py_file in src_path.rglob("*.py"):
            content = py_file.read_text()
            
            # Ignorar comentários e strings
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if (isinstance(node.func, ast.Name) and 
                        node.func.id == 'print'):
                        pytest.fail(f"Print statement encontrado em {py_file}:{node.lineno}")
    
    def test_no_hardcoded_paths(self):
        """Verifica ausência de caminhos hardcoded."""
        src_path = Path(__file__).parent.parent / "src"
        
        hardcoded_patterns = [
            r'["\'][C-Z]:\\',  # Windows paths
            r'["\']\/home\/',   # Linux home paths
            r'["\']\/Users\/',  # macOS paths
        ]
        
        for py_file in src_path.rglob("*.py"):
            content = py_file.read_text()
            
            for pattern in hardcoded_patterns:
                if re.search(pattern, content):
                    pytest.fail(f"Caminho hardcoded encontrado em {py_file}")
    
    def test_proper_exception_handling(self):
        """Verifica tratamento adequado de exceções."""
        src_path = Path(__file__).parent.parent / "src"
        
        for py_file in src_path.rglob("*.py"):
            content = py_file.read_text()
            
            # Verificar bare except
            if re.search(r'except\s*:', content):
                pytest.fail(f"Bare except encontrado em {py_file}")
            
            # Verificar pass em except
            if re.search(r'except.*:\s*pass', content, re.MULTILINE):
                pytest.fail(f"Except com pass encontrado em {py_file}")
    
    def test_function_complexity(self):
        """Verifica complexidade das funções."""
        src_path = Path(__file__).parent.parent / "src"
        
        for py_file in src_path.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue
                
            content = py_file.read_text()
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Contar linhas da função
                    func_lines = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0
                    
                    if func_lines > 50:
                        pytest.fail(f"Função muito longa: {node.name} em {py_file} ({func_lines} linhas)")
    
    def test_class_naming_convention(self):
        """Verifica convenções de nomenclatura de classes."""
        src_path = Path(__file__).parent.parent / "src"
        
        for py_file in src_path.rglob("*.py"):
            content = py_file.read_text()
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Verificar PascalCase
                    if not re.match(r'^[A-Z][a-zA-Z0-9]*$', node.name):
                        pytest.fail(f"Nome de classe inválido: {node.name} em {py_file}")
    
    def test_function_naming_convention(self):
        """Verifica convenções de nomenclatura de funções."""
        src_path = Path(__file__).parent.parent / "src"
        
        for py_file in src_path.rglob("*.py"):
            content = py_file.read_text()
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Verificar snake_case
                    if not re.match(r'^[a-z_][a-z0-9_]*$', node.name):
                        if not node.name.startswith('__'):  # Ignorar métodos especiais
                            pytest.fail(f"Nome de função inválido: {node.name} em {py_file}")
    
    def test_no_unused_imports(self):
        """Verifica imports não utilizados."""
        src_path = Path(__file__).parent.parent / "src"
        
        for py_file in src_path.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue
                
            content = py_file.read_text()
            tree = ast.parse(content)
            
            imports = set()
            used_names = set()
            
            # Coletar imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        imports.add(alias.name)
                elif isinstance(node, ast.Name):
                    used_names.add(node.id)
            
            # Verificar imports não utilizados (básico)
            unused = imports - used_names
            # Remover imports comuns que podem ser usados indiretamente
            unused = unused - {'os', 'sys', 'logging', 'warnings', 'typing'}
            
            if unused and len(unused) > 2:  # Tolerância para alguns imports
                pytest.fail(f"Possíveis imports não utilizados em {py_file}: {unused}")
    
    def test_docstring_format(self):
        """Verifica formato das docstrings."""
        src_path = Path(__file__).parent.parent / "src"
        
        for py_file in src_path.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue
                
            content = py_file.read_text()
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    # Verificar se tem docstring
                    if (node.body and 
                        isinstance(node.body[0], ast.Expr) and 
                        isinstance(node.body[0].value, ast.Constant)):
                        
                        docstring = node.body[0].value.value
                        if isinstance(docstring, str):
                            # Verificar formato básico
                            if not docstring.strip():
                                pytest.fail(f"Docstring vazia em {node.name} ({py_file})")
    
    def test_security_patterns(self):
        """Verifica padrões de segurança básicos."""
        src_path = Path(__file__).parent.parent / "src"
        
        dangerous_patterns = [
            (r'eval\s*\(', "uso de eval()"),
            (r'exec\s*\(', "uso de exec()"),
            (r'__import__\s*\(', "uso de __import__()"),
            (r'input\s*\(', "uso de input() - preferir argumentos"),
        ]
        
        for py_file in src_path.rglob("*.py"):
            content = py_file.read_text()
            
            for pattern, description in dangerous_patterns:
                if re.search(pattern, content):
                    pytest.fail(f"Padrão perigoso encontrado em {py_file}: {description}")
    
    def test_todo_comments(self):
        """Verifica comentários TODO pendentes."""
        src_path = Path(__file__).parent.parent / "src"
        
        todo_count = 0
        for py_file in src_path.rglob("*.py"):
            content = py_file.read_text()
            
            todos = re.findall(r'#.*TODO.*', content, re.IGNORECASE)
            todo_count += len(todos)
        
        # Permitir alguns TODOs, mas alertar se muitos
        if todo_count > 10:
            pytest.fail(f"Muitos comentários TODO pendentes: {todo_count}")

class TestDependencyValidation:
    """Testes de validação de dependências."""
    
    def test_requirements_file_exists(self):
        """Verifica existência do arquivo requirements.txt."""
        req_file = Path(__file__).parent.parent / "requirements.txt"
        assert req_file.exists(), "Arquivo requirements.txt não encontrado"
    
    def test_no_conflicting_dependencies(self):
        """Verifica conflitos de dependências."""
        req_file = Path(__file__).parent.parent / "requirements.txt"
        
        if not req_file.exists():
            pytest.skip("requirements.txt não encontrado")
        
        content = req_file.read_text()
        
        # Verificar versões conflitantes básicas
        lines = [line.strip() for line in content.split('\n') if line.strip() and not line.startswith('#')]
        
        packages = {}
        for line in lines:
            if '==' in line:
                pkg_name = line.split('==')[0].lower()
                if pkg_name in packages:
                    pytest.fail(f"Dependência duplicada: {pkg_name}")
                packages[pkg_name] = line
    
    def test_security_vulnerabilities(self):
        """Verifica vulnerabilidades conhecidas (básico)."""
        # Este teste seria mais robusto com ferramentas como safety
        req_file = Path(__file__).parent.parent / "requirements.txt"
        
        if not req_file.exists():
            pytest.skip("requirements.txt não encontrado")
        
        content = req_file.read_text()
        
        # Lista básica de pacotes com vulnerabilidades conhecidas
        vulnerable_patterns = [
            r'pillow==.*[1-7]\.',  # Versões antigas do Pillow
            r'requests==.*[0-1]\.',  # Versões muito antigas do requests
        ]
        
        for pattern in vulnerable_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                pytest.fail(f"Possível dependência vulnerável encontrada: {pattern}")