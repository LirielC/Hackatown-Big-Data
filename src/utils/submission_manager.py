"""
Sistema de Gerenciamento de Submissões - Hackathon 2025
Gerencia múltiplas estratégias de submissão, versionamento e comparação de performance.
"""

import os
import json
import hashlib
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import yaml
import logging
from dataclasses import dataclass, asdict
import subprocess

logger = logging.getLogger(__name__)


@dataclass
class SubmissionMetadata:
    """Metadados de uma submissão."""
    strategy_name: str
    version: str
    timestamp: str
    config_hash: str
    git_commit: Optional[str] = None
    performance_metrics: Optional[Dict[str, float]] = None
    model_parameters: Optional[Dict[str, Any]] = None
    feature_importance: Optional[Dict[str, float]] = None
    file_path: Optional[str] = None
    validation_results: Optional[Dict[str, Any]] = None


class SubmissionVersionManager:
    """Gerencia versionamento de submissões."""
    
    def __init__(self, base_dir: str = "submissions"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.versions_file = self.base_dir / "versions.json"
        self.metadata_dir = self.base_dir / "metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
    def get_next_version(self, strategy_name: str) -> str:
        """Obtém próxima versão para uma estratégia."""
        versions = self._load_versions()
        
        if strategy_name not in versions:
            versions[strategy_name] = {"major": 1, "minor": 0, "patch": 0}
        else:
            # Incrementa patch por padrão
            versions[strategy_name]["patch"] += 1
        
        self._save_versions(versions)
        
        version_info = versions[strategy_name]
        return f"v{version_info['major']}.{version_info['minor']}.{version_info['patch']}"
    
    def increment_minor_version(self, strategy_name: str) -> str:
        """Incrementa versão minor (mudanças significativas)."""
        versions = self._load_versions()
        
        if strategy_name not in versions:
            versions[strategy_name] = {"major": 1, "minor": 1, "patch": 0}
        else:
            versions[strategy_name]["minor"] += 1
            versions[strategy_name]["patch"] = 0
        
        self._save_versions(versions)
        
        version_info = versions[strategy_name]
        return f"v{version_info['major']}.{version_info['minor']}.{version_info['patch']}"
    
    def increment_major_version(self, strategy_name: str) -> str:
        """Incrementa versão major (mudanças revolucionárias)."""
        versions = self._load_versions()
        
        if strategy_name not in versions:
            versions[strategy_name] = {"major": 2, "minor": 0, "patch": 0}
        else:
            versions[strategy_name]["major"] += 1
            versions[strategy_name]["minor"] = 0
            versions[strategy_name]["patch"] = 0
        
        self._save_versions(versions)
        
        version_info = versions[strategy_name]
        return f"v{version_info['major']}.{version_info['minor']}.{version_info['patch']}"
    
    def save_metadata(self, metadata: SubmissionMetadata) -> str:
        """Salva metadados de uma submissão."""
        metadata_file = self.metadata_dir / f"{metadata.strategy_name}_{metadata.version}.json"
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(metadata), f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Metadados salvos: {metadata_file}")
        return str(metadata_file)
    
    def load_metadata(self, strategy_name: str, version: str) -> Optional[SubmissionMetadata]:
        """Carrega metadados de uma submissão."""
        metadata_file = self.metadata_dir / f"{strategy_name}_{version}.json"
        
        if not metadata_file.exists():
            return None
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return SubmissionMetadata(**data)
    
    def list_submissions(self, strategy_name: Optional[str] = None) -> List[SubmissionMetadata]:
        """Lista todas as submissões ou de uma estratégia específica."""
        submissions = []
        
        for metadata_file in self.metadata_dir.glob("*.json"):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                metadata = SubmissionMetadata(**data)
                
                if strategy_name is None or metadata.strategy_name == strategy_name:
                    submissions.append(metadata)
                    
            except Exception as e:
                logger.warning(f"Erro ao carregar metadados de {metadata_file}: {e}")
        
        # Ordenar por timestamp
        submissions.sort(key=lambda x: x.timestamp, reverse=True)
        return submissions
    
    def get_best_submission(self, strategy_name: Optional[str] = None, 
                          metric: str = "wmape") -> Optional[SubmissionMetadata]:
        """Obtém melhor submissão baseada em uma métrica."""
        submissions = self.list_submissions(strategy_name)
        
        valid_submissions = [
            s for s in submissions 
            if s.performance_metrics and metric in s.performance_metrics
        ]
        
        if not valid_submissions:
            return None
        
        # Para WMAPE, menor é melhor
        if metric.lower() in ['wmape', 'mae', 'rmse', 'mape']:
            best = min(valid_submissions, key=lambda x: x.performance_metrics[metric])
        else:
            # Para outras métricas, maior é melhor
            best = max(valid_submissions, key=lambda x: x.performance_metrics[metric])
        
        return best
    
    def _load_versions(self) -> Dict[str, Dict[str, int]]:
        """Carrega arquivo de versões."""
        if not self.versions_file.exists():
            return {}
        
        with open(self.versions_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _save_versions(self, versions: Dict[str, Dict[str, int]]) -> None:
        """Salva arquivo de versões."""
        with open(self.versions_file, 'w', encoding='utf-8') as f:
            json.dump(versions, f, indent=2, ensure_ascii=False)


class SubmissionValidator:
    """Valida submissões antes do envio."""
    
    def __init__(self):
        self.required_columns = ['semana', 'pdv', 'produto', 'quantidade']
        self.expected_weeks = [1, 2, 3, 4, 5]  # Semanas de janeiro/2023
    
    def validate_submission(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Valida formato e conteúdo da submissão."""
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        # Validar colunas obrigatórias
        missing_columns = set(self.required_columns) - set(df.columns)
        if missing_columns:
            results['errors'].append(f"Colunas faltantes: {missing_columns}")
            results['is_valid'] = False
        
        # Validar tipos de dados (mais flexível)
        if 'semana' in df.columns:
            # Aceitar inteiros ou strings que podem ser convertidas
            if not (pd.api.types.is_integer_dtype(df['semana']) or
                   (pd.api.types.is_object_dtype(df['semana']) and
                    df['semana'].str.match(r'^\d+$').all())):
                results['warnings'].append("Coluna 'semana' deveria ser inteira")
            else:
                results['is_valid'] = True

        if 'pdv' in df.columns:
            # Aceitar inteiros ou strings numéricas
            if not (pd.api.types.is_integer_dtype(df['pdv']) or
                   pd.api.types.is_numeric_dtype(df['pdv']) or
                   (pd.api.types.is_object_dtype(df['pdv']) and
                    pd.to_numeric(df['pdv'], errors='coerce').notna().all())):
                results['warnings'].append("Coluna 'pdv' deveria ser numérica")
            else:
                results['is_valid'] = True

        if 'produto' in df.columns:
            # Aceitar inteiros ou strings numéricas
            if not (pd.api.types.is_integer_dtype(df['produto']) or
                   pd.api.types.is_numeric_dtype(df['produto']) or
                   (pd.api.types.is_object_dtype(df['produto']) and
                    pd.to_numeric(df['produto'], errors='coerce').notna().all())):
                results['warnings'].append("Coluna 'produto' deveria ser numérica")
            else:
                results['is_valid'] = True
        
        if 'quantidade' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['quantidade']):
                results['errors'].append("Coluna 'quantidade' deve ser numérica")
                results['is_valid'] = False
        
        # Validar valores
        if results['is_valid']:
            # Verificar semanas
            unique_weeks = df['semana'].unique()
            missing_weeks = set(self.expected_weeks) - set(unique_weeks)
            if missing_weeks:
                results['warnings'].append(f"Semanas faltantes: {missing_weeks}")
            
            extra_weeks = set(unique_weeks) - set(self.expected_weeks)
            if extra_weeks:
                results['warnings'].append(f"Semanas extras: {extra_weeks}")
            
            # Verificar valores negativos
            if (df['quantidade'] < 0).any():
                results['errors'].append("Encontrados valores negativos em 'quantidade'")
                results['is_valid'] = False
            
            # Verificar valores nulos
            null_counts = df.isnull().sum()
            if null_counts.any():
                results['warnings'].append(f"Valores nulos encontrados: {null_counts.to_dict()}")
            
            # Estatísticas
            results['stats'] = {
                'total_records': len(df),
                'unique_pdvs': df['pdv'].nunique(),
                'unique_products': df['produto'].nunique(),
                'total_quantity': df['quantidade'].sum(),
                'avg_quantity': df['quantidade'].mean(),
                'min_quantity': df['quantidade'].min(),
                'max_quantity': df['quantidade'].max()
            }
        
        return results


class PerformanceComparator:
    """Compara performance entre diferentes submissões."""
    
    def __init__(self, version_manager: SubmissionVersionManager):
        self.version_manager = version_manager
    
    def compare_submissions(self, submissions: List[SubmissionMetadata], 
                          primary_metric: str = "wmape") -> pd.DataFrame:
        """Compara múltiplas submissões."""
        comparison_data = []
        
        for submission in submissions:
            if not submission.performance_metrics:
                continue
            
            row = {
                'strategy': submission.strategy_name,
                'version': submission.version,
                'timestamp': submission.timestamp,
                'config_hash': submission.config_hash[:8],
                'git_commit': submission.git_commit[:8] if submission.git_commit else 'N/A'
            }
            
            # Adicionar métricas
            for metric, value in submission.performance_metrics.items():
                row[f'metric_{metric}'] = value
            
            comparison_data.append(row)
        
        if not comparison_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(comparison_data)
        
        # Ordenar pelo métrica primária
        if f'metric_{primary_metric}' in df.columns:
            # Para WMAPE, menor é melhor
            ascending = primary_metric.lower() in ['wmape', 'mae', 'rmse', 'mape']
            df = df.sort_values(f'metric_{primary_metric}', ascending=ascending)
        
        return df
    
    def generate_comparison_report(self, submissions: List[SubmissionMetadata],
                                 output_path: str = "submission_comparison.html") -> str:
        """Gera relatório HTML de comparação."""
        comparison_df = self.compare_submissions(submissions)
        
        if comparison_df.empty:
            logger.warning("Nenhuma submissão com métricas para comparar")
            return ""
        
        html_content = self._generate_html_report(comparison_df, submissions)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Relatório de comparação gerado: {output_path}")
        return output_path
    
    def _generate_html_report(self, comparison_df: pd.DataFrame, 
                            submissions: List[SubmissionMetadata]) -> str:
        """Gera conteúdo HTML do relatório."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Relatório de Comparação de Submissões</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .best {{ background-color: #d4edda; }}
                .worst {{ background-color: #f8d7da; }}
                .metric {{ font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>Relatório de Comparação de Submissões</h1>
            <p>Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Resumo das Submissões</h2>
            <p>Total de submissões: {len(submissions)}</p>
            <p>Submissões com métricas: {len(comparison_df)}</p>
            
            <h2>Ranking de Performance</h2>
            {comparison_df.to_html(classes='table', escape=False, index=False)}
            
            <h2>Detalhes das Estratégias</h2>
        """
        
        # Adicionar detalhes de cada estratégia
        for submission in submissions[:5]:  # Top 5
            html += f"""
            <h3>{submission.strategy_name} - {submission.version}</h3>
            <ul>
                <li><strong>Timestamp:</strong> {submission.timestamp}</li>
                <li><strong>Config Hash:</strong> {submission.config_hash[:16]}</li>
                <li><strong>Git Commit:</strong> {submission.git_commit or 'N/A'}</li>
            </ul>
            """
            
            if submission.performance_metrics:
                html += "<h4>Métricas:</h4><ul>"
                for metric, value in submission.performance_metrics.items():
                    html += f"<li><strong>{metric}:</strong> {value:.6f}</li>"
                html += "</ul>"
        
        html += """
        </body>
        </html>
        """
        
        return html


class SubmissionManager:
    """Gerenciador principal de submissões."""
    
    def __init__(self, config_path: str = "configs/submission_strategies.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.version_manager = SubmissionVersionManager(
            self.config['submission']['output_dir']
        )
        self.validator = SubmissionValidator()
        self.comparator = PerformanceComparator(self.version_manager)
        
    def _load_config(self) -> Dict[str, Any]:
        """Carrega configuração de estratégias."""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def get_strategy_config(self, strategy_name: str) -> Dict[str, Any]:
        """Obtém configuração de uma estratégia específica."""
        if strategy_name not in self.config['strategies']:
            raise ValueError(f"Estratégia '{strategy_name}' não encontrada")
        
        return self.config['strategies'][strategy_name]
    
    def list_strategies(self) -> List[str]:
        """Lista estratégias disponíveis."""
        return list(self.config['strategies'].keys())
    
    def create_submission(self, strategy_name: str, predictions_df: pd.DataFrame,
                         performance_metrics: Dict[str, float],
                         model_parameters: Optional[Dict[str, Any]] = None,
                         version_type: str = "patch") -> SubmissionMetadata:
        """Cria nova submissão."""
        # Validar estratégia
        if strategy_name not in self.config['strategies']:
            raise ValueError(f"Estratégia '{strategy_name}' não encontrada")
        
        # Validar submissão
        validation_results = self.validator.validate_submission(predictions_df)
        if not validation_results['is_valid']:
            raise ValueError(f"Submissão inválida: {validation_results['errors']}")
        
        # Gerar versão
        if version_type == "major":
            version = self.version_manager.increment_major_version(strategy_name)
        elif version_type == "minor":
            version = self.version_manager.increment_minor_version(strategy_name)
        else:
            version = self.version_manager.get_next_version(strategy_name)
        
        # Gerar timestamp e hash
        timestamp = datetime.now().isoformat()
        config_hash = self._generate_config_hash(strategy_name)
        git_commit = self._get_git_commit()
        
        # Criar metadados
        metadata = SubmissionMetadata(
            strategy_name=strategy_name,
            version=version,
            timestamp=timestamp,
            config_hash=config_hash,
            git_commit=git_commit,
            performance_metrics=performance_metrics,
            model_parameters=model_parameters,
            validation_results=validation_results
        )
        
        # Salvar arquivos de submissão
        file_paths = self._save_submission_files(
            strategy_name, version, timestamp, predictions_df
        )
        metadata.file_path = file_paths[0]  # Arquivo principal
        
        # Salvar metadados
        self.version_manager.save_metadata(metadata)
        
        # Backup se configurado
        if self.config['submission']['backup']['enabled']:
            self._create_backup(metadata, file_paths)
        
        logger.info(f"Submissão criada: {strategy_name} {version}")
        return metadata
    
    def compare_all_submissions(self, strategy_name: Optional[str] = None) -> pd.DataFrame:
        """Compara todas as submissões."""
        submissions = self.version_manager.list_submissions(strategy_name)
        return self.comparator.compare_submissions(submissions)
    
    def generate_performance_report(self, output_path: Optional[str] = None) -> str:
        """Gera relatório de performance."""
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"submission_performance_report_{timestamp}.html"
        
        submissions = self.version_manager.list_submissions()
        return self.comparator.generate_comparison_report(submissions, output_path)
    
    def get_best_submission_by_strategy(self) -> Dict[str, SubmissionMetadata]:
        """Obtém melhor submissão por estratégia."""
        best_submissions = {}
        
        for strategy_name in self.list_strategies():
            best = self.version_manager.get_best_submission(strategy_name)
            if best:
                best_submissions[strategy_name] = best
        
        return best_submissions
    
    def cleanup_old_submissions(self, keep_versions: int = 10) -> None:
        """Remove submissões antigas mantendo apenas as melhores."""
        for strategy_name in self.list_strategies():
            submissions = self.version_manager.list_submissions(strategy_name)
            
            if len(submissions) <= keep_versions:
                continue
            
            # Manter as melhores por WMAPE
            submissions_with_metrics = [
                s for s in submissions 
                if s.performance_metrics and 'wmape' in s.performance_metrics
            ]
            
            if len(submissions_with_metrics) <= keep_versions:
                continue
            
            # Ordenar por WMAPE (menor é melhor)
            submissions_with_metrics.sort(
                key=lambda x: x.performance_metrics['wmape']
            )
            
            # Remover submissões piores
            to_remove = submissions_with_metrics[keep_versions:]
            
            for submission in to_remove:
                self._remove_submission(submission)
                logger.info(f"Submissão removida: {submission.strategy_name} {submission.version}")
    
    def _generate_config_hash(self, strategy_name: str) -> str:
        """Gera hash da configuração da estratégia."""
        strategy_config = self.get_strategy_config(strategy_name)
        config_str = json.dumps(strategy_config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _get_git_commit(self) -> Optional[str]:
        """Obtém commit atual do Git."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None
    
    def _save_submission_files(self, strategy_name: str, version: str, 
                             timestamp: str, predictions_df: pd.DataFrame) -> List[str]:
        """Salva arquivos da submissão."""
        base_dir = Path(self.config['submission']['output_dir'])
        strategy_dir = base_dir / strategy_name
        strategy_dir.mkdir(parents=True, exist_ok=True)
        
        # Gerar nome do arquivo
        timestamp_str = datetime.fromisoformat(timestamp).strftime('%Y%m%d_%H%M%S')
        filename_base = self.config['submission']['filename_template'].format(
            strategy=strategy_name,
            version=version,
            timestamp=timestamp_str
        )
        
        file_paths = []
        
        # Salvar em formatos configurados
        for format_type in self.config['submission']['formats']:
            if format_type == 'csv':
                file_path = strategy_dir / f"{filename_base}.csv"
                predictions_df.to_csv(
                    file_path,
                    sep=self.config.get('output', {}).get('csv_separator', ';'),
                    encoding=self.config.get('output', {}).get('csv_encoding', 'utf-8'),
                    index=False
                )
            elif format_type == 'parquet':
                file_path = strategy_dir / f"{filename_base}.parquet"
                predictions_df.to_parquet(file_path, index=False)
            
            file_paths.append(str(file_path))
            logger.info(f"Arquivo salvo: {file_path}")
        
        return file_paths
    
    def _create_backup(self, metadata: SubmissionMetadata, file_paths: List[str]) -> None:
        """Cria backup da submissão."""
        backup_dir = Path(self.config['submission']['output_dir']) / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        backup_subdir = backup_dir / f"{metadata.strategy_name}_{metadata.version}"
        backup_subdir.mkdir(parents=True, exist_ok=True)
        
        # Copiar arquivos
        for file_path in file_paths:
            src_path = Path(file_path)
            dst_path = backup_subdir / src_path.name
            shutil.copy2(src_path, dst_path)
        
        # Copiar metadados
        metadata_src = self.version_manager.metadata_dir / f"{metadata.strategy_name}_{metadata.version}.json"
        metadata_dst = backup_subdir / "metadata.json"
        shutil.copy2(metadata_src, metadata_dst)
        
        logger.info(f"Backup criado: {backup_subdir}")
    
    def _remove_submission(self, metadata: SubmissionMetadata) -> None:
        """Remove submissão e seus arquivos."""
        # Remover arquivos de submissão
        if metadata.file_path:
            file_path = Path(metadata.file_path)
            if file_path.exists():
                file_path.unlink()
            
            # Remover outros formatos
            for suffix in ['.csv', '.parquet']:
                alt_path = file_path.with_suffix(suffix)
                if alt_path.exists():
                    alt_path.unlink()
        
        # Remover metadados
        metadata_file = self.version_manager.metadata_dir / f"{metadata.strategy_name}_{metadata.version}.json"
        if metadata_file.exists():
            metadata_file.unlink()