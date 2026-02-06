"""
Configuration loader for RAG evaluation pipeline
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class DatasetConfig:
    """Dataset configuration"""
    name: str
    base_path: str
    corpus_file: str
    queries_file: str
    qrels_file: Optional[str] = None

    @property
    def corpus_path(self) -> Path:
        return Path(self.base_path) / self.corpus_file

    @property
    def queries_path(self) -> Path:
        return Path(self.base_path) / self.queries_file

    @property
    def qrels_path(self) -> Optional[Path]:
        if self.qrels_file is None:
            return None
        return Path(self.base_path) / self.qrels_file


@dataclass
class EmbeddingConfig:
    """Embedding model configuration"""
    model: str
    batch_size: int = 32
    normalize: bool = True


@dataclass
class RetrievalConfig:
    """Retrieval configuration"""
    method: str = "faiss"
    index_type: str = "flat_ip"
    top_k: int = 50
    query_subset: Optional[int] = None


@dataclass
class RerankerConfig:
    """Single reranker configuration"""
    name: str
    type: str  # cohere, zerank, jina, voyage, qwen
    model: Optional[str] = None  # Required for Qwen, optional for API rerankers
    api_key_env: str = ""
    top_k: int = 15
    
    @property
    def api_key(self) -> Optional[str]:
        if self.api_key_env is None or self.api_key_env == "":
            return None
        return os.getenv(self.api_key_env)


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    metrics: List[str] = field(default_factory=lambda: ["ndcg@5", "ndcg@10", "recall@5", "recall@10"])
    generate_plots: bool = True


@dataclass
class LLMJudgeConfig:
    """LLM Judge configuration"""
    enabled: bool = True
    provider: str = "azure_openai"
    azure_api_key_env: str = "AZURE_API_KEY"
    azure_resource_name_env: str = "AZURE_RESOURCE_NAME"
    azure_deployment_id_env: str = "AZURE_DEPLOYMENT_ID"
    elo_initial_rating: int = 1500
    elo_k_factor: int = 32
    prompt_truncate_doc_length: int = 300
    prompt_top_k_for_comparison: int = 15
    
    @property
    def azure_api_key(self) -> Optional[str]:
        return os.getenv(self.azure_api_key_env)
    
    @property
    def azure_resource_name(self) -> Optional[str]:
        return os.getenv(self.azure_resource_name_env)
    
    @property
    def azure_deployment_id(self) -> Optional[str]:
        return os.getenv(self.azure_deployment_id_env)


@dataclass
class PipelineConfig:
    """Main pipeline configuration"""
    stages: List[str] = field(default_factory=lambda: ["embed", "retrieve", "rerank", "evaluate", "llm_judge"])
    skip_if_exists: bool = False


@dataclass
class Config:
    """Main configuration container"""
    dataset: DatasetConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    rerankers: List[RerankerConfig]
    evaluation: EvaluationConfig
    llm_judge: LLMJudgeConfig
    pipeline: PipelineConfig
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Flatten nested llm_judge config (elo and prompt sections)
        llm_judge_data = data['llm_judge'].copy()
        if 'elo' in llm_judge_data:
            llm_judge_data['elo_initial_rating'] = llm_judge_data['elo'].get('initial_rating', 1500)
            llm_judge_data['elo_k_factor'] = llm_judge_data['elo'].get('k_factor', 32)
            del llm_judge_data['elo']
        if 'prompt' in llm_judge_data:
            llm_judge_data['prompt_truncate_doc_length'] = llm_judge_data['prompt'].get('truncate_doc_length', 300)
            llm_judge_data['prompt_top_k_for_comparison'] = llm_judge_data['prompt'].get('top_k_for_comparison', 15)
            del llm_judge_data['prompt']
        
        return cls(
            dataset=DatasetConfig(**data['dataset']),
            embedding=EmbeddingConfig(**data['embedding']),
            retrieval=RetrievalConfig(**data['retrieval']),
            rerankers=[RerankerConfig(**r) for r in data['rerankers']],
            evaluation=EvaluationConfig(**data['evaluation']),
            llm_judge=LLMJudgeConfig(**llm_judge_data),
            pipeline=PipelineConfig(**data['pipeline'])
        )
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        # Validate dataset paths exist
        if not self.dataset.corpus_path.exists():
            errors.append(f"Corpus file not found: {self.dataset.corpus_path}")
        if not self.dataset.queries_path.exists():
            errors.append(f"Queries file not found: {self.dataset.queries_path}")
        if self.dataset.qrels_path is not None and not self.dataset.qrels_path.exists():
            errors.append(f"Qrels file not found: {self.dataset.qrels_path}")
        
        # Validate rerankers have API keys (skip for local models like qwen where api_key_env is empty)
        for reranker in self.rerankers:
            if reranker.api_key_env and reranker.api_key_env != "" and reranker.api_key is None:
                errors.append(f"API key not set for reranker {reranker.name} (env: {reranker.api_key_env})")
            # Validate Qwen rerankers have model specified
            if reranker.type == "qwen" and not reranker.model:
                errors.append(f"Model not specified for Qwen reranker {reranker.name}")
        
        # Validate LLM judge if enabled
        if self.llm_judge.enabled:
            if self.llm_judge.provider == "azure_openai":
                if not self.llm_judge.azure_api_key:
                    errors.append(f"Azure API key not set (env: {self.llm_judge.azure_api_key_env})")
                if not self.llm_judge.azure_resource_name:
                    errors.append(f"Azure resource name not set (env: {self.llm_judge.azure_resource_name_env})")
                if not self.llm_judge.azure_deployment_id:
                    errors.append(f"Azure deployment ID not set (env: {self.llm_judge.azure_deployment_id_env})")
        
        return errors

