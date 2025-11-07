"""
Path management for pipeline runs
"""

from pathlib import Path
from datetime import datetime
from typing import Optional
import json


class RunPaths:
    """Manages paths for a pipeline run"""
    
    def __init__(self, dataset_name: str, base_runs_dir: str = "runs", timestamp: Optional[str] = None):
        """
        Initialize run paths
        
        Args:
            dataset_name: Name of the dataset
            base_runs_dir: Base directory for runs
            timestamp: Optional timestamp string (defaults to current time)
        """
        self.dataset_name = dataset_name
        self.base_runs_dir = Path(base_runs_dir)
        
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.timestamp = timestamp
        
        self.run_dir = self.base_runs_dir / dataset_name / timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.embeddings_dir = self.run_dir / "embeddings"
        self.retrieval_dir = self.run_dir / "retrieval"
        self.rerank_dir = self.run_dir / "rerank"
        self.evaluation_dir = self.run_dir / "evaluation"
        self.llm_judge_dir = self.run_dir / "llm_judge"
        
        for dir_path in [self.embeddings_dir, self.retrieval_dir, self.rerank_dir, 
                         self.evaluation_dir, self.llm_judge_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def save_metadata(self, metadata: dict):
        """Save metadata JSON to run directory"""
        metadata_file = self.run_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def get_embedding_file(self, model_name: str) -> Path:
        """Get path for embedding file"""
        safe_name = model_name.replace("/", "_").replace("-", "_")
        return self.embeddings_dir / f"{safe_name}.npz"
    
    def get_retrieval_file(self) -> Path:
        """Get path for retrieval results"""
        return self.retrieval_dir / "results.json"
    
    def get_reranked_file(self, reranker_name: str) -> Path:
        """Get path for reranked results"""
        return self.rerank_dir / f"reranked_{reranker_name}.jsonl"
    
    def get_latency_file(self, reranker_name: str) -> Path:
        """Get path for latency CSV"""
        return self.rerank_dir / f"latency_{reranker_name}.csv"
    
    def get_latency_summary_file(self) -> Path:
        """Get path for latency summary"""
        return self.rerank_dir / "latency_summary.csv"
    
    def get_latency_plot_file(self) -> Path:
        """Get path for latency plot"""
        return self.rerank_dir / "latency_plot.png"
    
    def get_evaluation_metrics_file(self) -> Path:
        """Get path for evaluation metrics CSV"""
        return self.evaluation_dir / "metrics.csv"
    
    def get_evaluation_plot_file(self) -> Path:
        """Get path for evaluation plot"""
        return self.evaluation_dir / "metrics_plot.png"
    
    def get_judgments_file(self) -> Path:
        """Get path for LLM judgments JSONL"""
        return self.llm_judge_dir / "judgments.jsonl"
    
    def get_judgments_csv_file(self) -> Path:
        """Get path for judgments CSV"""
        return self.llm_judge_dir / "judgments.csv"
    
    def get_elo_leaderboard_file(self) -> Path:
        """Get path for ELO leaderboard"""
        return self.llm_judge_dir / "elo_leaderboard.csv"
    
    def get_elo_plot_file(self) -> Path:
        """Get path for ELO plot"""
        return self.llm_judge_dir / "elo_leaderboard_plot.png"
    
    def __repr__(self):
        return f"RunPaths(dataset={self.dataset_name}, timestamp={self.timestamp}, run_dir={self.run_dir})"

