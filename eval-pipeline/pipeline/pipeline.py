"""
Main pipeline orchestrator
"""

import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from .config import Config
from .paths import RunPaths
from .logger import PipelineLogger
from .stages import (
    embed_stage,
    retrieve_stage,
    rerank_stage,
    evaluate_stage,
    llm_judge_stage
)


class Pipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize pipeline with configuration
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config = Config.from_yaml(config_path)
        self.errors = self.config.validate()
        
        if self.errors:
            raise ValueError(f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in self.errors))
        
        # Initialize run paths
        self.paths = RunPaths(self.config.dataset.name)
        
        # Store results
        self.results = {}
    
    def run(self, stages: list = None):
        """
        Run the pipeline
        
        Args:
            stages: Optional list of stages to run (defaults to config.pipeline.stages)
        """
        if stages is None:
            stages = self.config.pipeline.stages
        
        print("\n" + "="*70)
        print("🚀 RAG EVALUATION PIPELINE")
        print("="*70)
        print(f"📁 Dataset: {self.config.dataset.name}")
        print(f"💾 Run directory: {self.paths.run_dir}")
        print(f"📋 Stages to run: {', '.join(stages)}")
        print(f"📊 Total stages: {len(stages)}")
        print("="*70)
        
        # Save metadata
        metadata = {
            'dataset': self.config.dataset.name,
            'timestamp': self.paths.timestamp,
            'run_dir': str(self.paths.run_dir),
            'config': {
                'embedding_model': self.config.embedding.model,
                'retrieval_top_k': self.config.retrieval.top_k,
                'num_rerankers': len(self.config.rerankers),
                'rerankers': [r.name for r in self.config.rerankers],
                'metrics': self.config.evaluation.metrics,
                'llm_judge_enabled': self.config.llm_judge.enabled
            }
        }
        self.paths.save_metadata(metadata)
        
        # Run stages
        stage_map = {
            'embed': embed_stage,
            'retrieve': retrieve_stage,
            'rerank': rerank_stage,
            'evaluate': evaluate_stage,
            'llm_judge': llm_judge_stage
        }
        
        total_stages = len(stages)
        for stage_idx, stage_name in enumerate(stages, 1):
            if stage_name not in stage_map:
                print(f"\n⚠️  Warning: Unknown stage '{stage_name}', skipping...")
                continue
            
            print(f"\n{'='*70}")
            print(f"📍 STAGE {stage_idx}/{total_stages}: {stage_name.upper()}")
            print(f"{'='*70}")
            
            with PipelineLogger(self.paths, stage_name) as logger:
                try:
                    stage_func = stage_map[stage_name]
                    print(f"⏳ Running {stage_name}...")
                    result = stage_func(self.config, self.paths, logger)
                    self.results[stage_name] = result
                    status = result.get('status', 'unknown')
                    
                    if status == 'completed':
                        print(f"✅ Stage '{stage_name}' completed successfully!")
                    elif status == 'skipped':
                        print(f"⏭️  Stage '{stage_name}' skipped (already exists)")
                    else:
                        print(f"✓ Stage '{stage_name}' finished: {status}")
                    
                except Exception as e:
                    print(f"\n❌ ERROR: Stage '{stage_name}' failed!")
                    print(f"   Error: {str(e)}")
                    logger.error(f"Stage '{stage_name}' failed: {e}", exc_info=True)
                    self.results[stage_name] = {'status': 'failed', 'error': str(e)}
                    raise
        
        print("\n" + "="*70)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"📂 Results saved to: {self.paths.run_dir}")
        print("\n📊 Output Summary:")
        print(f"   • Embeddings: {self.paths.embeddings_dir}")
        print(f"   • Retrieval: {self.paths.retrieval_dir}")
        print(f"   • Reranking: {self.paths.rerank_dir}")
        print(f"   • Evaluation: {self.paths.evaluation_dir}")
        if self.config.llm_judge.enabled:
            print(f"   • LLM Judge: {self.paths.llm_judge_dir}")
        print("="*70 + "\n")
        
        return self.results


def main():
    """Main entry point"""
    import sys
    
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    
    pipeline = Pipeline(config_path)
    pipeline.run()


if __name__ == '__main__':
    main()

