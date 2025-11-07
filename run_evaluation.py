#!/usr/bin/env python3
"""
Run evaluation on existing run directories that have qrels but missing metrics
"""

import sys
from pathlib import Path

# Add parent directory to path to import pipeline
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.stages.evaluate import evaluate_stage
from pipeline.config import Config
from pipeline.paths import RunPaths
from pipeline.logger import PipelineLogger

# Datasets with their run directories and qrels paths
DATASETS_TO_EVALUATE = {
    'msmarco': {
        'run_dir': 'runs/msmarco/20251105_195820',
        'dataset_name': 'msmarco',
        'qrels_path': 'datasets/msmarco/qrels/dev.tsv'
    },
    'dbpedia': {
        'run_dir': 'runs/dbpedia/20251105_210320',
        'dataset_name': 'dbpedia-entity',
        'qrels_path': 'datasets/dbpedia-entity/qrels/dev.tsv'
    }
}

def run_evaluation_for_dataset(dataset_key, dataset_info):
    """Run evaluation for a specific dataset"""
    print(f"\n{'='*70}")
    print(f"Evaluating: {dataset_key}")
    print(f"{'='*70}")

    run_dir = Path(dataset_info['run_dir'])

    if not run_dir.exists():
        print(f"❌ Run directory not found: {run_dir}")
        return False

    # Check if metrics already exist
    metrics_file = run_dir / 'evaluation' / 'metrics.csv'
    if metrics_file.exists():
        print(f"⏭️  Metrics already exist: {metrics_file}")
        return True

    # Detect rerankers from the rerank directory
    rerank_dir = run_dir / 'rerank'
    reranker_names = []
    for f in rerank_dir.glob('reranked_*.jsonl'):
        reranker_name = f.stem.replace('reranked_', '')
        reranker_names.append(reranker_name)

    print(f"   Found {len(reranker_names)} rerankers: {', '.join(reranker_names)}")

    # Load config (we'll create a minimal one)
    class MinimalConfig:
        class Dataset:
            def __init__(self, name, qrels_path):
                self.name = name
                self.qrels_file = qrels_path
                self.qrels_path = qrels_path  # Add both attributes for compatibility

        class Evaluation:
            def __init__(self):
                self.metrics = ["ndcg@5", "ndcg@10", "recall@5", "recall@10"]
                self.generate_plots = False

        class Reranker:
            def __init__(self, name):
                self.name = name

        def __init__(self, dataset_name, qrels_path, reranker_names):
            self.dataset = self.Dataset(dataset_name, qrels_path)
            self.evaluation = self.Evaluation()
            self.rerankers = [self.Reranker(name) for name in reranker_names]

    config = MinimalConfig(dataset_info['dataset_name'], dataset_info['qrels_path'], reranker_names)

    # Create RunPaths pointing to existing run directory
    class ExistingRunPaths:
        def __init__(self, run_dir):
            self.run_dir = Path(run_dir)
            self.rerank_dir = self.run_dir / 'rerank'
            self.evaluation_dir = self.run_dir / 'evaluation'
            self.evaluation_dir.mkdir(exist_ok=True)

        def get_evaluation_metrics_file(self) -> Path:
            """Get path for evaluation metrics CSV"""
            return self.evaluation_dir / "metrics.csv"

        def get_evaluation_plot_file(self) -> Path:
            """Get path for evaluation plot"""
            return self.evaluation_dir / "metrics_plot.png"

        def get_reranked_file(self, reranker_name: str) -> Path:
            """Get path for reranked results"""
            return self.rerank_dir / f"reranked_{reranker_name}.jsonl"

    paths = ExistingRunPaths(run_dir)

    # Create logger
    with PipelineLogger(paths, 'evaluate') as logger:
        try:
            print(f"⏳ Running evaluation...")
            result = evaluate_stage(config, paths, logger)

            if result.get('status') == 'completed':
                print(f"✅ Evaluation completed successfully!")
                print(f"   Metrics saved to: {metrics_file}")
                return True
            else:
                print(f"⚠️  Evaluation finished with status: {result.get('status')}")
                return False

        except Exception as e:
            print(f"❌ ERROR: Evaluation failed!")
            print(f"   Error: {str(e)}")
            logger.error(f"Evaluation failed: {e}", exc_info=True)
            return False

def main():
    """Main entry point"""
    print("\n" + "="*70)
    print("🔬 RUNNING EVALUATIONS FOR DATASETS WITH MISSING METRICS")
    print("="*70)

    success_count = 0
    total_count = len(DATASETS_TO_EVALUATE)

    for dataset_key, dataset_info in DATASETS_TO_EVALUATE.items():
        if run_evaluation_for_dataset(dataset_key, dataset_info):
            success_count += 1

    print("\n" + "="*70)
    print(f"📊 EVALUATION SUMMARY")
    print("="*70)
    print(f"   Total datasets: {total_count}")
    print(f"   Successful: {success_count}")
    print(f"   Failed: {total_count - success_count}")
    print("="*70 + "\n")

    if success_count == total_count:
        print("✅ All evaluations completed successfully!")
        print("\nNext step: Run generate_results.py to update results.json with the new metrics")
    else:
        print("⚠️  Some evaluations failed. Check the logs above for details.")

if __name__ == '__main__':
    main()
