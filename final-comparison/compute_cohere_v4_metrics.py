#!/usr/bin/env python3
"""
Compute metrics ONLY for Cohere v4 models and append to existing metrics.csv
"""

import yaml
import csv
import json
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.logger import setup_logger


def _dcg_at_k(relevance_scores, k):
    """Calculate DCG at rank k"""
    relevance_scores = np.array(relevance_scores[:k])
    if len(relevance_scores) == 0:
        return 0.0
    gains = 2**relevance_scores - 1
    discounts = np.log2(np.arange(len(relevance_scores)) + 2)
    return np.sum(gains / discounts)


def _ndcg_at_k(relevance_scores, k):
    """Calculate NDCG at rank k"""
    dcg = _dcg_at_k(relevance_scores, k)
    if dcg == 0:
        return 0.0
    ideal_relevance = sorted(relevance_scores, reverse=True)
    ideal_dcg = _dcg_at_k(ideal_relevance, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg / ideal_dcg


def _recall_at_k(relevance_scores, k, num_relevant):
    """Calculate Recall at rank k"""
    if num_relevant == 0:
        return 0.0
    retrieved = sum(relevance_scores[:k])
    return retrieved / num_relevant


def _load_qrels(qrels_file):
    """Load relevance judgments from TSV file"""
    qrels = {}
    with open(qrels_file, 'r') as f:
        lines = f.readlines()[1:]  # Skip header
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            query_id = parts[0]
            doc_id = parts[1]
            rel = int(parts[2])
            if query_id not in qrels:
                qrels[query_id] = {}
            qrels[query_id][doc_id] = rel
    return qrels


def _evaluate_reranker(reranked_file, qrels, metrics):
    """Evaluate a single reranker's results"""
    results = {m: [] for m in metrics}

    with open(reranked_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            query_id = data['query_id']

            if query_id not in qrels:
                continue

            relevant_docs = set(qrels[query_id].keys())
            num_relevant = len(relevant_docs)

            # Create relevance scores for retrieved docs
            relevance_scores = []
            for doc in data['results']:
                doc_id = doc['doc_id']
                rel = qrels[query_id].get(doc_id, 0)
                relevance_scores.append(rel)

            # Calculate metrics
            if 'ndcg@5' in metrics:
                results['ndcg@5'].append(_ndcg_at_k(relevance_scores, 5))
            if 'ndcg@10' in metrics:
                results['ndcg@10'].append(_ndcg_at_k(relevance_scores, 10))
            if 'recall@5' in metrics:
                results['recall@5'].append(_recall_at_k(relevance_scores, 5, num_relevant))
            if 'recall@10' in metrics:
                results['recall@10'].append(_recall_at_k(relevance_scores, 10, num_relevant))

    # Calculate averages
    return {m: np.mean(results[m]) if results[m] else 0.0 for m in metrics}


def compute_and_append_metrics(dataset_name, run_dir, config):
    """Compute metrics for Cohere v4 models and append to metrics.csv"""

    logger = setup_logger("compute_cohere_v4", run_dir / "compute_cohere_v4.log")

    print(f"\n📊 Dataset: {dataset_name}")

    # Check for Cohere v4 reranked files
    rerank_dir = run_dir / "rerank"
    cohere_v4_files = {
        "cohere-v4": rerank_dir / "reranked_cohere-v4.jsonl",
        "cohere-v4-fast": rerank_dir / "reranked_cohere-v4-fast.jsonl"
    }

    # Find which models need metrics
    metrics_file = run_dir / "evaluation" / "metrics.csv"

    if not metrics_file.exists():
        print(f"  ⚠️  No metrics.csv found, skipping")
        return

    # Read existing metrics to see what's already there
    existing_models = set()
    with open(metrics_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Handle both "model" and "reranker" column names
            model_name = row.get('model') or row.get('reranker')
            if model_name:
                existing_models.add(model_name)

    # Find models that need metrics
    models_to_evaluate = []
    for model_name, reranked_file in cohere_v4_files.items():
        if reranked_file.exists() and model_name not in existing_models:
            models_to_evaluate.append((model_name, reranked_file))

    if not models_to_evaluate:
        print(f"  ✅ All Cohere v4 models already have metrics")
        return

    print(f"  Found {len(models_to_evaluate)} new models to evaluate")

    # Load qrels
    dataset_config = config["dataset"].copy()
    dataset_config["name"] = dataset_name
    dataset_config["base_path"] = f"datasets/{dataset_name}"
    qrels_path = Path(dataset_config["base_path"]) / dataset_config.get("qrels_file", "qrels/test.tsv")

    if not qrels_path.exists():
        print(f"  ⚠️  Qrels not found at {qrels_path}, skipping")
        return

    qrels = _load_qrels(str(qrels_path))
    metrics = config["evaluation"]["metrics"]

    # Evaluate each new model
    new_results = []
    for model_name, reranked_file in models_to_evaluate:
        print(f"  ⏳ Evaluating {model_name}...")

        metrics_dict = _evaluate_reranker(str(reranked_file), qrels, metrics)

        print(f"      {', '.join([f'{m}={metrics_dict[m]:.4f}' for m in metrics])}")

        new_results.append((model_name, metrics_dict))

    # Append to metrics.csv
    with open(metrics_file, 'a', newline='') as f:
        writer = csv.writer(f)
        for model_name, metrics_dict in new_results:
            row = [model_name] + [f"{metrics_dict[m]:.4f}" for m in metrics]
            writer.writerow(row)

    print(f"  ✅ Added {len(new_results)} models to {metrics_file}")


def main():
    # Load config
    config_path = Path("config.yaml")
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return 1

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Find all dataset runs
    runs_dir = Path("runs")
    if not runs_dir.exists():
        print(f"❌ No runs directory found")
        return 1

    datasets = [d for d in runs_dir.iterdir() if d.is_dir()]

    if not datasets:
        print(f"❌ No dataset directories found in {runs_dir}")
        return 1

    print(f"\n🔄 Computing metrics for Cohere v4 models only...")
    print(f"Found {len(datasets)} datasets")

    for dataset_dir in sorted(datasets):
        dataset_name = dataset_dir.name

        # Get latest run for this dataset
        run_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
        if not run_dirs:
            continue

        latest_run = sorted(run_dirs)[-1]
        compute_and_append_metrics(dataset_name, latest_run, config)

    print(f"\n✅ Done! Cohere v4 metrics added to all datasets")
    print(f"\nNext step: Run 'python aggregate_all_results.py' to update results.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
