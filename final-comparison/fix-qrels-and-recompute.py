#!/usr/bin/env python3
"""
Fix qrels file paths and recompute metrics for datasets with mismatches.
"""

import json
import csv
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.stages.evaluate import _load_qrels, _evaluate_reranker


# Mapping of dataset to correct qrels file
QRELS_MAPPING = {
    'dbpedia': 'qrels/dev.tsv',  # queries are in dev, not test
    'msmarco': 'qrels/train.tsv',  # queries are in train, not test
    'business-reports': None,  # no matching qrels found
    'pg': None,  # no matching qrels found
}


def find_matching_qrels(dataset_path: Path, query_ids: set) -> Path:
    """Find which qrels file contains the query IDs."""
    qrels_dir = dataset_path / "qrels"
    if not qrels_dir.exists():
        return None

    for qrels_file in qrels_dir.glob("*.tsv"):
        # Check if this qrels file has any of our queries
        with open(qrels_file) as f:
            next(f)  # skip header
            for line in f:
                qid = line.split('\t')[0]
                if qid in query_ids:
                    return qrels_file

    return None


def recompute_metrics_for_dataset(dataset_name: str, run_dir: Path):
    """Recompute metrics using the correct qrels file."""

    print(f"\n📊 {dataset_name}")

    # Load queries to get query IDs
    dataset_path = Path(f"datasets/{dataset_name}")
    queries_file = dataset_path / "queries.jsonl"

    if not queries_file.exists():
        print(f"  ⚠️  No queries file found")
        return

    query_ids = set()
    with open(queries_file) as f:
        for line in f:
            data = json.loads(line)
            query_ids.add(data['_id'])

    print(f"  Found {len(query_ids)} queries")

    # Find correct qrels file
    if dataset_name in QRELS_MAPPING and QRELS_MAPPING[dataset_name] is not None:
        qrels_path = dataset_path / QRELS_MAPPING[dataset_name]
    else:
        qrels_path = find_matching_qrels(dataset_path, query_ids)

    if not qrels_path or not qrels_path.exists():
        print(f"  ❌ No matching qrels file found - skipping")
        return

    print(f"  Using qrels: {qrels_path.relative_to(dataset_path)}")

    # Load qrels
    qrels = _load_qrels(str(qrels_path))

    # Verify overlap
    qrels_query_ids = set(qrels.keys())
    overlap = query_ids & qrels_query_ids
    print(f"  Overlap: {len(overlap)}/{len(query_ids)} queries have relevance judgments")

    if len(overlap) == 0:
        print(f"  ⚠️  No overlap - skipping")
        return

    # Get all reranked files
    rerank_dir = run_dir / "rerank"
    if not rerank_dir.exists():
        print(f"  ⚠️  No rerank directory")
        return

    reranked_files = list(rerank_dir.glob("reranked_*.jsonl"))

    if not reranked_files:
        print(f"  ⚠️  No reranked files")
        return

    print(f"  Evaluating {len(reranked_files)} rerankers...")

    # Evaluate each reranker
    metrics_list = ['ndcg@5', 'ndcg@10', 'recall@5', 'recall@10']
    results = {}

    for reranked_file in reranked_files:
        reranker_name = reranked_file.stem.replace("reranked_", "")
        metrics = _evaluate_reranker(str(reranked_file), qrels, metrics_list)
        results[reranker_name] = metrics

    # Save updated metrics
    metrics_file = run_dir / "evaluation" / "metrics.csv"
    metrics_file.parent.mkdir(parents=True, exist_ok=True)

    # Determine column name (model or reranker)
    column_name = "model"  # default
    if metrics_file.exists():
        with open(metrics_file) as f:
            first_line = f.readline().strip()
            if first_line.startswith("reranker,"):
                column_name = "reranker"

    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [column_name] + metrics_list
        writer.writerow(header)
        for reranker_name, metrics in results.items():
            row = [reranker_name] + [f"{metrics[m]:.4f}" for m in metrics_list]
            writer.writerow(row)

    print(f"  ✅ Updated metrics.csv")

    # Show summary
    non_zero = sum(1 for m in results.values() if m['ndcg@10'] > 0)
    print(f"  📈 {non_zero}/{len(results)} rerankers have non-zero metrics")


def main():
    print("🔧 Fixing qrels paths and recomputing metrics...\n")

    # Datasets to fix
    datasets_to_fix = ['dbpedia', 'msmarco', 'business-reports', 'pg']

    runs_dir = Path("runs")

    for dataset_name in datasets_to_fix:
        dataset_dir = runs_dir / dataset_name

        if not dataset_dir.exists():
            print(f"⚠️  No runs found for {dataset_name}")
            continue

        # Get latest run
        run_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
        if not run_dirs:
            print(f"⚠️  No run directories for {dataset_name}")
            continue

        latest_run = run_dirs[-1]
        recompute_metrics_for_dataset(dataset_name, latest_run)

    print("\n✅ Done! Now run 'python3 aggregate_all_results.py' to update results.json")


if __name__ == "__main__":
    main()
