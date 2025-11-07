"""
Evaluation stage: Calculate NDCG and Recall metrics
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path
from typing import Dict, List
import logging

from ..config import Config
from ..paths import RunPaths


def _dcg_at_k(relevance_scores: List[int], k: int) -> float:
    """Calculate DCG at rank k"""
    relevance_scores = np.array(relevance_scores[:k])
    if len(relevance_scores) == 0:
        return 0.0
    gains = 2**relevance_scores - 1
    discounts = np.log2(np.arange(len(relevance_scores)) + 2)
    return np.sum(gains / discounts)


def _ndcg_at_k(relevance_scores: List[int], k: int) -> float:
    """Calculate NDCG at rank k"""
    dcg = _dcg_at_k(relevance_scores, k)
    if dcg == 0:
        return 0.0
    ideal_relevance = sorted(relevance_scores, reverse=True)
    ideal_dcg = _dcg_at_k(ideal_relevance, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg / ideal_dcg


def _recall_at_k(relevance_scores: List[int], k: int, num_relevant: int) -> float:
    """Calculate Recall at rank k"""
    if num_relevant == 0:
        return 0.0
    retrieved = sum(relevance_scores[:k])
    return retrieved / num_relevant


def _load_qrels(qrels_file: str) -> Dict[str, Dict[str, int]]:
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


def _evaluate_reranker(reranked_file: str, qrels: Dict[str, Dict[str, int]], metrics: List[str]) -> Dict:
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


def evaluate_stage(config: Config, paths: RunPaths, logger: logging.Logger) -> Dict:
    """
    Evaluate reranked results using ground truth labels
    
    Args:
        config: Pipeline configuration
        paths: Run paths manager
        logger: Logger instance
    
    Returns:
        Dictionary with stage results metadata
    """
    logger.info("Starting evaluation stage...")
    
    # Check if results already exist
    metrics_file = paths.get_evaluation_metrics_file()
    if metrics_file.exists() and config.pipeline.skip_if_exists:
        logger.info(f"Evaluation results already exist at {metrics_file}, skipping...")
        return {'status': 'skipped', 'metrics_file': str(metrics_file)}
    
    # Load qrels
    logger.info(f"Loading relevance judgments from {config.dataset.qrels_path}")
    qrels = _load_qrels(config.dataset.qrels_path)
    logger.info(f"Loaded relevance judgments for {len(qrels)} queries")
    
    # Evaluate each reranker
    logger.info(f"Evaluating {len(config.rerankers)} rerankers...")
    print(f"   📊 Evaluating {len(config.rerankers)} rerankers...")
    evaluation_results = {}
    model_names = {}
    
    for reranker_idx, reranker in enumerate(config.rerankers, 1):
        reranked_file = paths.get_reranked_file(reranker.name)
        if not reranked_file.exists():
            logger.warning(f"Reranked results not found for {reranker.name}, skipping...")
            continue
        
        logger.info(f"Evaluating {reranker.name}...")
        print(f"   🔄 [{reranker_idx}/{len(config.rerankers)}] Evaluating {reranker.name}...")
        metrics = _evaluate_reranker(str(reranked_file), qrels, config.evaluation.metrics)
        evaluation_results[reranker.name] = metrics
        
        # Get model name from first result
        with open(reranked_file, 'r') as f:
            line = f.readline()
            if line:
                data = json.loads(line)
                model_names[reranker.name] = data.get('model', reranker.name)
        
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        print(f"      ✓ Metrics calculated: {', '.join([f'{m}={metrics[m]:.3f}' for m in config.evaluation.metrics])}")
    
    # Save metrics to CSV
    logger.info(f"Saving metrics to {metrics_file}")
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['model'] + config.evaluation.metrics
        writer.writerow(header)
        for reranker_name, metrics in evaluation_results.items():
            row = [reranker_name] + [f"{metrics[m]:.4f}" for m in config.evaluation.metrics]
            writer.writerow(row)
    
    # Create plots if requested
    if config.evaluation.generate_plots and evaluation_results:
        plot_file = paths.get_evaluation_plot_file()
        logger.info(f"Creating evaluation plot at {plot_file}")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Reranking Performance Comparison', fontsize=16, fontweight='bold')
        
        models = list(evaluation_results.keys())
        labels = [model_names.get(m, m) for m in models]
        
        metric_positions = {
            'ndcg@5': (0, 0),
            'ndcg@10': (0, 1),
            'recall@5': (1, 0),
            'recall@10': (1, 1)
        }
        base_color = '#4C78A8'
        highlight_color = '#E45756'
        
        for idx, metric in enumerate(config.evaluation.metrics):
            if metric in metric_positions:
                row, col = metric_positions[metric]
                values = [evaluation_results[m][metric] for m in models]
                
                # Determine colors: highlight the highest score
                bar_colors = []
                max_value = max(values)
                for v in values:
                    if v == max_value:
                        bar_colors.append(highlight_color)
                    else:
                        bar_colors.append(base_color)
                
                bars = axes[row, col].bar(labels, values, color=bar_colors)
                axes[row, col].set_title(metric.upper(), fontweight='bold')
                axes[row, col].set_ylabel('Score')
                axes[row, col].set_ylim([0, 1])
                axes[row, col].tick_params(axis='x', rotation=45)
                axes[row, col].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    result = {
        'status': 'completed',
        'metrics_file': str(metrics_file),
        'num_rerankers': len(evaluation_results),
        'metrics': config.evaluation.metrics
    }
    
    logger.info("Evaluation stage completed successfully")
    return result

