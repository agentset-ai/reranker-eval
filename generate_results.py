#!/usr/bin/env python3
"""
Generate unified results.json from run directories
Includes all 8 rerankers with ELO scores, metrics, and latencies
"""

import json
import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import statistics

# Model information
MODEL_INFO = {
    'ctxl': {
        'display_name': 'Contextual AI Rerank v2 Instruct',
        'provider': 'Contextual AI',
        'license': 'Proprietary'
    },
    'bge-m3': {
        'display_name': 'BAAI BGE Reranker v2 M3',
        'provider': 'BAAI (via Replicate)',
        'license': 'MIT'
    },
    'voyage': {
        'display_name': 'Voyage Rerank 2.5',
        'provider': 'Voyage AI',
        'license': 'Proprietary'
    },
    'voyage-light': {
        'display_name': 'Voyage Rerank 2.5 Lite',
        'provider': 'Voyage AI',
        'license': 'Proprietary'
    },
    'zerank1': {
        'display_name': 'Zerank 1',
        'provider': 'ZeroTruth AI',
        'license': 'Proprietary'
    },
    'zerank-light': {
        'display_name': 'Zerank 1 Small',
        'provider': 'ZeroTruth AI',
        'license': 'Proprietary'
    },
    'jina': {
        'display_name': 'Jina Reranker v2 Base Multilingual',
        'provider': 'Jina AI',
        'license': 'Apache 2.0'
    },
    'cohere': {
        'display_name': 'Cohere Rerank 3.5',
        'provider': 'Cohere',
        'license': 'Proprietary'
    }
}

# Dataset configurations - BGE run directories and fallback directories
DATASETS = {
    'fiqa_small': {
        'run_dir': 'runs/fiqa_small/20251105_112147',  # BGE run (has LLM judge for all 8 + BGE latency/metrics)
        'fallback_run_dirs': [
            'runs/fiqa_small/20251105_101717',  # Second run (has 7 old rerankers' latency/metrics)
            'runs/fiqa_small/20251029_125030'   # Original run (has zerank1, voyage latency)
        ],
        'display_name': 'FiQA Small',
        'result_key': 'BEIR/fiqa'  # Key to use in results.json
    },
    'scifact': {
        'run_dir': 'runs/scifact/20251105_113705',  # BGE run
        'fallback_run_dirs': [
            'runs/scifact/20251105_102950',  # Second run
            'runs/scifact/20251102_145415'   # Original run (has zerank1, voyage latency)
        ],
        'display_name': 'SciFact',
        'result_key': 'BEIR/scifact'
    },
    'pg_essays_clean': {
        'run_dir': 'runs/pg_essays_clean/20251105_115243',  # BGE run
        'fallback_run_dirs': [
            'runs/pg_essays_clean/20251105_104907',  # Second run
            'runs/pg_essays_clean/20251102_131656'   # Original run (has zerank1, voyage latency)
        ],
        'display_name': 'PG Essays',
        'result_key': 'PG'
    },
    'business_reports': {
        'run_dir': 'runs/business_reports/20251105_145143',  # Full run with all 8 rerankers
        'display_name': 'Business Reports',
        'result_key': 'business_reports'
    },
    'nfcorpus': {
        'run_dir': 'runs/nfcorpus/20251105_181540',  # Medical/nutrition facts dataset
        'display_name': 'NFCorpus',
        'result_key': 'BEIR/nfcorpus'
    },
    'msmarco': {
        'run_dir': 'runs/msmarco/20251105_195820',  # Web search, short passages dataset
        'display_name': 'MSMARCO',
        'result_key': 'BEIR/msmarco'
    },
    'dbpedia': {
        'run_dir': 'runs/dbpedia/20251105_210320',  # Entity-based Q&A, short factual passages
        'display_name': 'DBPedia',
        'result_key': 'BEIR/dbpedia-entity'
    }
}

def load_judgments(judgments_file: Path) -> List[Dict]:
    """Load LLM judgments from JSONL file"""
    judgments = []
    if not judgments_file.exists():
        print(f"Warning: Judgments file not found: {judgments_file}")
        return judgments

    with open(judgments_file) as f:
        for line in f:
            judgments.append(json.loads(line))
    return judgments

def load_latencies(run_dir: Path, model_name: str, fallback_run_dirs: list = None) -> Dict:
    """Load latency statistics from CSV with fallback support"""
    # Try primary directory first
    latency_file = run_dir / 'rerank' / f'latency_{model_name}.csv'

    # If not found and fallbacks are provided, try each fallback
    if not latency_file.exists() and fallback_run_dirs:
        for fallback_dir in fallback_run_dirs:
            fallback_file = Path(fallback_dir) / 'rerank' / f'latency_{model_name}.csv'
            if fallback_file.exists():
                latency_file = fallback_file
                break

    if not latency_file.exists():
        return {}

    latencies = []
    with open(latency_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Handle both field name formats
            if 'latency_ms' in row:
                latencies.append(float(row['latency_ms']))
            elif 'latency_seconds' in row:
                latencies.append(float(row['latency_seconds']) * 1000)  # Convert to ms

    if not latencies:
        return {}

    latencies.sort()
    return {
        'mean_ms': round(statistics.mean(latencies), 2),
        'p50_ms': round(latencies[len(latencies) // 2], 2),
        'p90_ms': round(latencies[int(len(latencies) * 0.9)], 2)
    }

def load_metrics(run_dir: Path, model_name: str, fallback_run_dirs: list = None) -> Dict:
    """Load metrics from evaluation CSV with fallback support"""
    # Try primary directory first
    metrics_file = run_dir / 'evaluation' / 'metrics.csv'
    if metrics_file.exists():
        with open(metrics_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['model'] == model_name:
                    return {
                        'ndcg@5': round(float(row.get('ndcg@5', 0)), 4),
                        'ndcg@10': round(float(row.get('ndcg@10', 0)), 4),
                        'recall@5': round(float(row.get('recall@5', 0)), 4),
                        'recall@10': round(float(row.get('recall@10', 0)), 4)
                    }

    # If not found and fallbacks are provided, try each fallback
    if fallback_run_dirs:
        for fallback_dir in fallback_run_dirs:
            metrics_file = Path(fallback_dir) / 'evaluation' / 'metrics.csv'
            if metrics_file.exists():
                with open(metrics_file) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row['model'] == model_name:
                            return {
                                'ndcg@5': round(float(row.get('ndcg@5', 0)), 4),
                                'ndcg@10': round(float(row.get('ndcg@10', 0)), 4),
                                'recall@5': round(float(row.get('recall@5', 0)), 4),
                                'recall@10': round(float(row.get('recall@10', 0)), 4)
                            }

    return {}

def calculate_elo_ratings(judgments: List[Dict], initial_rating: int = 1500, k_factor: int = 32) -> Dict[str, float]:
    """Calculate ELO ratings from judgments"""
    ratings = defaultdict(lambda: initial_rating)

    for judgment in judgments:
        # Handle both field name formats
        model_a = judgment.get('a_model', judgment.get('model_a'))
        model_b = judgment.get('b_model', judgment.get('model_b'))
        winner = judgment['winner']

        # Current ratings
        rating_a = ratings[model_a]
        rating_b = ratings[model_b]

        # Expected scores
        expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
        expected_b = 1 / (1 + 10 ** ((rating_a - rating_b) / 400))

        # Actual scores
        if winner == model_a:
            score_a, score_b = 1.0, 0.0
        elif winner == model_b:
            score_a, score_b = 0.0, 1.0
        else:  # tie
            score_a, score_b = 0.5, 0.5

        # Update ratings
        ratings[model_a] += k_factor * (score_a - expected_a)
        ratings[model_b] += k_factor * (score_b - expected_b)

    return dict(ratings)

def calculate_elo_std(judgments: List[Dict], model: str, k_factor: int = 32) -> float:
    """Estimate ELO rating standard deviation using bootstrap"""
    import random

    # Simple bootstrap estimation
    n_bootstrap = 100
    elos = []

    for _ in range(n_bootstrap):
        sample = random.choices(judgments, k=len(judgments))
        ratings = calculate_elo_ratings(sample, k_factor=k_factor)
        if model in ratings:
            elos.append(ratings[model])

    return round(statistics.stdev(elos), 2) if len(elos) > 1 else 0.0

def calculate_comparisons(judgments: List[Dict], model: str) -> Dict:
    """Calculate head-to-head comparisons for a model"""
    comparisons = defaultdict(lambda: {'wins': 0, 'losses': 0, 'ties': 0})

    for judgment in judgments:
        # Handle both field name formats
        model_a = judgment.get('a_model', judgment.get('model_a'))
        model_b = judgment.get('b_model', judgment.get('model_b'))
        winner = judgment['winner']

        if model_a == model:
            opponent = model_b
            if winner == model:
                comparisons[opponent]['wins'] += 1
            elif winner == opponent:
                comparisons[opponent]['losses'] += 1
            else:
                comparisons[opponent]['ties'] += 1
        elif model_b == model:
            opponent = model_a
            if winner == model:
                comparisons[opponent]['wins'] += 1
            elif winner == opponent:
                comparisons[opponent]['losses'] += 1
            else:
                comparisons[opponent]['ties'] += 1

    # Format output
    result = {}
    for opponent, stats in comparisons.items():
        total = stats['wins'] + stats['losses'] + stats['ties']
        result[opponent] = {
            'wins': stats['wins'],
            'losses': stats['losses'],
            'ties': stats['ties'],
            'total': total,
            'win_rate': round(stats['wins'] / total, 4) if total > 0 else 0.0
        }

    return result

def generate_results():
    """Generate unified results.json"""
    results = []

    # Get all models from the first dataset
    first_dataset = list(DATASETS.values())[0]
    run_dir = Path(first_dataset['run_dir'])
    rerank_dir = run_dir / 'rerank'

    models = []
    for f in rerank_dir.glob('reranked_*.jsonl'):
        model_name = f.stem.replace('reranked_', '')
        if model_name in MODEL_INFO:
            models.append(model_name)

    print(f"Found {len(models)} models: {', '.join(models)}")

    for model in models:
        print(f"\nProcessing {model}...")

        model_result = {
            'name': model,
            'display_name': MODEL_INFO[model]['display_name'],
            'provider': MODEL_INFO[model]['provider'],
            'license': MODEL_INFO[model]['license'],
            'overall': {},
            'by_dataset': {},
            'comparisons': {}
        }

        # Collect data across all datasets
        all_judgments = []
        total_wins = 0
        total_losses = 0
        total_ties = 0
        all_latencies = []
        all_ndcg10 = []

        for dataset_key, dataset_config in DATASETS.items():
            run_dir = Path(dataset_config['run_dir'])
            # Support both old fallback_run_dir and new fallback_run_dirs
            fallback_run_dirs = dataset_config.get('fallback_run_dirs', [])
            if not fallback_run_dirs and 'fallback_run_dir' in dataset_config:
                fallback_run_dirs = [dataset_config['fallback_run_dir']]
            judgments_file = run_dir / 'llm_judge' / 'judgments.jsonl'

            # Load judgments for this dataset
            judgments = load_judgments(judgments_file)
            dataset_judgments = [j for j in judgments if model in [j.get('a_model', j.get('model_a')), j.get('b_model', j.get('model_b'))]]
            all_judgments.extend(dataset_judgments)

            # Calculate dataset-specific stats
            wins = sum(1 for j in dataset_judgments if j['winner'] == model)
            losses = sum(1 for j in dataset_judgments if j['winner'] not in [model, 'tie'] and not j.get('tie', False))
            ties = sum(1 for j in dataset_judgments if j['winner'] == 'tie' or j.get('tie', False))
            total = len(dataset_judgments)

            total_wins += wins
            total_losses += losses
            total_ties += ties

            # Calculate ELO for this dataset
            elo_ratings = calculate_elo_ratings(judgments)
            elo = elo_ratings.get(model, 1500)
            elo_std = calculate_elo_std(judgments, model)

            # Load latencies with fallback
            latency = load_latencies(run_dir, model, fallback_run_dirs)
            if latency:
                all_latencies.extend([latency['mean_ms']])

            # Load metrics with fallback
            metrics = load_metrics(run_dir, model, fallback_run_dirs)

            # Collect ndcg@10 for average (only if non-zero)
            if metrics and metrics.get('ndcg@10', 0) > 0:
                all_ndcg10.append(metrics['ndcg@10'])

            # Build dataset result
            dataset_result = {
                'elo': round(elo, 2),
                'elo_std': elo_std,
                'wins': wins,
                'losses': losses,
                'ties': ties,
                'win_rate': round(wins / total, 4) if total > 0 else 0.0
            }

            if metrics:
                dataset_result['metrics'] = metrics

            if latency:
                dataset_result['latency'] = latency

            # Use result_key if available, otherwise use dataset_key
            result_key = dataset_config.get('result_key', dataset_key)
            model_result['by_dataset'][result_key] = dataset_result

        # Calculate overall stats
        overall_elo_ratings = calculate_elo_ratings(all_judgments)
        overall_elo = overall_elo_ratings.get(model, 1500)
        overall_elo_std = calculate_elo_std(all_judgments, model)

        total_judgments = total_wins + total_losses + total_ties

        model_result['overall'] = {
            'elo': round(overall_elo, 2),
            'elo_std': overall_elo_std,
            'wins': total_wins,
            'losses': total_losses,
            'ties': total_ties,
            'win_rate': round(total_wins / total_judgments, 4) if total_judgments > 0 else 0.0,
            'total_judgments': total_judgments
        }

        if all_latencies:
            model_result['overall']['avg_latency_ms'] = round(statistics.mean(all_latencies), 2)

        if all_ndcg10:
            model_result['overall']['avg_ndcg_10'] = round(statistics.mean(all_ndcg10), 5)

        # Calculate head-to-head comparisons (overall)
        model_result['comparisons'] = calculate_comparisons(all_judgments, model)

        results.append(model_result)

    # Sort by overall ELO (descending)
    results.sort(key=lambda x: x['overall']['elo'], reverse=True)

    # Write results
    output_file = Path('results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results written to {output_file}")
    print(f"   Total models: {len(results)}")
    print(f"\nELO Leaderboard:")
    for i, result in enumerate(results, 1):
        print(f"   {i}. {result['display_name']}: {result['overall']['elo']:.1f} ± {result['overall']['elo_std']:.1f}")

if __name__ == '__main__':
    generate_results()