#!/usr/bin/env python3
"""
Aggregate all results from judgments, evaluations, and latencies into a comprehensive JSON.
"""

import json
import csv
import numpy as np
import yaml
from pathlib import Path
from collections import defaultdict


def load_judgments(judgments_file):
    """Load judgments and extract ELO, wins/losses/ties by dataset"""
    elo_ratings = {}
    win_loss_tie = defaultdict(lambda: {'wins': 0, 'losses': 0, 'ties': 0})
    comparisons = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'losses': 0, 'ties': 0}))
    elo_history = defaultdict(list)

    with open(judgments_file, 'r') as f:
        for line in f:
            data = json.loads(line)

            # Track final ELO
            model_x = data['model_x']
            model_y = data['model_y']
            elo_ratings[model_x] = data['model_x_elo_after']
            elo_ratings[model_y] = data['model_y_elo_after']

            # Track ELO history for std calculation
            elo_history[model_x].append(data['model_x_elo_after'])
            elo_history[model_y].append(data['model_y_elo_after'])

            # Track wins/losses/ties
            winner = data['winner']
            if winner == 'TIE' or winner == 'tie' or winner.lower() == 'tie':
                win_loss_tie[model_x]['ties'] += 1
                win_loss_tie[model_y]['ties'] += 1
                comparisons[model_x][model_y]['ties'] += 1
                comparisons[model_y][model_x]['ties'] += 1
            elif winner == model_x:
                win_loss_tie[model_x]['wins'] += 1
                win_loss_tie[model_y]['losses'] += 1
                comparisons[model_x][model_y]['wins'] += 1
                comparisons[model_y][model_x]['losses'] += 1
            elif winner == model_y:
                win_loss_tie[model_y]['wins'] += 1
                win_loss_tie[model_x]['losses'] += 1
                comparisons[model_y][model_x]['wins'] += 1
                comparisons[model_x][model_y]['losses'] += 1

    # Calculate ELO std per model
    elo_std = {}
    for model, history in elo_history.items():
        if len(history) > 1:
            elo_std[model] = np.std(history)
        else:
            elo_std[model] = 0.0

    return elo_ratings, win_loss_tie, comparisons, elo_std


def load_metrics(metrics_file):
    """Load evaluation metrics from CSV"""
    metrics = {}
    if not metrics_file.exists():
        return metrics

    try:
        with open(metrics_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Support both 'reranker' and 'model' column names
                if 'reranker' in row:
                    reranker = row['reranker']
                elif 'model' in row:
                    reranker = row['model']
                else:
                    continue

                metrics[reranker] = {
                    'ndcg@5': float(row.get('ndcg@5', 0)),
                    'ndcg@10': float(row.get('ndcg@10', 0)),
                    'recall@5': float(row.get('recall@5', 0)),
                    'recall@10': float(row.get('recall@10', 0))
                }
    except (KeyError, ValueError) as e:
        print(f"  ⚠️  Error loading metrics from {metrics_file}: {e}")
        return {}

    return metrics


def load_latency(latency_file):
    """Load latency data from CSV"""
    if not latency_file.exists():
        return None

    latencies = []
    try:
        with open(latency_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'latency_ms' in row:
                    latencies.append(float(row['latency_ms']))
    except (KeyError, ValueError):
        return None

    if not latencies:
        return None

    return {
        'mean_ms': np.mean(latencies),
        'p50_ms': np.percentile(latencies, 50),
        'p90_ms': np.percentile(latencies, 90)
    }


def process_dataset(dataset_name, run_dir):
    """Process a single dataset"""
    judgments_file = run_dir / "llm_judge" / "judgments.jsonl"
    metrics_file = run_dir / "evaluation" / "metrics.csv"
    rerank_dir = run_dir / "rerank"

    result = {}

    # Load judgments
    if judgments_file.exists():
        elo_ratings, win_loss_tie, comparisons, elo_std = load_judgments(judgments_file)
        result['elo_ratings'] = elo_ratings
        result['win_loss_tie'] = win_loss_tie
        result['comparisons'] = comparisons
        result['elo_std'] = elo_std

    # Load metrics
    if metrics_file.exists():
        result['metrics'] = load_metrics(metrics_file)

    # Load latencies
    result['latencies'] = {}
    if rerank_dir.exists():
        for latency_file in rerank_dir.glob("latency_*.csv"):
            reranker = latency_file.stem.replace("latency_", "")
            latency_data = load_latency(latency_file)
            if latency_data:
                result['latencies'][reranker] = latency_data

    return result


def load_display_name_mapping():
    """Load mapping from reranker names to model names from model-info.json and config.yaml"""
    # Load model-info.json
    model_info_file = Path("model-info.json")
    model_name_to_name = {}
    
    if model_info_file.exists():
        with open(model_info_file, 'r') as f:
            model_info = json.load(f)
            for model in model_info:
                # Use the "name" field, not "display_name"
                model_name_to_name[model['name']] = model['name']
    
    # Load config.yaml to map reranker names to model names
    config_file = Path("config.yaml")
    reranker_to_model = {}
    
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            for reranker in config.get('rerankers', []):
                reranker_name = reranker['name']
                model_name = reranker['model']
                reranker_to_model[reranker_name] = model_name
    
    # Create final mapping: reranker_name -> model name (from model-info.json "name" field)
    reranker_to_display = {}
    
    # Manual mapping for special cases
    special_mappings = {
        'cohere': 'cohere-rerank-v3.5',  # rerank-english-v3.0 -> cohere-rerank-v3.5
        'cohere-v4': 'rerank-v4.0-pro',
        'cohere-v4-fast': 'rerank-v4.0-fast',
        'jina': 'jina-reranker-v2-base-multilingual',
        'voyage': 'voyage-rerank-2.5',  # rerank-2.5 -> voyage-rerank-2.5
        'voyage-light': 'voyage-rerank-2.5-lite',  # rerank-2.5-lite -> voyage-rerank-2.5-lite
        'zerank1': 'zerank-1',
        'zerank-light': 'zerank-1-small',
        'zerank2': 'zerank-2',
        'ctxl': 'ctxl-rerank-v2-instruct-multilingual',
        'bge-m3': 'bge-reranker-v2-m3',  # replicate model string contains this
    }
    
    for reranker_name, model_name in reranker_to_model.items():
        # Check special mappings first
        if reranker_name in special_mappings:
            model_key = special_mappings[reranker_name]
            if model_key in model_name_to_name:
                reranker_to_display[reranker_name] = model_name_to_name[model_key]
            else:
                reranker_to_display[reranker_name] = reranker_name
        # Try exact match
        elif model_name in model_name_to_name:
            reranker_to_display[reranker_name] = model_name_to_name[model_name]
        else:
            # Try partial matches for replicate models
            found = False
            for model_key, name_value in model_name_to_name.items():
                if model_key in model_name:
                    reranker_to_display[reranker_name] = name_value
                    found = True
                    break
            
            if not found:
                # Fallback: use reranker name as-is
                reranker_to_display[reranker_name] = reranker_name
    
    return reranker_to_display


def aggregate_results():
    """Aggregate all results across datasets"""
    runs_dir = Path("runs")
    
    # Load display name mapping
    display_name_map = load_display_name_mapping()

    # Dataset name mapping
    dataset_mapping = {
        'fiqa_small': 'FiQa',
        'scifact': 'SciFact',
        'pg': 'PG',
        'business-reports': 'business_reports',
        'msmarco': 'MSMARCO',
        'dbpedia': 'DBPedia'
    }

    # Datasets to exclude
    excluded_datasets = {'SciFact'}

    # Collect data by dataset
    datasets_data = {}
    all_rerankers = set()

    for dataset_dir in runs_dir.iterdir():
        if not dataset_dir.is_dir():
            continue

        dataset_name = dataset_dir.name

        # Find latest run
        run_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
        if not run_dirs:
            continue

        latest_run = run_dirs[-1]

        # Process dataset
        data = process_dataset(dataset_name, latest_run)
        if data:
            display_name = dataset_mapping.get(dataset_name, dataset_name)
            # Skip excluded datasets
            if display_name in excluded_datasets:
                print(f"  ⏭️  Skipping excluded dataset: {display_name}")
                continue
            datasets_data[display_name] = data
            if 'elo_ratings' in data:
                all_rerankers.update(data['elo_ratings'].keys())

    # Build reranker-centric view
    results = []

    for reranker in sorted(all_rerankers):
        # Use display name if available, otherwise use reranker name
        display_name = display_name_map.get(reranker, reranker)
        
        reranker_data = {
            'name': display_name,
            'overall': {},
            'by_dataset': {},
            'comparisons': {}
        }

        # Aggregate overall stats
        all_elos = []
        total_wins = 0
        total_losses = 0
        total_ties = 0
        all_latencies = []
        all_ndcg10 = []

        # Aggregate comparisons across datasets
        overall_comparisons = defaultdict(lambda: {'wins': 0, 'losses': 0, 'ties': 0})

        for dataset_name, data in datasets_data.items():
            dataset_info = {}

            # ELO rating
            if 'elo_ratings' in data and reranker in data['elo_ratings']:
                elo = data['elo_ratings'][reranker]
                all_elos.append(elo)
                dataset_info['elo'] = round(elo, 2)

                # ELO std per dataset
                if 'elo_std' in data and reranker in data['elo_std']:
                    dataset_info['elo_std'] = round(data['elo_std'][reranker], 2)

            # Win/loss/tie
            if 'win_loss_tie' in data and reranker in data['win_loss_tie']:
                wlt = data['win_loss_tie'][reranker]
                dataset_info['wins'] = wlt['wins']
                dataset_info['losses'] = wlt['losses']
                dataset_info['ties'] = wlt['ties']
                total_wins += wlt['wins']
                total_losses += wlt['losses']
                total_ties += wlt['ties']

                total = wlt['wins'] + wlt['losses'] + wlt['ties']
                if total > 0:
                    dataset_info['win_rate'] = round(wlt['wins'] / total, 4)

            # Metrics
            if 'metrics' in data and reranker in data['metrics']:
                metrics = data['metrics'][reranker]
                dataset_info['metrics'] = {
                    'ndcg@5': round(metrics['ndcg@5'], 4),
                    'ndcg@10': round(metrics['ndcg@10'], 4),
                    'recall@5': round(metrics['recall@5'], 4),
                    'recall@10': round(metrics['recall@10'], 4)
                }
                all_ndcg10.append(metrics['ndcg@10'])

            # Latency
            if 'latencies' in data and reranker in data['latencies']:
                latency = data['latencies'][reranker]
                dataset_info['latency'] = {
                    'mean_ms': round(latency['mean_ms'], 2),
                    'p50_ms': round(latency['p50_ms'], 2),
                    'p90_ms': round(latency['p90_ms'], 2)
                }
                all_latencies.append(latency['mean_ms'])

            # Comparisons (for this dataset)
            if 'comparisons' in data and reranker in data['comparisons']:
                for opponent, comp_stats in data['comparisons'][reranker].items():
                    overall_comparisons[opponent]['wins'] += comp_stats['wins']
                    overall_comparisons[opponent]['losses'] += comp_stats['losses']
                    overall_comparisons[opponent]['ties'] += comp_stats['ties']

            if dataset_info:
                reranker_data['by_dataset'][dataset_name] = dataset_info

        # Calculate overall stats
        if all_elos:
            reranker_data['overall']['elo'] = round(np.mean(all_elos), 2)
            reranker_data['overall']['elo_std'] = round(np.std(all_elos), 2)

        if total_wins + total_losses + total_ties > 0:
            reranker_data['overall']['wins'] = total_wins
            reranker_data['overall']['losses'] = total_losses
            reranker_data['overall']['ties'] = total_ties
            reranker_data['overall']['win_rate'] = round(total_wins / (total_wins + total_losses + total_ties), 3)
            reranker_data['overall']['total_judgments'] = total_wins + total_losses + total_ties

        if all_latencies:
            reranker_data['overall']['avg_latency_ms'] = round(np.mean(all_latencies), 2)

        if all_ndcg10:
            reranker_data['overall']['avg_ndcg_10'] = round(np.mean(all_ndcg10), 5)

        # Add comparison stats
        # Use display names ONLY for cohere-v4 models to match benchmarks.json format
        for opponent, comp_stats in overall_comparisons.items():
            # Special case: cohere-v4 models use display names in comparisons
            if opponent in ['cohere-v4', 'cohere-v4-fast']:
                opponent_key = display_name_map.get(opponent, opponent)
            else:
                opponent_key = opponent

            total = comp_stats['wins'] + comp_stats['losses'] + comp_stats['ties']
            reranker_data['comparisons'][opponent_key] = {
                'wins': comp_stats['wins'],
                'losses': comp_stats['losses'],
                'ties': comp_stats['ties'],
                'total': total,
                'win_rate': round(comp_stats['wins'] / total, 4) if total > 0 else 0
            }

        results.append(reranker_data)

    # Sort by overall ELO
    results.sort(key=lambda x: x['overall'].get('elo', 0), reverse=True)

    return results


def main():
    print("🔄 Aggregating all results...")

    results = aggregate_results()

    output_file = Path("results_all_datasets.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✅ Saved comprehensive results to: {output_file}")
    print(f"\n📊 Summary:")
    print(f"   - Total rerankers: {len(results)}")
    print(f"   - Datasets included: {len(set(ds for r in results for ds in r['by_dataset'].keys()))}")

    print(f"\n🏆 Top 5 by overall ELO:")
    for i, reranker in enumerate(results[:5], 1):
        elo = reranker['overall'].get('elo', 0)
        win_rate = reranker['overall'].get('win_rate', 0)
        print(f"   {i}. {reranker['name']}: ELO={elo:.2f}, Win Rate={win_rate:.1%}")


if __name__ == "__main__":
    main()
