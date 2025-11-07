"""
LLM Judge stage: Compare rerankers using LLM judgments and ELO ratings
"""

import json
import random
import csv
import itertools
from pathlib import Path
from typing import Dict, List, Tuple
from openai import AzureOpenAI
import matplotlib.pyplot as plt
import logging

from ..config import Config
from ..paths import RunPaths


def _truncate_text(text: str, max_chars: int = 300) -> str:
    """Truncate text to max_chars, preserving words"""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rsplit(' ', 1)[0]
    return truncated + '...' if truncated != text[:max_chars] else truncated + '...'


def _format_ranking_for_prompt(results: List[Dict], corpus: Dict[str, str], k: int = 15) -> str:
    """Format ranking results for LLM prompt"""
    lines = []
    for i, result in enumerate(results[:k], 1):
        doc_id = result['doc_id']
        doc_text = corpus.get(doc_id, '')
        truncated = _truncate_text(doc_text, 300)
        lines.append(f"{i}. {doc_id}: {truncated}")
    return '\n'.join(lines)


def _create_judge_prompt(query_text: str, ranking_a: str, ranking_b: str, k: int = 15) -> str:
    """Create the judge prompt"""
    prompt = f"""Given a user query and a list of results coming from reranker, determine which reranker returns more relevant results. Return "A", "B" or "TIE". 

The ordered list represents the relevance of the snippet to be, the higher the more relevant

Query:
{query_text}

Ranking A (top {k}):
{ranking_a}

Ranking B (top {k}):
{ranking_b}

Answer with exactly one token: A, B, or TIE."""
    return prompt


def _judge_with_llm(prompt: str, client: AzureOpenAI, deployment_id: str) -> str:
    """Get judgment from LLM"""
    try:
        response = client.chat.completions.create(
            model=deployment_id,
            messages=[
                {"role": "system", "content": "You are an expert judge evaluating search result relevance. Respond with only one word: A, B, or TIE."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=10
        )
        judgment = response.choices[0].message.content.strip().upper()
        if 'TIE' in judgment or judgment == 'T':
            return 'TIE'
        elif 'A' in judgment:
            return 'A'
        elif 'B' in judgment:
            return 'B'
        else:
            return 'TIE'
    except Exception as e:
        logging.error(f"Error in LLM judgment: {e}")
        return 'TIE'


def _expected_score(elo_a: float, elo_b: float) -> float:
    """Calculate expected score for player A"""
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))


def _update_elo(elo_a: float, elo_b: float, score_a: float, k_factor: int) -> Tuple[float, float]:
    """Update ELO ratings for both players"""
    expected_a = _expected_score(elo_a, elo_b)
    expected_b = _expected_score(elo_b, elo_a)
    new_elo_a = elo_a + k_factor * (score_a - expected_a)
    new_elo_b = elo_b + k_factor * ((1 - score_a) - expected_b)
    return new_elo_a, new_elo_b


def _get_score_for_judgment(judgment: str, a_is_model_x: bool, model_x: str, model_y: str) -> Tuple[float, float]:
    """Convert judgment to scores for both models"""
    if judgment == 'TIE':
        return 0.5, 0.5
    elif judgment == 'A':
        if a_is_model_x:
            return 1.0, 0.0
        else:
            return 0.0, 1.0
    else:
        if a_is_model_x:
            return 0.0, 1.0
        else:
            return 1.0, 0.0


def llm_judge_stage(config: Config, paths: RunPaths, logger: logging.Logger) -> Dict:
    """
    LLM Judge stage: Compare rerankers using LLM and maintain ELO ratings
    
    Args:
        config: Pipeline configuration
        paths: Run paths manager
        logger: Logger instance
    
    Returns:
        Dictionary with stage results metadata
    """
    logger.info("Starting LLM Judge stage...")
    
    if not config.llm_judge.enabled:
        logger.info("LLM Judge is disabled, skipping...")
        return {'status': 'skipped', 'reason': 'disabled'}
    
    # Validate Azure OpenAI config
    if config.llm_judge.provider != "azure_openai":
        raise ValueError(f"Unsupported LLM judge provider: {config.llm_judge.provider}")
    
    if not all([config.llm_judge.azure_api_key, config.llm_judge.azure_resource_name, 
                config.llm_judge.azure_deployment_id]):
        raise ValueError("Azure OpenAI credentials not set in environment variables")
    
    # Initialize Azure OpenAI client
    client = AzureOpenAI(
        api_key=config.llm_judge.azure_api_key,
        api_version="2024-02-15-preview",
        azure_endpoint=f"https://{config.llm_judge.azure_resource_name}.openai.azure.com"
    )
    
    # Load queries
    logger.info(f"Loading queries from {config.dataset.queries_path}")
    queries = {}
    with open(config.dataset.queries_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            queries[data['_id']] = data['text']
    logger.info(f"Loaded {len(queries)} queries")
    
    # Load corpus
    logger.info(f"Loading corpus from {config.dataset.corpus_path}")
    corpus = {}
    with open(config.dataset.corpus_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            doc_id = data['_id']
            title = data.get('title', '')
            text = data.get('text', '')
            corpus[doc_id] = (title + ' ' + text).strip()
    logger.info(f"Loaded {len(corpus)} documents")
    
    # Load reranked results
    logger.info("Loading reranked results...")
    reranked_data = {}
    model_names = {}
    available_rerankers = []
    
    for reranker in config.rerankers:
        reranked_file = paths.get_reranked_file(reranker.name)
        if not reranked_file.exists():
            logger.warning(f"Reranked results not found for {reranker.name}, skipping...")
            continue
        
        reranked_data[reranker.name] = {}
        with open(reranked_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                query_id = data['query_id']
                reranked_data[reranker.name][query_id] = data
                if query_id not in model_names:
                    model_names[reranker.name] = data.get('model', reranker.name)
        
        available_rerankers.append(reranker.name)
        logger.info(f"Loaded {len(reranked_data[reranker.name])} queries for {reranker.name}")
    
    if len(available_rerankers) < 2:
        raise ValueError("Need at least 2 rerankers to compare")
    
    # Find common query IDs
    common_query_ids = set(queries.keys())
    for reranker_name in available_rerankers:
        common_query_ids &= set(reranked_data[reranker_name].keys())
    common_query_ids = sorted(list(common_query_ids))

    logger.info(f"Found {len(common_query_ids)} common queries")
    print(f"   📊 Found {len(common_query_ids)} common queries")

    # Generate all pairs
    pairs = list(itertools.combinations(available_rerankers, 2))
    logger.info(f"Will compare {len(pairs)} pairs of rerankers")
    print(f"   🔄 Will compare {len(pairs)} pairs ({len(available_rerankers)} rerankers)")
    print(f"   📈 Total comparisons: {len(common_query_ids) * len(pairs)}")

    # Initialize ELO
    elo_ratings = {r: config.llm_judge.elo_initial_rating for r in available_rerankers}
    win_loss_tie = {r: {'wins': 0, 'losses': 0, 'ties': 0} for r in available_rerankers}

    # Create all comparison tasks (query, pair)
    # Shuffle to eliminate order bias in ELO calculations
    comparison_tasks = []
    for query_id in common_query_ids:
        for model_x, model_y in pairs:
            comparison_tasks.append((query_id, model_x, model_y))

    # SHUFFLE to eliminate path-dependent bias
    random.shuffle(comparison_tasks)
    logger.info(f"Shuffled {len(comparison_tasks)} comparison tasks to eliminate order bias")
    print(f"   🔀 Shuffled {len(comparison_tasks)} comparisons to eliminate ELO order bias")

    # Process comparisons
    total_comparisons = len(comparison_tasks)
    judgments_list = []
    comparison_count = 0

    logger.info("Starting LLM judgments...")
    print(f"   🤖 Starting LLM judgments...")
    for query_id, model_x, model_y in comparison_tasks:
        query_text = queries[query_id]
        comparison_count += 1
        if comparison_count % 10 == 0:
            logger.info(f"Processing {comparison_count}/{total_comparisons} comparisons...")
            print(f"      ⏳ Progress: {comparison_count}/{total_comparisons} comparisons ({comparison_count*100/total_comparisons:.1f}%)")

        results_x = reranked_data[model_x][query_id]['results']
        results_y = reranked_data[model_y][query_id]['results']

        # Randomize A/B
        if random.random() < 0.5:
            a_model, b_model = model_x, model_y
            ranking_a = _format_ranking_for_prompt(results_x, corpus, config.llm_judge.prompt_top_k_for_comparison)
            ranking_b = _format_ranking_for_prompt(results_y, corpus, config.llm_judge.prompt_top_k_for_comparison)
            a_is_model_x = True
        else:
            a_model, b_model = model_y, model_x
            ranking_a = _format_ranking_for_prompt(results_y, corpus, config.llm_judge.prompt_top_k_for_comparison)
            ranking_b = _format_ranking_for_prompt(results_x, corpus, config.llm_judge.prompt_top_k_for_comparison)
            a_is_model_x = False

        # Get judgment
        prompt = _create_judge_prompt(query_text, ranking_a, ranking_b, config.llm_judge.prompt_top_k_for_comparison)
        judgment = _judge_with_llm(prompt, client, config.llm_judge.azure_deployment_id)

        # Get scores and update ELO
        score_x, score_y = _get_score_for_judgment(judgment, a_is_model_x, model_x, model_y)
        old_elo_x = elo_ratings[model_x]
        old_elo_y = elo_ratings[model_y]
        new_elo_x, new_elo_y = _update_elo(old_elo_x, old_elo_y, score_x, config.llm_judge.elo_k_factor)
        elo_ratings[model_x] = new_elo_x
        elo_ratings[model_y] = new_elo_y

        # Update W/L/T
        if score_x > score_y:
            win_loss_tie[model_x]['wins'] += 1
            win_loss_tie[model_y]['losses'] += 1
            winner = model_x
            loser = model_y
        elif score_x < score_y:
            win_loss_tie[model_x]['losses'] += 1
            win_loss_tie[model_y]['wins'] += 1
            winner = model_y
            loser = model_x
        else:
            win_loss_tie[model_x]['ties'] += 1
            win_loss_tie[model_y]['ties'] += 1
            winner = None
            loser = None

        # Store judgment
        judgment_record = {
            'query_id': query_id,
            'a_model': a_model,
            'b_model': b_model,
            'judge': judgment,
            'winner': winner if winner else 'TIE',
            'loser': loser if loser else 'TIE',
            'tie': judgment == 'TIE',
            'a_is_model_x': a_is_model_x,
            'model_x': model_x,
            'model_y': model_y,
            'model_x_score': score_x,
            'model_y_score': score_y,
            'model_x_elo_before': old_elo_x,
            'model_y_elo_before': old_elo_y,
            'model_x_elo_after': new_elo_x,
            'model_y_elo_after': new_elo_y
        }
        judgments_list.append(judgment_record)
    
    # Save judgments
    judgments_file = paths.get_judgments_file()
    logger.info(f"Saving judgments to {judgments_file}")
    with open(judgments_file, 'w') as f:
        for judgment in judgments_list:
            f.write(json.dumps(judgment) + '\n')
    
    judgments_csv = paths.get_judgments_csv_file()
    logger.info(f"Saving judgments CSV to {judgments_csv}")
    with open(judgments_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['query_id', 'model_x', 'model_y', 'a_model', 'b_model', 
                        'judge', 'winner', 'loser', 'tie', 'model_x_score', 'model_y_score'])
        for j in judgments_list:
            writer.writerow([j['query_id'], j['model_x'], j['model_y'], j['a_model'], j['b_model'],
                           j['judge'], j['winner'], j['loser'], j['tie'],
                           j['model_x_score'], j['model_y_score']])
    
    # Save ELO leaderboard
    sorted_models = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)
    elo_file = paths.get_elo_leaderboard_file()
    logger.info(f"Saving ELO leaderboard to {elo_file}")
    with open(elo_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['rank', 'model', 'model_name', 'elo', 'wins', 'losses', 'ties', 'win_rate'])
        for rank, (model, elo) in enumerate(sorted_models, 1):
            wlt = win_loss_tie[model]
            total = wlt['wins'] + wlt['losses'] + wlt['ties']
            win_rate = wlt['wins'] / total if total > 0 else 0.0
            model_name = model_names.get(model, model)
            writer.writerow([rank, model, model_name, f"{elo:.2f}",
                           wlt['wins'], wlt['losses'], wlt['ties'], f"{win_rate:.4f}"])
    
    # Create ELO plot
    plot_file = paths.get_elo_plot_file()
    logger.info(f"Creating ELO plot at {plot_file}")
    models_ordered = [m[0] for m in sorted_models]
    elos_ordered = [m[1] for m in sorted_models]
    model_labels = [model_names.get(m, m) for m in models_ordered]
    
    plt.figure(figsize=(12, 6))
    bars = plt.barh(range(len(models_ordered)), elos_ordered, color='#4C78A8')
    plt.yticks(range(len(models_ordered)), model_labels)
    plt.xlabel('ELO Rating', fontweight='bold')
    plt.title('ELO Leaderboard - Reranker Comparison', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    for i, (bar, elo) in enumerate(zip(bars, elos_ordered)):
        width = bar.get_width()
        plt.text(width + 10, bar.get_y() + bar.get_height()/2, 
                f'{elo:.0f}', ha='left', va='center', fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    result = {
        'status': 'completed',
        'num_comparisons': total_comparisons,
        'num_rerankers': len(available_rerankers),
        'elo_file': str(elo_file)
    }
    
    logger.info("LLM Judge stage completed successfully")
    return result

