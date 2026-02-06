"""
LLM Judge stage: Compare rerankers using LLM judgments and ELO ratings with parallel processing
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
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..config import Config
from ..paths import RunPaths


def _truncate_text(text: str, max_chars: int = 400) -> str:
    """Truncate text to max_chars, preserving words"""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rsplit(' ', 1)[0]
    return truncated + '...' if truncated != text[:max_chars] else truncated + '...'


def _format_ranking_for_prompt(results: List[Dict], corpus: Dict[str, str], k: int = 15, max_chars: int = 400) -> str:
    """Format ranking results for LLM prompt"""
    lines = []
    for i, result in enumerate(results[:k], 1):
        doc_id = result['doc_id']
        doc_text = corpus.get(doc_id, '')
        truncated = _truncate_text(doc_text, max_chars)
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


def _judge_single_comparison(args):
    """Judge a single comparison (for parallel execution)"""
    query_id, model_x, model_y, query_text, reranked_data, corpus, client, deployment_id, k, max_chars = args

    results_x = reranked_data[model_x][query_id]['results']
    results_y = reranked_data[model_y][query_id]['results']

    # Randomize A/B to reduce position bias
    if random.random() < 0.5:
        a_model, b_model = model_x, model_y
        ranking_a = _format_ranking_for_prompt(results_x, corpus, k, max_chars)
        ranking_b = _format_ranking_for_prompt(results_y, corpus, k, max_chars)
        a_is_model_x = True
    else:
        a_model, b_model = model_y, model_x
        ranking_a = _format_ranking_for_prompt(results_y, corpus, k, max_chars)
        ranking_b = _format_ranking_for_prompt(results_x, corpus, k, max_chars)
        a_is_model_x = False

    prompt = _create_judge_prompt(query_text, ranking_a, ranking_b, k)
    judgment = _judge_with_llm(prompt, client, deployment_id)

    return {
        'query_id': query_id,
        'model_x': model_x,
        'model_y': model_y,
        'a_model': a_model,
        'b_model': b_model,
        'a_is_model_x': a_is_model_x,
        'judgment': judgment
    }


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


def _get_score_for_judgment(judgment: str, a_is_model_x: bool) -> Tuple[float, float]:
    """Convert judgment to scores (score_x, score_y)"""
    if judgment == 'TIE':
        return 0.5, 0.5
    elif judgment == 'A':
        if a_is_model_x:
            return 1.0, 0.0
        else:
            return 0.0, 1.0
    else:  # B
        if a_is_model_x:
            return 0.0, 1.0
        else:
            return 1.0, 0.0


def llm_judge_stage(config: Config, paths: RunPaths, logger: logging.Logger) -> Dict:
    """
    LLM Judge stage: Compare rerankers using LLM with parallel processing

    Args:
        config: Pipeline configuration
        paths: Run paths manager
        logger: Logger instance

    Returns:
        Dictionary with stage results metadata
    """
    logger.info("Starting LLM Judge stage with parallel processing...")

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
                if reranker.name not in model_names:
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

    # Load existing judgments if they exist
    judgments_file = paths.get_judgments_file()
    existing_judgments = {}
    new_rerankers = set()  # Track which rerankers are new (not in existing judgments)
    
    if judgments_file.exists():
        logger.info(f"Loading existing judgments from {judgments_file}")
        existing_rerankers = set()
        with open(judgments_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                query_id = data['query_id']
                model_x = data['model_x']
                model_y = data['model_y']
                # Store as (query_id, model_x, model_y) and (query_id, model_y, model_x) for lookup
                key1 = (query_id, model_x, model_y)
                key2 = (query_id, model_y, model_x)
                existing_judgments[key1] = data
                existing_judgments[key2] = data
                existing_rerankers.add(model_x)
                existing_rerankers.add(model_y)
        
        # Find new rerankers (ones not in existing judgments)
        new_rerankers = set(available_rerankers) - existing_rerankers
        if new_rerankers:
            logger.info(f"Found new rerankers: {new_rerankers}")
            print(f"   🆕 Found new rerankers: {', '.join(new_rerankers)}")
            print(f"   ♻️  Will reuse existing judgments for old pairs")
            print(f"   🆕 Will make new judgments only for pairs involving new rerankers")
        else:
            logger.info("All rerankers already have judgments, will reuse all")
            print(f"   ♻️  All rerankers already have judgments, will reuse all")
    else:
        logger.info("No existing judgments found, will make all new judgments")
        print(f"   🆕 No existing judgments found, will make all new judgments")

    # Initialize ELO
    elo_ratings = {r: config.llm_judge.elo_initial_rating for r in available_rerankers}
    win_loss_tie = {r: {'wins': 0, 'losses': 0, 'ties': 0} for r in available_rerankers}

    # Get parameters
    k = config.llm_judge.prompt_top_k_for_comparison
    max_chars = config.llm_judge.prompt_truncate_doc_length
    max_workers = 10  # Number of parallel workers

    all_judgments = []
    query_count = 0
    reused_count = 0
    new_count = 0

    logger.info(f"Starting LLM judgments with {max_workers} parallel workers...")
    print(f"   ⚡ Using {max_workers} parallel workers")
    print(f"   🤖 Starting LLM judgments (query-by-query parallel processing)...\n")

    # Process each query's comparisons in parallel
    for query_id in common_query_ids:
        query_count += 1
        query_text = queries[query_id]

        # Create tasks for pairs that need new judgments
        tasks = []
        pairs_to_judge = []
        for model_x, model_y in pairs:
            # Check if we need to make a new judgment
            # Make new judgment if:
            # 1. No existing judgments file, OR
            # 2. This pair involves a new reranker, OR
            # 3. This specific (query_id, model_x, model_y) pair doesn't exist
            key = (query_id, model_x, model_y)
            if key in existing_judgments:
                # Reuse existing judgment
                judgment_record = existing_judgments[key].copy()
                all_judgments.append(judgment_record)
                reused_count += 1
            else:
                # Need to make new judgment
                tasks.append((query_id, model_x, model_y, query_text, reranked_data, corpus,
                             client, config.llm_judge.azure_deployment_id, k, max_chars))
                pairs_to_judge.append((model_x, model_y))

        # Execute new comparisons for this query in parallel
        if tasks:
            query_judgments = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_judge_single_comparison, task) for task in tasks]
                for future in as_completed(futures):
                    result = future.result()
                    query_judgments.append(result)
                    new_count += 1

            # Process new judgments for this query
            for result in query_judgments:
                judgment = result['judgment']
                model_x = result['model_x']
                model_y = result['model_y']
                a_is_model_x = result['a_is_model_x']

                # Get scores
                score_x, score_y = _get_score_for_judgment(judgment, a_is_model_x)

                # Update win/loss/tie
                if score_x > score_y:
                    winner = model_x
                elif score_x < score_y:
                    winner = model_y
                else:
                    winner = 'TIE'

                # Store judgment (ELO will be recalculated later from all judgments)
                judgment_record = {
                    'query_id': result['query_id'],
                    'a_model': result['a_model'],
                    'b_model': result['b_model'],
                    'judge': judgment,
                    'winner': winner,
                    'a_is_model_x': a_is_model_x,
                    'model_x': model_x,
                    'model_y': model_y,
                    'model_x_score': score_x,
                    'model_y_score': score_y,
                    'model_x_elo_before': 0,  # Will be set during ELO recalculation
                    'model_y_elo_before': 0,  # Will be set during ELO recalculation
                    'model_x_elo_after': 0,   # Will be set during ELO recalculation
                    'model_y_elo_after': 0    # Will be set during ELO recalculation
                }
                all_judgments.append(judgment_record)

        # Progress update
        if query_count % 10 == 0 or query_count == len(common_query_ids):
            progress = query_count * 100 / len(common_query_ids)
            total_comps = len(all_judgments)
            logger.info(f"Progress: {query_count}/{len(common_query_ids)} queries ({progress:.1f}%)")
            print(f"      ⏳ Progress: {query_count}/{len(common_query_ids)} queries ({progress:.1f}%) - {total_comps} comparisons ({reused_count} reused, {new_count} new)")

    # Recalculate ELO from all judgments (existing + new) in order
    logger.info("Recalculating ELO from all judgments...")
    elo_ratings = {r: config.llm_judge.elo_initial_rating for r in available_rerankers}
    win_loss_tie = {r: {'wins': 0, 'losses': 0, 'ties': 0} for r in available_rerankers}
    
    # Sort judgments by query_id to maintain consistent order
    all_judgments_sorted = sorted(all_judgments, key=lambda x: (x['query_id'], x['model_x'], x['model_y']))
    
    # Process all judgments to recalculate ELO
    for judgment_record in all_judgments_sorted:
        model_x = judgment_record['model_x']
        model_y = judgment_record['model_y']
        score_x = judgment_record['model_x_score']
        score_y = judgment_record['model_y_score']
        
        # Update ELO
        old_elo_x = elo_ratings[model_x]
        old_elo_y = elo_ratings[model_y]
        elo_ratings[model_x], elo_ratings[model_y] = _update_elo(
            old_elo_x, old_elo_y, score_x, config.llm_judge.elo_k_factor
        )
        
        # Update win/loss/tie (only count once, not per judgment record)
        if score_x > score_y:
            win_loss_tie[model_x]['wins'] += 1
            win_loss_tie[model_y]['losses'] += 1
        elif score_x < score_y:
            win_loss_tie[model_x]['losses'] += 1
            win_loss_tie[model_y]['wins'] += 1
        else:
            win_loss_tie[model_x]['ties'] += 1
            win_loss_tie[model_y]['ties'] += 1
        
        # Update ELO in judgment record
        judgment_record['model_x_elo_before'] = old_elo_x
        judgment_record['model_y_elo_before'] = old_elo_y
        judgment_record['model_x_elo_after'] = elo_ratings[model_x]
        judgment_record['model_y_elo_after'] = elo_ratings[model_y]

    # Save judgments
    judgments_file = paths.get_judgments_file()
    logger.info(f"Saving judgments to {judgments_file}")
    with open(judgments_file, 'w') as f:
        for judgment in all_judgments_sorted:
            f.write(json.dumps(judgment) + '\n')

    judgments_csv = paths.get_judgments_csv_file()
    logger.info(f"Saving judgments CSV to {judgments_csv}")
    with open(judgments_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['query_id', 'model_x', 'model_y', 'a_model', 'b_model',
                        'judge', 'winner', 'model_x_score', 'model_y_score'])
        for j in all_judgments:
            writer.writerow([j['query_id'], j['model_x'], j['model_y'], j['a_model'], j['b_model'],
                           j['judge'], j['winner'], j['model_x_score'], j['model_y_score']])

    # Save ELO ratings
    elo_file = paths.get_elo_leaderboard_file()
    logger.info(f"Saving ELO ratings to {elo_file}")
    with open(elo_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['reranker', 'elo_rating', 'wins', 'losses', 'ties', 'win_rate'])
        sorted_rerankers = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)
        for reranker, elo in sorted_rerankers:
            wlt = win_loss_tie[reranker]
            total = wlt['wins'] + wlt['losses'] + wlt['ties']
            win_rate = wlt['wins'] / total if total > 0 else 0.0
            writer.writerow([reranker, f"{elo:.2f}", wlt['wins'], wlt['losses'], wlt['ties'], f"{win_rate:.4f}"])

    # Create ELO plot
    plot_file = paths.get_elo_plot_file()
    logger.info(f"Creating ELO plot at {plot_file}")
    sorted_rerankers = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)
    rerankers = [r[0] for r in sorted_rerankers]
    elos = [r[1] for r in sorted_rerankers]

    plt.figure(figsize=(10, max(6, len(rerankers) * 0.5)))
    bars = plt.barh(range(len(rerankers)), elos, color='#4C78A8')
    plt.yticks(range(len(rerankers)), rerankers)
    plt.xlabel('ELO Rating', fontweight='bold')
    plt.title(f'Reranker ELO Ratings - {config.dataset.name}', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()

    # Add value labels
    for i, (bar, elo) in enumerate(zip(bars, elos)):
        width = bar.get_width()
        plt.text(width + 10, bar.get_y() + bar.get_height()/2,
                f'{elo:.0f}', ha='left', va='center', fontweight='bold')

    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()

    # Print final results
    logger.info("LLM Judge stage complete!")
    print(f"\n   ✅ LLM Judge complete!")
    if reused_count > 0:
        print(f"   ♻️  Reused {reused_count} existing judgments")
    if new_count > 0:
        print(f"   🆕 Made {new_count} new judgments")
    print(f"\n   📊 ELO Leaderboard:")
    for rank, (reranker, elo) in enumerate(sorted_rerankers, 1):
        wlt = win_loss_tie[reranker]
        total = wlt['wins'] + wlt['losses'] + wlt['ties']
        win_rate = wlt['wins'] / total if total > 0 else 0.0
        print(f"      {rank}. {reranker}: {elo:.0f} ELO (W:{wlt['wins']} L:{wlt['losses']} T:{wlt['ties']}, WR:{win_rate:.1%})")

    return {
        'status': 'success',
        'num_comparisons': len(all_judgments),
        'num_rerankers': len(available_rerankers),
        'num_queries': len(common_query_ids),
        'judgments_file': str(judgments_file),
        'elo_file': str(elo_file),
        'plot_file': str(plot_file)
    }
