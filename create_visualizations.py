#!/usr/bin/env python3
"""
Visualization Script for Reranker Evaluation Results

Creates visualizations from evaluation metrics and LLM judge results.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import numpy as np

def load_data(metrics_file: str, llm_results_file: str):
    """Load evaluation metrics and LLM judge results."""
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    with open(llm_results_file, 'r') as f:
        llm_results = json.load(f)
    
    return metrics, llm_results


def plot_latency_comparison(metrics: dict, output_file: str):
    """Plot latency comparison between Cohere and ZeRank."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    perf = metrics['performance']
    cohere_latency = perf['cohere_avg_latency_ms']
    zerank_latency = perf['zerank_avg_latency_ms']
    
    # Bar chart
    models = ['Cohere', 'ZeRank']
    latencies = [cohere_latency, zerank_latency]
    colors = ['#FF6B6B', '#4ECDC4']
    
    bars = ax1.bar(models, latencies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Average Latency (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Average Latency Comparison', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, latency in zip(bars, latencies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{latency:.1f} ms',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Comparison pie chart
    sizes = [cohere_latency, zerank_latency]
    labels = [f'Cohere\n{cohere_latency:.1f} ms', f'ZeRank\n{zerank_latency:.1f} ms']
    
    ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
            startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax2.set_title('Latency Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved latency comparison to: {output_file}")


def plot_evaluation_metrics(metrics: dict, output_file: str):
    """Plot nDCG and Recall metrics."""
    if not metrics.get('evaluation'):
        print("No evaluation metrics available (no ground truth)")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    eval_metrics = metrics['evaluation']
    
    # Extract data
    nDCG = eval_metrics['ndcg']
    Recall = eval_metrics['recall']
    
    ks = ['@1', '@5', '@10']
    cohere_ndcg = [nDCG[k]['cohere'] for k in ks]
    zerank_ndcg = [nDCG[k]['zerank'] for k in ks]
    cohere_recall = [Recall[k]['cohere'] for k in ks]
    zerank_recall = [Recall[k]['zerank'] for k in ks]
    
    x = np.arange(len(ks))
    width = 0.35
    
    # nDCG plot
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, cohere_ndcg, width, label='Cohere', color='#FF6B6B', alpha=0.7, edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, zerank_ndcg, width, label='ZeRank', color='#4ECDC4', alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax1.set_ylabel('nDCG Score', fontsize=12, fontweight='bold')
    ax1.set_title('nDCG@K Comparison (with ground truth)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['k=1', 'k=5', 'k=10'])
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9)
    
    # Recall plot
    ax2 = axes[1]
    bars3 = ax2.bar(x - width/2, cohere_recall, width, label='Cohere', color='#FF6B6B', alpha=0.7, edgecolor='black', linewidth=1.5)
    bars4 = ax2.bar(x + width/2, zerank_recall, width, label='ZeRank', color='#4ECDC4', alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax2.set_ylabel('Recall Score', fontsize=12, fontweight='bold')
    ax2.set_title('Recall@K Comparison (with ground truth)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['k=1', 'k=5', 'k=10'])
    ax2.legend(fontsize=11)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # Add value labels
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved evaluation metrics to: {output_file}")


def plot_elo_progression(llm_results: dict, output_file: str):
    """Plot ELO rating progression over queries."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    results = llm_results['results']
    queries = list(range(1, len(results) + 1))
    
    cohere_elos = [r['cohere_elo'] for r in results]
    zerank_elos = [r['zerank_elo'] for r in results]
    
    ax.plot(queries, cohere_elos, label='Cohere', color='#FF6B6B', linewidth=2.5, marker='o', markersize=4)
    ax.plot(queries, zerank_elos, label='ZeRank', color='#4ECDC4', linewidth=2.5, marker='s', markersize=4)
    
    ax.set_xlabel('Query Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('ELO Rating', fontsize=12, fontweight='bold')
    ax.set_title('ELO Rating Progression Over 50 Queries', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add initial and final ratings
    ax.axhline(y=1500, color='gray', linestyle='--', alpha=0.5, label='Initial Rating (1500)')
    ax.text(1, cohere_elos[0], f'{cohere_elos[0]:.0f}', ha='left', va='bottom', fontsize=9, fontweight='bold')
    ax.text(len(queries), cohere_elos[-1], f'{cohere_elos[-1]:.0f}', ha='right', va='top', fontsize=9, fontweight='bold')
    ax.text(1, zerank_elos[0], f'{zerank_elos[0]:.0f}', ha='left', va='top', fontsize=9, fontweight='bold')
    ax.text(len(queries), zerank_elos[-1], f'{zerank_elos[-1]:.0f}', ha='right', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved ELO progression to: {output_file}")


def plot_win_distribution(llm_results: dict, output_file: str):
    """Plot win distribution comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    cohere_wins = llm_results['cohere_wins']
    zerank_wins = llm_results['zerank_wins']
    ties = llm_results['ties']
    total = llm_results['total_queries']
    
    # Bar chart
    categories = ['Cohere\nWins', 'ZeRank\nWins', 'Ties']
    counts = [cohere_wins, zerank_wins, ties]
    percentages = [c/total*100 for c in counts]
    colors = ['#FF6B6B', '#4ECDC4', '#95A5A6']
    
    bars = ax1.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Number of Queries', fontsize=12, fontweight='bold')
    ax1.set_title('Win Distribution (Count)', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, count, pct in zip(bars, counts, percentages):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Pie chart
    labels = [f'Cohere\n{cohere_wins}', f'ZeRank\n{zerank_wins}', f'Ties\n{ties}']
    
    ax2.pie([cohere_wins, zerank_wins, ties], labels=labels, colors=colors, 
            autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax2.set_title('Win Distribution (Percentage)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved win distribution to: {output_file}")


def plot_elo_changes(llm_results: dict, output_file: str):
    """Plot ELO change per query."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    results = llm_results['results']
    queries = list(range(1, len(results) + 1))
    
    elo_changes = [r['elo_change'] for r in results]
    colors = ['#FF6B6B' if change < 0 else '#4ECDC4' if change > 0 else '#95A5A6' for change in elo_changes]
    
    bars = ax.bar(queries, elo_changes, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Query Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('ELO Change', fontsize=12, fontweight='bold')
    ax.set_title('ELO Change Per Query (Positive = Cohere Wins, Negative = ZeRank Wins)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add legend
    red_patch = mpatches.Patch(color='#FF6B6B', label='ZeRank Wins (Cohere loses points)')
    green_patch = mpatches.Patch(color='#4ECDC4', label='Cohere Wins (Cohere gains points)')
    gray_patch = mpatches.Patch(color='#95A5A6', label='Tie')
    ax.legend(handles=[red_patch, green_patch, gray_patch], fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved ELO changes to: {output_file}")


def plot_summary_dashboard(metrics: dict, llm_results: dict, output_file: str):
    """Create a comprehensive summary dashboard."""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Latency comparison (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    perf = metrics['performance']
    models = ['Cohere', 'ZeRank']
    latencies = [perf['cohere_avg_latency_ms'], perf['zerank_avg_latency_ms']]
    colors_lat = ['#FF6B6B', '#4ECDC4']
    bars = ax1.bar(models, latencies, color=colors_lat, alpha=0.7, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Latency (ms)', fontsize=10, fontweight='bold')
    ax1.set_title('Average Latency', fontsize=11, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    for bar, latency in zip(bars, latencies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{latency:.0f}ms', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 2. Win distribution (top center)
    ax2 = fig.add_subplot(gs[0, 1])
    cohere_wins = llm_results['cohere_wins']
    zerank_wins = llm_results['zerank_wins']
    ties = llm_results['ties']
    ax2.pie([cohere_wins, zerank_wins, ties], labels=['Cohere', 'ZeRank', 'Ties'],
            colors=['#FF6B6B', '#4ECDC4', '#95A5A6'], autopct='%1.1f%%', startangle=90)
    ax2.set_title('Win Distribution', fontsize=11, fontweight='bold')
    
    # 3. Final ELO ratings (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    cohere_elo = llm_results['final_cohere_elo']
    zerank_elo = llm_results['final_zerank_elo']
    elos = [cohere_elo, zerank_elo]
    bars = ax3.bar(models, elos, color=colors_lat, alpha=0.7, edgecolor='black', linewidth=2)
    ax3.set_ylabel('ELO Rating', fontsize=10, fontweight='bold')
    ax3.set_title('Final ELO Ratings', fontsize=11, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    ax3.axhline(y=1500, color='gray', linestyle='--', alpha=0.5)
    for bar, elo in zip(bars, elos):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{elo:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 4. ELO progression (middle row, spans 2 columns)
    ax4 = fig.add_subplot(gs[1, :2])
    results = llm_results['results']
    queries = list(range(1, len(results) + 1))
    cohere_elos = [r['cohere_elo'] for r in results]
    zerank_elos = [r['zerank_elo'] for r in results]
    ax4.plot(queries, cohere_elos, label='Cohere', color='#FF6B6B', linewidth=2, marker='o', markersize=3)
    ax4.plot(queries, zerank_elos, label='ZeRank', color='#4ECDC4', linewidth=2, marker='s', markersize=3)
    ax4.set_xlabel('Query Number', fontsize=10, fontweight='bold')
    ax4.set_ylabel('ELO Rating', fontsize=10, fontweight='bold')
    ax4.set_title('ELO Progression', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # 5. ELO changes (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    elo_changes = [r['elo_change'] for r in results]
    colors = ['#FF6B6B' if change < 0 else '#4ECDC4' if change > 0 else '#95A5A6' for change in elo_changes]
    ax5.bar(queries, elo_changes, color=colors, alpha=0.7, edgecolor='black', linewidth=0.3)
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax5.set_xlabel('Query Number', fontsize=10, fontweight='bold')
    ax5.set_ylabel('ELO Change', fontsize=10, fontweight='bold')
    ax5.set_title('ELO Changes', fontsize=11, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)
    
    # 6. Evaluation metrics (bottom row, spans 3 columns)
    if metrics.get('evaluation'):
        ax6 = fig.add_subplot(gs[2, :])
        eval_metrics = metrics['evaluation']
        nDCG = eval_metrics['ndcg']
        Recall = eval_metrics['recall']
        
        ks = ['@1', '@5', '@10']
        cohere_ndcg = [nDCG[k]['cohere'] for k in ks]
        zerank_ndcg = [nDCG[k]['zerank'] for k in ks]
        cohere_recall = [Recall[k]['cohere'] for k in ks]
        zerank_recall = [Recall[k]['zerank'] for k in ks]
        
        x = np.arange(len(ks))
        width = 0.35
        
        ax6.bar(x - width/2, cohere_ndcg, width, label='Cohere nDCG', color='#FF6B6B', alpha=0.5, edgecolor='black')
        ax6.bar(x + width/2, zerank_ndcg, width, label='ZeRank nDCG', color='#4ECDC4', alpha=0.5, edgecolor='black')
        ax6.bar(x - width/2, cohere_recall, width, label='Cohere Recall', color='#FF6B6B', alpha=0.8, edgecolor='black', hatch='//')
        ax6.bar(x + width/2, zerank_recall, width, label='ZeRank Recall', color='#4ECDC4', alpha=0.8, edgecolor='black', hatch='//')
        
        ax6.set_ylabel('Score', fontsize=10, fontweight='bold')
        ax6.set_title('Evaluation Metrics (nDCG & Recall)', fontsize=11, fontweight='bold')
        ax6.set_xticks(x)
        ax6.set_xticklabels(['k=1', 'k=5', 'k=10'])
        ax6.legend(fontsize=9)
        ax6.grid(axis='y', alpha=0.3)
        ax6.set_ylim([0, 1])
    
    # Add overall title
    fig.suptitle('Reranker Evaluation Dashboard - 50 Queries', fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved dashboard to: {output_file}")


def main():
    metrics_file = 'runs/evaluation_metrics.json'
    llm_results_file = 'runs/llm_judge_results_50queries.json'
    output_dir = Path('runs/visualizations')
    output_dir.mkdir(exist_ok=True)
    
    print("Loading data...")
    metrics, llm_results = load_data(metrics_file, llm_results_file)
    
    print("\nGenerating visualizations...")
    
    # Individual plots
    plot_latency_comparison(metrics, output_dir / '1_latency_comparison.png')
    plot_evaluation_metrics(metrics, output_dir / '2_evaluation_metrics.png')
    plot_elo_progression(llm_results, output_dir / '3_elo_progression.png')
    plot_win_distribution(llm_results, output_dir / '4_win_distribution.png')
    plot_elo_changes(llm_results, output_dir / '5_elo_changes.png')
    
    # Combined dashboard
    plot_summary_dashboard(metrics, llm_results, output_dir / 'dashboard.png')
    
    print("\n" + "="*60)
    print("All visualizations saved to:", output_dir)
    print("="*60)


if __name__ == "__main__":
    main()


