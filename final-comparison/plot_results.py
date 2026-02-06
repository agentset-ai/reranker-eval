#!/usr/bin/env python3
"""
Create visualization plots from aggregated results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_latency(results, output_file):
    """Plot average per-query latency for all rerankers"""
    rerankers = []
    latencies = []

    for reranker_data in results:
        if 'avg_latency_ms' in reranker_data['overall']:
            rerankers.append(reranker_data['name'])
            latencies.append(reranker_data['overall']['avg_latency_ms'])

    # Sort by latency (ascending)
    sorted_indices = np.argsort(latencies)
    rerankers = [rerankers[i] for i in sorted_indices]
    latencies = [latencies[i] for i in sorted_indices]

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(rerankers)))
    bars = ax.barh(range(len(rerankers)), latencies, color=colors)

    ax.set_yticks(range(len(rerankers)))
    ax.set_yticklabels(rerankers)
    ax.set_xlabel('Average Latency (ms)', fontsize=12)
    ax.set_title('Average Per-Query Latency by Reranker', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # Add value labels
    for i, (bar, latency) in enumerate(zip(bars, latencies)):
        ax.text(latency + 5, i, f'{latency:.1f}ms', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved latency plot: {output_file}")


def plot_elo_leaderboard(results, output_file):
    """Plot ELO leaderboard for all rerankers"""
    rerankers = []
    elos = []
    elo_stds = []

    for reranker_data in results:
        if 'elo' in reranker_data['overall']:
            rerankers.append(reranker_data['name'])
            elos.append(reranker_data['overall']['elo'])
            elo_stds.append(reranker_data['overall'].get('elo_std', 0))

    # Already sorted by ELO (descending) in results
    rerankers = rerankers[::-1]  # Reverse for horizontal bar chart (top to bottom)
    elos = elos[::-1]
    elo_stds = elo_stds[::-1]

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))

    # Color gradient based on ELO
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(rerankers)))
    bars = ax.barh(range(len(rerankers)), elos, xerr=elo_stds,
                   color=colors, capsize=4, alpha=0.8, edgecolor='black', linewidth=1.5)

    ax.set_yticks(range(len(rerankers)))
    ax.set_yticklabels(rerankers, fontsize=11, fontweight='bold')
    ax.set_xlabel('ELO Rating', fontsize=12, fontweight='bold')
    ax.set_title('Reranker ELO Leaderboard (Across All Datasets)', fontsize=14, fontweight='bold')
    ax.axvline(x=1500, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Initial Rating (1500)')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', fontsize=10)

    # Add value labels
    for i, (bar, elo, std) in enumerate(zip(bars, elos, elo_stds)):
        ax.text(elo + std + 10, i, f'{elo:.1f}', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved ELO leaderboard plot: {output_file}")


def plot_elo_by_dataset(results, output_file):
    """Plot ELO ratings per dataset for all rerankers"""
    # Get all datasets
    all_datasets = set()
    for reranker_data in results:
        all_datasets.update(reranker_data['by_dataset'].keys())

    datasets = sorted(all_datasets)

    # Prepare data
    rerankers = [r['name'] for r in results]
    elo_matrix = []

    for reranker_data in results:
        elo_row = []
        for dataset in datasets:
            if dataset in reranker_data['by_dataset'] and 'elo' in reranker_data['by_dataset'][dataset]:
                elo_row.append(reranker_data['by_dataset'][dataset]['elo'])
            else:
                elo_row.append(None)
        elo_matrix.append(elo_row)

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(datasets))
    width = 0.08

    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(rerankers)))

    for i, (reranker, elo_row) in enumerate(zip(rerankers, elo_matrix)):
        # Filter out None values
        valid_data = [(j, elo) for j, elo in enumerate(elo_row) if elo is not None]
        if valid_data:
            indices, values = zip(*valid_data)
            positions = [x[idx] + i * width - (len(rerankers) * width / 2) + width/2 for idx in indices]
            ax.bar(positions, values, width, label=reranker, color=colors[i],
                   alpha=0.85, edgecolor='black', linewidth=0.8)

    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('ELO Rating', fontsize=12, fontweight='bold')
    ax.set_title('Reranker ELO Ratings by Dataset', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=0, ha='center', fontsize=11)
    ax.axhline(y=1500, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Initial Rating')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Set y-axis limits with some padding
    all_elos = [elo for row in elo_matrix for elo in row if elo is not None]
    if all_elos:
        ymin = min(all_elos) - 50
        ymax = max(all_elos) + 50
        ax.set_ylim(ymin, ymax)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved per-dataset ELO plot: {output_file}")


def main():
    # Load results
    results_file = Path("results_all_datasets.json")

    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        print("   Run aggregate_all_results.py first!")
        return 1

    print("📊 Loading results...")
    with open(results_file, 'r') as f:
        results = json.load(f)

    print(f"✅ Loaded data for {len(results)} rerankers\n")

    # Create output directory
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)

    # Generate plots
    print("🎨 Generating plots...\n")

    plot_latency(results, output_dir / "latency_comparison.png")
    plot_elo_leaderboard(results, output_dir / "elo_leaderboard.png")
    plot_elo_by_dataset(results, output_dir / "elo_by_dataset.png")

    print(f"\n✅ All plots saved to: {output_dir}/")
    print("\nGenerated files:")
    print(f"  - {output_dir}/latency_comparison.png")
    print(f"  - {output_dir}/elo_leaderboard.png")
    print(f"  - {output_dir}/elo_by_dataset.png")


if __name__ == "__main__":
    import sys
    sys.exit(main())
