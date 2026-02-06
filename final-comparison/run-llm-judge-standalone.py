#!/usr/bin/env python3
"""
Run LLM Judge as standalone on existing reranker results.
No need to rerun embed/retrieve/rerank.
"""

import sys
from pathlib import Path
import yaml
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.stages.llm-judge import llm_judge_stage
from pipeline.config import Config
from pipeline.paths import RunPaths
from pipeline.logger import setup_logger


def find_latest_run(dataset_name: str, runs_dir: Path = Path("runs")) -> Path:
    """Find the latest run directory for a dataset."""
    dataset_runs = runs_dir / dataset_name
    if not dataset_runs.exists():
        raise FileNotFoundError(f"No runs found for dataset: {dataset_name}")

    run_dirs = [d for d in dataset_runs.iterdir() if d.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in {dataset_runs}")

    latest_run = sorted(run_dirs)[-1]
    return latest_run


def run_llm_judge_for_dataset(dataset_name: str, config_path: Path, run_dir: Path = None):
    """Run LLM judge for a specific dataset"""

    print(f"\n{'='*70}")
    print(f"Running LLM Judge for: {dataset_name}")
    print(f"{'='*70}")

    try:
        # Find run directory
        if run_dir is None:
            run_dir = find_latest_run(dataset_name)
        print(f"📁 Run directory: {run_dir}")

        # Update config file temporarily with dataset name
        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        config_data['dataset']['name'] = dataset_name
        config_data['dataset']['base_path'] = f"datasets/{dataset_name}"

        # Create temporary config file
        temp_config = Path("/tmp/llm_judge_config_temp.yaml")
        with open(temp_config, 'w') as f:
            yaml.dump(config_data, f)

        # Load config using from_yaml
        config = Config.from_yaml(str(temp_config))

        # Create RunPaths with proper parameters
        # Extract timestamp from run_dir (format: runs/dataset/YYYYMMDD_HHMMSS)
        timestamp = run_dir.name
        paths = RunPaths(dataset_name, timestamp=timestamp)

        logger = setup_logger("llm_judge", run_dir / "llm_judge.log")

        # Run LLM judge
        result = llm_judge_stage(config, paths, logger)

        if result['status'] == 'success':
            print(f"\n✅ LLM Judge complete for {dataset_name}!")
            print(f"📊 Judgments: {result['num_comparisons']}")
            print(f"📈 ELO ratings saved to: {result['elo_file']}")
            print(f"📉 Plot saved to: {result['plot_file']}")
            return True
        else:
            print(f"⚠️  LLM Judge skipped: {result.get('reason', 'unknown')}")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Run LLM Judge on existing reranker results")
    parser.add_argument('--dataset', help='Dataset name (or "all" for all datasets)')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--run-dir', help='Specific run directory (optional)')

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return 1

    # Datasets to process
    if args.dataset and args.dataset.lower() != 'all':
        datasets = [args.dataset]
    else:
        # Auto-detect datasets from runs directory
        runs_dir = Path("runs")
        if runs_dir.exists():
            datasets = [d.name for d in runs_dir.iterdir() if d.is_dir()]
            print(f"🔍 Found datasets: {', '.join(datasets)}")
        else:
            print("❌ No runs directory found")
            return 1

    # Process each dataset
    results = {}
    for dataset in datasets:
        run_dir = Path(args.run_dir) if args.run_dir else None
        success = run_llm_judge_for_dataset(dataset, config_path, run_dir)
        results[dataset] = "✅ Success" if success else "❌ Failed"

    # Print summary
    print(f"\n{'='*70}")
    print("LLM JUDGE SUMMARY")
    print(f"{'='*70}")
    for dataset, status in results.items():
        print(f"{dataset:20s} {status}")
    print(f"{'='*70}\n")

    return 0 if all("Success" in s for s in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
