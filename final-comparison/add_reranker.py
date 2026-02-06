#!/usr/bin/env python3
"""
Add a single new reranker to existing dataset runs.
Reuses existing embeddings, retrieval, and other reranker results.
"""

import yaml
import argparse
from pathlib import Path
import sys

# Add parent directory to path so pipeline can be imported as a package
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.stages.rerank import rerank_documents
from pipeline.stages.evaluate import evaluate_rerankers
from pipeline.logger import setup_logger


def find_latest_run(dataset_name: str, runs_dir: Path = Path("runs")) -> Path:
    """Find the latest run directory for a dataset."""
    dataset_runs = runs_dir / dataset_name
    if not dataset_runs.exists():
        raise FileNotFoundError(f"No runs found for dataset: {dataset_name}")

    # Get all timestamped directories
    run_dirs = [d for d in dataset_runs.iterdir() if d.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in {dataset_runs}")

    # Sort by timestamp (directory name) and get latest
    latest_run = sorted(run_dirs)[-1]
    return latest_run


def add_reranker_to_run(
    run_dir: Path,
    reranker_config: dict,
    dataset_config: dict,
    retrieval_config: dict,
    skip_evaluate: bool = False
):
    """Add a new reranker to an existing run."""

    logger = setup_logger("add_reranker", run_dir / "add_reranker.log")

    # Check if reranker already exists
    reranker_name = reranker_config["name"]
    reranked_file = run_dir / "rerank" / f"reranked_{reranker_name}.jsonl"

    if reranked_file.exists():
        logger.info(f"Reranker {reranker_name} already exists, skipping")
        print(f"⚠️  Reranker '{reranker_name}' already exists in this run")
        return

    # Verify required directories exist
    embeddings_dir = run_dir / "embeddings"
    retrieval_dir = run_dir / "retrieval"

    if not embeddings_dir.exists() or not retrieval_dir.exists():
        raise FileNotFoundError(
            f"Missing embeddings or retrieval directory in {run_dir}"
        )

    print(f"\n🔄 Adding reranker '{reranker_name}' to run: {run_dir.name}")

    # Run reranking for the new reranker only
    try:
        logger.info(f"Starting rerank for {reranker_name}")
        print(f"  ⏳ Reranking with {reranker_name}...")

        rerank_documents(
            dataset_config=dataset_config,
            rerankers=[reranker_config],  # Only the new reranker
            retrieval_config=retrieval_config,
            run_dir=run_dir,
            logger=logger
        )

        print(f"  ✅ Reranking complete")
        logger.info(f"Rerank complete for {reranker_name}")

    except Exception as e:
        logger.error(f"Error during reranking: {e}")
        raise

    # Re-run evaluation with all rerankers
    if not skip_evaluate:
        try:
            logger.info("Re-running evaluation with all rerankers")
            print(f"  ⏳ Re-evaluating all rerankers...")

            # Get all reranker files (including the new one)
            all_reranked_files = list((run_dir / "rerank").glob("reranked_*.jsonl"))

            evaluate_rerankers(
                dataset_config=dataset_config,
                reranked_files=all_reranked_files,
                run_dir=run_dir,
                logger=logger
            )

            print(f"  ✅ Evaluation complete")
            logger.info("Evaluation complete")

        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            print(f"  ⚠️  Evaluation failed: {e}")

    print(f"\n✅ Successfully added reranker '{reranker_name}'")


def main():
    parser = argparse.ArgumentParser(
        description="Add a new reranker to existing dataset runs"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name (e.g., 'business-reports', 'fiqa_small')"
    )
    parser.add_argument(
        "--reranker-name",
        required=True,
        help="Name of the reranker to add (must be defined in config.yaml)"
    )
    parser.add_argument(
        "--run-dir",
        help="Specific run directory (default: latest run for dataset)"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml)"
    )
    parser.add_argument(
        "--skip-evaluate",
        action="store_true",
        help="Skip re-evaluation after adding reranker"
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return 1

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Find the reranker config
    reranker_config = None
    for r in config["rerankers"]:
        if r["name"] == args.reranker_name:
            reranker_config = r
            break

    if not reranker_config:
        print(f"❌ Reranker '{args.reranker_name}' not found in config")
        print(f"Available rerankers: {[r['name'] for r in config['rerankers']]}")
        return 1

    # Find run directory
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            print(f"❌ Run directory not found: {run_dir}")
            return 1
    else:
        try:
            run_dir = find_latest_run(args.dataset)
        except FileNotFoundError as e:
            print(f"❌ {e}")
            return 1

    print(f"📊 Dataset: {args.dataset}")
    print(f"📁 Run directory: {run_dir}")
    print(f"🔧 Adding reranker: {args.reranker_name}")

    # Get dataset config for this run
    # Use dataset name from command line argument to construct proper config
    dataset_config = config["dataset"].copy()
    dataset_config["name"] = args.dataset
    dataset_config["base_path"] = f"datasets/{args.dataset}"

    # Add the reranker
    try:
        add_reranker_to_run(
            run_dir=run_dir,
            reranker_config=reranker_config,
            dataset_config=dataset_config,
            retrieval_config=config["retrieval"],
            skip_evaluate=args.skip_evaluate
        )
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
