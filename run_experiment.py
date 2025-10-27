#!/usr/bin/env python3
"""
Orchestrate end-to-end experiment:
  1) Build FAISS index
  2) Retrieve top-50 per query
  3) Rerank with Cohere (v3.5) and ZeRank, log latencies, compute metrics
  4) Plot latency and evaluation metrics
  5) LLM judge and win distribution plot
  6) Save per-step and total timings
Outputs are written into the same experiment directory.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path


def run(cmd: list, cwd: Path = None):
    print("\n" + "=" * 80)
    print("RUN:", " ".join(cmd))
    print("=" * 80)
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def main():
    project_root = Path(__file__).resolve().parents[2]
    exp_dir = Path(__file__).resolve().parent

    # Paths
    dataset_dir = exp_dir / "fiqa_small"
    index_dir = exp_dir / "indexes" / "fiqa_small_bge"
    top50_file = exp_dir / "fiqa_small_top50.jsonl"
    cohere_out = exp_dir / "cohere_v35.jsonl"
    zerank_out = exp_dir / "zerank.jsonl"
    metrics_out = exp_dir / "metrics.json"
    latency_plot = exp_dir / "latency_comparison.png"
    eval_plot = exp_dir / "evaluation_metrics.png"
    llm_out = exp_dir / "llm_judge.json"
    win_plot = exp_dir / "win_distribution.png"
    timings_out = exp_dir / "timings.json"

    embed_model = "BAAI/bge-small-en-v1.5"
    cohere_model = "rerank-english-v3.5"

    timings = {}
    t0_all = time.perf_counter()

    # 1) Build index (idempotent)
    t0 = time.perf_counter()
    build_code = f"""
import json
from pathlib import Path
import sys
sys.path.append(str(Path('{project_root}')))
from eval.eval_rerankers import create_or_load_faiss_index
from tools.sample_beir_subset import load_corpus

dataset_dir = Path(r"{dataset_dir}")
index_dir = Path(r"{index_dir}")
corpus_path = dataset_dir / 'corpus.jsonl'
corpus = load_corpus(corpus_path)
index, doc_ids = create_or_load_faiss_index(corpus, r"{embed_model}", index_dir)
print(f"Index ready with {{len(doc_ids)}} documents at: {index_dir}")
"""
    run([sys.executable, "-c", build_code])
    timings["build_index_s"] = round(time.perf_counter() - t0, 3)

    # 2) Retrieve top-50
    t0 = time.perf_counter()
    run([
        sys.executable, str(project_root / "tools" / "retrieve_topk.py"),
        "--dataset_dir", str(dataset_dir),
        "--index_dir", str(index_dir),
        "--embed_model", embed_model,
        "--topk", "50",
        "--num_queries", "50",
        "--out_file", str(top50_file),
    ])
    timings["retrieve_top50_s"] = round(time.perf_counter() - t0, 3)

    # 3) Rerank + metrics
    t0 = time.perf_counter()
    qrels_path = dataset_dir / "qrels" / "test.tsv"
    run([
        sys.executable, str(project_root / "tools" / "rerank_retrieved_optimized.py"),
        "--input_file", str(top50_file),
        "--cohere_output", str(cohere_out),
        "--zerank_output", str(zerank_out),
        "--metrics_output", str(metrics_out),
        "--qrels_path", str(qrels_path),
        "--top_k", "15",
        "--randomize_order",
        "--cohere_model", cohere_model,
        "--zerank_model", "zerank-1",
    ])
    timings["rerank_and_metrics_s"] = round(time.perf_counter() - t0, 3)

    # 4) Plots (latency + evaluation)
    t0 = time.perf_counter()
    plot_code = f"""
import json
from pathlib import Path
import sys
sys.path.append(str(Path('{project_root}')))
from tools.create_visualizations import plot_latency_comparison, plot_evaluation_metrics

with open(r"{metrics_out}", 'r') as f:
    metrics = json.load(f)

plot_latency_comparison(metrics, r"{latency_plot}")
plot_evaluation_metrics(metrics, r"{eval_plot}")
print("Saved plots:", r"{latency_plot}", r"{eval_plot}")
"""
    run([sys.executable, "-c", plot_code])
    timings["plots_latency_metrics_s"] = round(time.perf_counter() - t0, 3)

    # 5) LLM judge + win distribution plot
    t0 = time.perf_counter()
    run([
        sys.executable, str(project_root / "tools" / "llm_judge.py"),
        "--cohere_file", str(cohere_out),
        "--zerank_file", str(zerank_out),
        "--top_k", "5",
        "--output_file", str(llm_out),
    ])

    plot_judge_code = f"""
import json
from pathlib import Path
import sys
sys.path.append(str(Path('{project_root}')))
from tools.create_visualizations import plot_win_distribution

import json
with open(r"{llm_out}", 'r') as f:
    llm = json.load(f)
plot_win_distribution(llm, r"{win_plot}")
print("Saved:", r"{win_plot}")
"""
    run([sys.executable, "-c", plot_judge_code])
    timings["llm_judge_and_plot_s"] = round(time.perf_counter() - t0, 3)

    timings["total_s"] = round(time.perf_counter() - t0_all, 3)

    with open(timings_out, "w") as f:
        json.dump(timings, f, indent=2)
    print("\nTimings saved to:", timings_out)


if __name__ == "__main__":
    main()


