# Reranker Evaluation: Cohere v3.5 vs ZeRank-1

A comprehensive evaluation comparing two state-of-the-art reranking models on the FiQA financial Q&A dataset. Benchmarking Cohere v3.5 and ZeRank-1 rerankers in a RAG pipeline, evaluating accuracy (nDCG, Recall), latency, and LLM preferences.

## Overview

This project evaluates and compares two reranking models:
- **Cohere `rerank-v3.5`**: Via Python SDK (v2 endpoint)
- **ZeRank-1**: Via HTTP API

The evaluation measures performance across multiple dimensions: accuracy (nDCG/Recall), speed (latency), and human preference (via LLM judge).

## Project Structure

### Core Scripts

- **`run_experiment.py`** - Main orchestrator that runs the full evaluation pipeline
- **`eval_rerankers.py`** - FAISS index creation and management
- **`retrieve_topk.py`** - Initial retrieval of top-K candidates using FAISS
- **`rerank_retrieved_optimized.py`** - Async reranking with both models, latency tracking, and metrics
- **`llm_judge.py`** - LLM-based judge using GPT-5 to compare reranker quality
- **`create_visualizations.py`** - Visualization generation for results
- **`sample_beir_subset.py`** - Dataset sampling utility

### Data Files

- **`fiqa_small/`** - Evaluation dataset
  - `corpus.jsonl` - Document corpus
  - `queries.jsonl` - Test queries
  - `qrels/test.tsv` - Ground truth relevance judgments
- **`fiqa_small_top50.jsonl`** - Retrieved top-50 candidates per query (intermediate)
- **`indexes/`** - FAISS indexes for embeddings

### Results

- **`cohere_final.jsonl`** - Cohere v3.5 reranked results
- **`zerank_final.jsonl`** - ZeRank-1 reranked results
- **`metrics_final.json`** - All evaluation metrics
- **`llm_judge_results.json`** - LLM judge results with ELO scores

### Visualizations

- `results_table.png` - Simple comparison table
- `experiment_summary.png` - Comprehensive 4-panel summary
- `latency_comparison.png` - Latency breakdown (mean, p50, p90)
- `evaluation_metrics.png` - nDCG and Recall comparison
- `win_distribution.png` - LLM judge win distribution

## Experiment Setup

### Dataset
- **Source**: FiQA (Financial Q&A) from BEIR benchmark
- **Documents**: 500 financial documents
- **Queries**: 50 evaluation queries
- **Embedding Model**: `BAAI/bge-small-en-v1.5` for initial retrieval
- **Retrieval**: Top-50 candidates per query using FAISS

### Methodology
1. **Initial Retrieval**: Embed queries and retrieve top-50 candidates using FAISS
2. **Reranking**: Both models rerank the same 50 candidates per query
3. **Evaluation**: Compute nDCG@1,5,10 and Recall@1,5,10 using ground truth
4. **LLM Judging**: GPT-5 judges quality on 50 queries (ELO rating system)
5. **Latency Measurement**: Record API call latencies (p50, mean, p90)

### Implementation Details
- Shared preprocessing: Text truncated to 3KB per document
- Separate async sessions per provider for fair comparison
- Alternating call order to mitigate warm-connection bias
- Concurrency: 4 parallel requests per provider
- Connection pooling with proper timeout handling

## Results

### Summary Table

| Model | nDCG@10 | Recall@10 | LLM Wins | Mean Latency |
|-------|---------|-----------|----------|--------------|
| Cohere v3.5 | 0.092 | 0.097 | 9 | 512 ms |
| ZeRank-1 | 0.115 | 0.125 | 39 | 788 ms |

### Key Findings

**1. Speed (Latency)**
- Cohere v3.5: 512ms average, 580ms p90
- ZeRank-1: 788ms average, 1673ms p90 (high variance)
- **Winner**: Cohere v3.5 (1.5x faster)

**2. Accuracy (nDCG@10)**
- Cohere v3.5: 0.092
- ZeRank-1: 0.115
- **Winner**: ZeRank-1 (25% better)

**3. Human Preference (LLM Judge)**
- Cohere v3.5: 9 wins
- ZeRank-1: 39 wins (2 ties)
- **Winner**: ZeRank-1 (81% preference)

### Verdict

- **For Speed-Critical Applications**: Choose Cohere v3.5
  - Faster and more consistent latency
  - Predictable p90 performance
  
- **For Quality-Critical Applications**: Choose ZeRank-1
  - Better accuracy and relevance
  - Higher human preference
  
**Trade-off**: Speed vs. Quality

## Getting Started

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file or set environment variables:

```bash
export COHERE_API_KEY="your_cohere_key"
export ZERANK_API_KEY="your_zerank_key"

# For LLM judge
export AZURE_API_KEY="your_azure_key"
export AZURE_RESOURCE_NAME="your_resource"
export AZURE_DEPLOYMENT_ID="your_deployment"
```

## Usage

### Full Pipeline (Recommended)

Run the complete experiment pipeline:

```bash
python run_experiment.py
```

This orchestrates:
1. FAISS index creation (if needed)
2. Top-50 retrieval
3. Reranking with both models
4. Metrics calculation
5. Visualization generation
6. LLM judging

### Individual Steps

#### 1. Create FAISS Index

```python
from eval_rerankers import create_or_load_faiss_index
from sample_beir_subset import load_corpus

corpus = load_corpus(Path("fiqa_small/corpus.jsonl"))
index, doc_ids = create_or_load_faiss_index(
    corpus, 
    "BAAI/bge-small-en-v1.5", 
    Path("indexes/fiqa_small_bge")
)
```

#### 2. Retrieve Top-K Candidates

```bash
python retrieve_topk.py \
    --dataset_dir fiqa_small \
    --index_dir indexes/fiqa_small_bge \
    --embed_model BAAI/bge-small-en-v1.5 \
    --topk 50 \
    --num_queries 50 \
    --out_file fiqa_small_top50.jsonl
```

#### 3. Rerank Candidates

```bash
python rerank_retrieved_optimized.py \
    --input_file fiqa_small_top50.jsonl \
    --cohere_output cohere_final.jsonl \
    --zerank_output zerank_final.jsonl \
    --top_k 15 \
    --cohere_concurrency 4 \
    --zerank_concurrency 4 \
    --max_text_bytes 3000 \
    --store_trimmed_text \
    --randomize_order \
    --metrics_output metrics_final.json \
    --qrels_path fiqa_small/qrels/test.tsv
```

#### 4. LLM Judge

```bash
python llm_judge.py \
    --cohere_file cohere_final.jsonl \
    --zerank_file zerank_final.jsonl \
    --top_k 5 \
    --output_file llm_judge_results.json
```

#### 5. Generate Visualizations

```python
from create_visualizations import (
    plot_latency_comparison,
    plot_evaluation_metrics,
    plot_win_distribution
)
import json

with open("metrics_final.json") as f:
    metrics = json.load(f)

with open("llm_judge_results.json") as f:
    llm_results = json.load(f)

plot_latency_comparison(metrics, "latency_comparison.png")
plot_evaluation_metrics(metrics, "evaluation_metrics.png")
plot_win_distribution(llm_results, "win_distribution.png")
```

## Technical Details

### Reranking Implementation

- **Async/Await**: Uses `aiohttp` for concurrent API calls
- **Connection Pooling**: TCP connection reuse with proper limits
- **Semaphores**: Per-provider concurrency control (excludes wait time from latency)
- **Error Handling**: Graceful degradation on API failures
- **Text Truncation**: UTF-8 safe truncation at sentence boundaries

### Evaluation Metrics

- **nDCG** (Normalized Discounted Cumulative Gain): Ranked retrieval quality
- **Recall**: Coverage of relevant documents
- **Latency**: p50, mean, and p90 percentiles
- **ELO Rating**: Pairwise comparison via LLM judge

### LLM Judge

- Uses Azure GPT-5 for relevance judgments
- ELO rating system (base 1500, ±24 points per judgment)
- Compares top-5 results from each model
- Returns "A", "B", or "TIE" for each query

## Requirements

See `requirements.txt` for full dependency list. Key dependencies:
- `fastembed` - Embedding models
- `faiss-cpu` - Vector similarity search
- `cohere` - Cohere SDK
- `aiohttp` - Async HTTP client
- `openai` - Azure OpenAI for LLM judge
- `matplotlib` - Visualizations
- `beir` - Dataset utilities

## Notes

- Cohere v3.5 requires Python SDK with v2 endpoint access
- ZeRank uses HTTP API with US/EU endpoints
- Text truncation to 3KB per document ensures fair comparison
- LLM judge requires Azure OpenAI GPT-5 deployment
- All timings exclude local queue wait time for accurate latency measurement
