# Reranker Evaluation Framework

Comprehensive evaluation of 8 reranking models across 6 datasets using LLM-as-judge pairwise comparisons.

## What is this?

A framework to benchmark reranking models on real-world retrieval tasks. We compare 8 commercial and open-source rerankers across 6 diverse datasets (financial Q&A, scientific papers, essays, business documents, web search, factual Q&A) using GPT-4 as a judge for pairwise comparisons.

## Results

### Overall Leaderboard (ELO Ratings)

| Rank | Model | ELO | Win Rate | Avg Latency |
|------|-------|-----|----------|-------------|
| 1 | Zerank 1 | 1642.0 ±45.3 | 62.9% | 1,126ms |
| 2 | Voyage Rerank 2.5 | 1628.8 ±49.8 | 62.6% | 610ms |
| 3 | Contextual AI Rerank v2 | 1549.7 ±51.2 | 45.2% | 3,010ms |
| 4 | Voyage Rerank 2.5 Lite | 1509.6 ±50.1 | 50.4% | 607ms |
| 5 | BGE Reranker v2 M3 | 1467.6 ±53.9 | 33.0% | 1,891ms |
| 6 | Zerank 1 Small | 1458.4 ±54.6 | 56.2% | 1,109ms |
| 7 | Cohere Rerank 3.5 | 1402.7 ±51.3 | 37.8% | 492ms |
| 8 | Jina Reranker v2 Base | 1335.1 ±50.7 | 37.8% | 1,411ms |

**Best by Domain:**
- **Finance Q&A**: Zerank 1 (67.5% win rate)
- **Scientific Papers**: Voyage 2.5 (68.9% win rate)
- **Long-form Essays**: Voyage 2.5 Lite (52.3% win rate)
- **Business Documents**: Voyage 2.5 Lite (58.7% win rate)
- **Web Search**: Zerank 1 Small (58.6% win rate)
- **Factual Q&A**: Voyage 2.5 (68.0% win rate)

Full results: [results.json](results.json)

## Quick Start

### Prerequisites

```bash
pip install torch sentence-transformers faiss-cpu openai cohere jina voyageai pyyaml
```

API keys needed:
- `VOYAGE_API_KEY`
- `COHERE_API_KEY`
- `JINA_API_KEY`
- `ZERANK_API_KEY`
- `CONTEXTUAL_API_KEY`
- `REPLICATE_API_TOKEN`
- Azure OpenAI: `AZURE_API_KEY`, `AZURE_RESOURCE_NAME`, `AZURE_DEPLOYMENT_ID`

### Download Datasets

The benchmark uses 6 datasets from BEIR. Download them here:

- **FiQa**: https://huggingface.co/datasets/BeIR/fiqa
- **SciFact**: https://huggingface.co/datasets/BeIR/scifact
- **MSMARCO**: https://huggingface.co/datasets/BeIR/msmarco
- **DBPedia**: https://huggingface.co/datasets/BeIR/dbpedia-entity
- **Paul Graham Essays**: Custom dataset (essays chunked from paulgraham.com)
- **Business Reports**: Custom dataset

Place downloaded datasets in the `datasets/` directory.

### Run on Your Own Dataset

1. **Prepare your dataset** in BEIR format:
```
your_dataset/
├── corpus.jsonl      # {"_id": "doc1", "title": "...", "text": "..."}
├── queries.jsonl     # {"_id": "q1", "text": "..."}
└── qrels/
    └── test.tsv      # query-id\tcorpus-id\tscore
```

2. **Update config.yaml**:
```yaml
dataset:
  name: "your_dataset"
  base_path: "your_dataset"
  corpus_file: "corpus.jsonl"
  queries_file: "queries.jsonl"
  qrels_file: "qrels/test.tsv"

retrieval:
  top_k: 50
  query_subset: 50  # or null for all queries

rerankers:
  - name: "voyage"
    type: "voyage"
    model: "rerank-2.5"
    api_key_env: "VOYAGE_API_KEY"
    top_k: 15
  # Add more rerankers...

llm_judge:
  enabled: true
  provider: "azure_openai"
  azure_api_key_env: "AZURE_API_KEY"
  azure_resource_name_env: "AZURE_RESOURCE_NAME"
  azure_deployment_id_env: "AZURE_DEPLOYMENT_ID"
```

3. **Run the pipeline**:
```bash
export VOYAGE_API_KEY="your-key"
export AZURE_API_KEY="your-key"
# ... set other API keys

python3 -m pipeline config.yaml
```

The pipeline will:
1. Embed documents using BGE-small-en-v1.5
2. Retrieve top-50 candidates with FAISS
3. Rerank using all configured models
4. Generate pairwise judgments using GPT-4
5. Calculate ELO ratings

Results saved to: `runs/your_dataset/TIMESTAMP/`

## How It Works

### Evaluation Method

1. **Embedding & Retrieval**: Documents embedded with BGE-small-en-v1.5, FAISS retrieves top-50 candidates
2. **Reranking**: Each model reranks the top-50 to produce top-15 results
3. **LLM-as-Judge**: GPT-4 compares all reranker pairs on each query
   - Randomized A/B positions to prevent bias
   - Truncated documents (300 chars) for fairness
   - Returns winner: A, B, or TIE
4. **ELO Calculation**: Standard chess ELO (K=32, initial=1500) from pairwise results

### Why ELO?

- Handles pairwise comparisons naturally
- Accounts for strength of opponents
- More robust than simple win rates
- Standard deviation shows consistency

## Project Structure

```
.
├── config.yaml              # Main configuration
├── pipeline/                # Evaluation pipeline
│   ├── stages/
│   │   ├── embed.py        # Document embedding
│   │   ├── retrieve.py     # FAISS retrieval
│   │   ├── rerank.py       # Reranker APIs
│   │   └── llm_judge.py    # GPT-4 judgments
│   ├── config.py
│   └── paths.py
├── datasets/                # Your datasets here
├── runs/                    # Experiment outputs
├── results.json             # Final results
└── generate_results.py      # Aggregate results
```


