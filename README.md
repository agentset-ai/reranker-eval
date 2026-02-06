# Reranker Leaderboard

Evaluation of 12 reranking models using LLM-as-judge pairwise comparisons across 6 datasets.

## Leaderboard

| Rank | Model | ELO | Win Rate |
|------|-------|-----|----------|
| 1 | Zerank 2 | 1638 | 57% |
| 2 | Cohere Rerank 4 Pro | 1629 | 58% |
| 3 | Zerank 1 | 1573 | 57% |
| 4 | Voyage AI Rerank 2.5 | 1544 | 58% |
| 5 | Zerank 1 Small | 1539 | 55% |
| 6 | Voyage AI Rerank 2.5 Lite | 1520 | 53% |
| 7 | Cohere Rerank 4 Fast | 1510 | 50% |
| 8 | Qwen3 Reranker 8B | 1473 | 51% |
| 9 | Contextual AI Rerank v2 Instruct | 1469 | 42% |
| 10 | Cohere Rerank 3.5 | 1451 | 41% |
| 11 | BAAI/BGE Reranker v2 M3 | 1327 | 29% |
| 12 | Jina Reranker v2 Base Multilingual | 1327 | 28% |


## Datasets

- MSMARCO (web search)
- Arguana (argument mining)
- FiQa (financial Q&A)
- Business Reports
- Paul Graham Essays
- DBPedia (entity retrieval)

## Methodology

1. Embed documents with BAAI/bge-small-en-v1.5
2. Retrieve top-50 candidates using FAISS
3. Rerank to top-15 with each model
4. Generate pairwise judgments using GPT-5
5. Calculate ELO ratings from pairwise comparisons

## Usage

See [eval-pipeline/ADD_NEW_RERANKER_GUIDE.md](eval-pipeline/ADD_NEW_RERANKER_GUIDE.md) for instructions on adding new rerankers.

## Project Structure

```
eval-pipeline/
├── config.yaml              Configuration
├── pipeline/                Evaluation pipeline
│   └── stages/
│       ├── embed.py         Document embedding
│       ├── retrieve.py      FAISS retrieval
│       ├── rerank.py        Reranker integrations
│       └── llm-judge.py     LLM-as-judge evaluation
├── add-reranker.py          Add new reranker
├── compare-rerankers.py     Compare all rerankers (ELO)
└── aggregate-all-results.py Aggregate cross-dataset results
```
