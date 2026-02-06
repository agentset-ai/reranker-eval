# Adding a New Reranker

## Prerequisites

- Existing embeddings and retrieval results are reused
- Only new pairwise judgments are computed
- ELO ratings recalculated from all judgments

## Steps

### 1. Update Configuration

Add to `config.yaml`:

```yaml
rerankers:
  - name: "your-reranker"
    type: "your-type"
    model: "model-name"
    api_key_env: "YOUR_API_KEY"
    top_k: 15
```

Add to `model-info.json`:

```json
{
  "name": "your-reranker",
  "display_name": "Your Reranker",
  "provider": "Provider Name",
  "license": "License Type",
  "cost_per_1m_tokens": 0.0,
  "release_date": "YYYY-MM-DD",
  "about_model": "Description"
}
```

### 2. Set API Key

```bash
export YOUR_API_KEY="your-key"
```

### 3. Run Reranking

```bash
cd final-comparison

for dataset in msmarco arguana fiqa_small business-reports pg dbpedia scifact; do
  python add-reranker.py --dataset $dataset --reranker-name "your-reranker" --skip-evaluate
done
```

### 4. Run LLM Judge

```bash
for dataset in msmarco arguana fiqa_small business-reports pg dbpedia scifact; do
  python run-llm-judge-standalone.py --dataset $dataset
done
```

### 5. Aggregate Results

```bash
python aggregate-all-results.py
```

Results saved to `benchmarks.json`.

## Verification

```bash
# Check reranked files
ls runs/*/*/rerank/reranked_your-reranker.jsonl

# Check ELO ratings
cat runs/msmarco/*/llm_judge/elo_leaderboard.csv

# Check final results
cat benchmarks.json | jq '.[] | select(.name == "your-reranker")'
```

## What Gets Computed

**Reused:**
- Embeddings (BAAI/bge-small-en-v1.5)
- Retrieval results (top-50 per query)
- Existing pairwise judgments

**New:**
- Your reranker's results
- Pairwise judgments: your-reranker vs each existing reranker

**Recalculated:**
- ELO ratings from all judgments

## Notes

- 7 datasets: msmarco, arguana, fiqa_small, business-reports, pg, dbpedia, scifact
- LLM judge: Azure OpenAI GPT-5
- 50 queries per dataset
- ~550 new judgments per dataset (11 existing rerankers)
