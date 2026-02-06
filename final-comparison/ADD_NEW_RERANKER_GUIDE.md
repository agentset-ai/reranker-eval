# Guide: Adding a New Reranker

This guide shows you how to add a new reranker to all datasets, reuse existing judgments, and calculate ELO ratings.

## Overview

- **6 datasets**: msmarco, arguana, fiqa_small, business-reports, pg, dbpedia
- **Embeddings**: Should already exist (they're reused automatically)
- **Judgments**: Existing judgments are reused; only new pairs involving the new reranker are judged
- **ELO**: Recalculated from all judgments (existing + new)

## Step-by-Step Instructions

### 1. Add the Reranker to config.yaml

Edit `config.yaml` and add your new reranker to the `rerankers` list:

```yaml
rerankers:
  # ... existing rerankers ...
  - name: "your-new-reranker"
    type: "your-reranker-type"  # e.g., "zerank", "voyage", "cohere", etc.
    model: "your-model-name"
    api_key_env: "YOUR_API_KEY_ENV"
    top_k: 15
```

### 2. Set API Key (if needed)

If your reranker requires an API key:

```bash
export YOUR_API_KEY_ENV="your-api-key-here"
```

### 3. Add Reranker to Each Dataset

```bash
cd final-comparison

# Add to each dataset (embeddings & retrieval are reused automatically)
python add-reranker.py --dataset msmarco --reranker-name "your-new-reranker" --skip-evaluate
python add-reranker.py --dataset arguana --reranker-name "your-new-reranker" --skip-evaluate
python add-reranker.py --dataset fiqa_small --reranker-name "your-new-reranker" --skip-evaluate
python add-reranker.py --dataset business-reports --reranker-name "your-new-reranker" --skip-evaluate
python add-reranker.py --dataset pg --reranker-name "your-new-reranker" --skip-evaluate
python add-reranker.py --dataset dbpedia --reranker-name "your-new-reranker" --skip-evaluate
```

### 4. Run LLM Judge (Reuses Existing Judgments)

```bash
# Run judge for each dataset (automatically reuses existing judgments)
python run-llm-judge-standalone.py --dataset msmarco
python run-llm-judge-standalone.py --dataset arguana
python run-llm-judge-standalone.py --dataset fiqa_small
python run-llm-judge-standalone.py --dataset business-reports
python run-llm-judge-standalone.py --dataset pg
python run-llm-judge-standalone.py --dataset dbpedia
```

### 5. Aggregate Results

After all datasets are processed, aggregate the results:

```bash
python aggregate-all-results.py
```

This creates `benchmarks.json` with:
- Overall ELO ratings across all datasets
- Per-dataset ELO ratings
- Win/loss/tie statistics
- Comparison matrices between rerankers

## What Gets Reused vs. What's New

### ✅ Reused (No Re-computation):
- **Embeddings**: Already exist in `runs/{dataset}/{timestamp}/embeddings/`
- **Retrieval**: Already exists in `runs/{dataset}/{timestamp}/retrieval/`
- **Old Reranker Results**: All existing reranked files are kept
- **Old Judgments**: All judgments between existing rerankers are reused

### 🆕 New (Computed):
- **New Reranker Results**: Only your new reranker is run
- **New Judgments**: Only pairs involving your new reranker are judged
  - Example: If you add "reranker-x", new judgments are made for:
    - reranker-x vs. zerank1
    - reranker-x vs. zerank2
    - reranker-x vs. voyage
    - ... (all pairs with reranker-x)
  - Old pairs (e.g., zerank1 vs. voyage) are reused

### 🔄 Recalculated:
- **ELO Ratings**: Recalculated from ALL judgments (existing + new) in order

## Verification

Check that everything worked:

```bash
# Check reranked results exist
ls runs/*/latest/rerank/reranked_your-new-reranker.jsonl

# Check judgments were made/reused
ls runs/*/latest/llm_judge/judgments.jsonl

# Check ELO ratings
cat runs/*/latest/llm_judge/elo_leaderboard.csv

# Check aggregated results
cat benchmarks.json | jq '.[] | select(.name == "your-new-reranker")'
```

## Troubleshooting

### Reranker Already Exists
If the reranker was already added, you can delete the existing reranked file and re-run the add-reranker.py script.

### Missing Embeddings
If embeddings don't exist, you'll need to run the full pipeline once. But typically they should already exist from previous runs.

### API Key Issues
Make sure your API key environment variable is set:
```bash
echo $YOUR_API_KEY_ENV
```

## Example: Adding "my-reranker"

```bash
# 1. Add to config.yaml (edit manually)
# 2. Set API key
export MY_RERANKER_API_KEY="key-here"

# 3. Add reranker to each dataset
cd final-comparison
for dataset in msmarco arguana fiqa_small business-reports pg dbpedia; do
  python add-reranker.py --dataset $dataset --reranker-name "my-reranker" --skip-evaluate
done

# 4. Run LLM judge for each dataset
for dataset in msmarco arguana fiqa_small business-reports pg dbpedia; do
  python run-llm-judge-standalone.py --dataset $dataset
done

# 5. Aggregate results
python aggregate-all-results.py

# 6. Check results
cat benchmarks.json | jq '.[] | select(.name == "my-reranker")'
```

## Notes

- The LLM judge uses Azure OpenAI GPT-5 (as configured in config.yaml)
- Judgments are made in parallel (10 workers) for speed
- ELO is recalculated from scratch using all judgments to ensure consistency
- The process preserves all existing work - nothing is deleted or overwritten unnecessarily
