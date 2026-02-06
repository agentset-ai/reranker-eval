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

### 3. Run the Automated Script

The easiest way is to use the automated script that does everything:

```bash
cd final-comparison
python add_reranker_and_judge.py --reranker-name "your-new-reranker"
```

This script will:
- ✅ Add the reranker to all 6 datasets (reuses existing embeddings & retrieval)
- ✅ Run LLM judge which:
  - Reuses all existing judgments between old rerankers
  - Only makes new judgments for pairs involving your new reranker
  - Recalculates ELO from all judgments (existing + new)

### 4. Alternative: Manual Steps

If you prefer to run steps manually:

#### Step 4a: Add Reranker to Each Dataset

```bash
cd final-comparison

# Add to each dataset (embeddings & retrieval are reused automatically)
python add_reranker.py --dataset msmarco --reranker-name "your-new-reranker" --skip-evaluate
python add_reranker.py --dataset arguana --reranker-name "your-new-reranker" --skip-evaluate
python add_reranker.py --dataset fiqa_small --reranker-name "your-new-reranker" --skip-evaluate
python add_reranker.py --dataset business-reports --reranker-name "your-new-reranker" --skip-evaluate
python add_reranker.py --dataset pg --reranker-name "your-new-reranker" --skip-evaluate
python add_reranker.py --dataset dbpedia --reranker-name "your-new-reranker" --skip-evaluate
```

#### Step 4b: Run LLM Judge (Reuses Existing Judgments)

```bash
# Run judge for each dataset (automatically reuses existing judgments)
python run_llm_judge_standalone.py --dataset msmarco
python run_llm_judge_standalone.py --dataset arguana
python run_llm_judge_standalone.py --dataset fiqa_small
python run_llm_judge_standalone.py --dataset business-reports
python run_llm_judge_standalone.py --dataset pg
python run_llm_judge_standalone.py --dataset dbpedia
```

### 5. Aggregate Results

After all datasets are processed, aggregate the results:

```bash
python aggregate_all_results.py
```

This creates `results_all_datasets.json` with:
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
cat results_all_datasets.json | jq '.[] | select(.name == "your-new-reranker")'
```

## Troubleshooting

### Reranker Already Exists
If you see "Reranker already exists, skipping", the reranker was already added. You can:
- Skip the add step: `python add_reranker_and_judge.py --reranker-name "name" --skip-add`
- Or delete the existing file and re-run

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

# 3. Run automated script
cd final-comparison
python add_reranker_and_judge.py --reranker-name "my-reranker"

# 4. Aggregate results
python aggregate_all_results.py

# 5. Check results
cat results_all_datasets.json | jq '.[] | select(.name == "my-reranker")'
```

## Notes

- The LLM judge uses Azure OpenAI GPT-5 (as configured in config.yaml)
- Judgments are made in parallel (10 workers) for speed
- ELO is recalculated from scratch using all judgments to ensure consistency
- The process preserves all existing work - nothing is deleted or overwritten unnecessarily
