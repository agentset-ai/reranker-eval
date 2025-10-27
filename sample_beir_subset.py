#!/usr/bin/env python3
"""
BEIR Dataset Sampling Utility

Creates a smaller subset of a BEIR dataset by sampling queries and their relevant documents,
optionally adding random documents up to a corpus cap.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Set


def load_corpus(corpus_path: Path) -> Dict[str, Dict]:
    """Load corpus.jsonl into a dictionary."""
    corpus = {}
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line.strip())
            corpus[doc['_id']] = doc
    return corpus


def load_queries(queries_path: Path) -> Dict[str, Dict]:
    """Load queries.jsonl into a dictionary."""
    queries = {}
    with open(queries_path, 'r', encoding='utf-8') as f:
        for line in f:
            query = json.loads(line.strip())
            queries[query['_id']] = query
    return queries


def load_qrels(qrels_path: Path) -> Dict[str, Dict[str, int]]:
    """Load qrels/test.tsv into a dictionary."""
    qrels = {}
    with open(qrels_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            # Skip header line
            if parts[0] == 'query-id':
                continue
            
            # Handle different BEIR formats
            if len(parts) >= 3:
                if len(parts) == 3:
                    # Format: query-id, corpus-id, score
                    qid, doc_id, relevance = parts[0], parts[1], int(parts[2])
                elif len(parts) >= 4:
                    # Format: query-id, corpus-id, corpus-id, rel
                    qid, doc_id, relevance = parts[0], parts[2], int(parts[3])
                else:
                    continue
                
                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][doc_id] = relevance
    return qrels


def sample_queries(queries, num_queries, seed=42, qrels=None):
    random.seed(seed)
    qids = list(queries.keys())
    if qrels is not None:
        qids = [q for q in qids if q in qrels and any(v > 0 for v in qrels[q].values())]
    return random.sample(qids, min(num_queries, len(qids)))


def get_relevant_docs(sampled_query_ids: List[str], qrels: Dict[str, Dict[str, int]]) -> Set[str]:
    """Get all document IDs that are relevant to the sampled queries."""
    relevant_docs = set()
    for qid in sampled_query_ids:
        if qid in qrels:
            relevant_docs.update(qrels[qid].keys())
    return relevant_docs


def sample_corpus(corpus, relevant_docs, corpus_cap=None, seed=42):
    random.seed(seed)
    sampled = {doc_id: corpus[doc_id] for doc_id in relevant_docs if doc_id in corpus}

    if corpus_cap is not None and corpus_cap < len(sampled):
        print(f"[warn] corpus_cap ({corpus_cap}) < #gold docs ({len(sampled)}). Keeping all gold docs and ignoring cap.")
        return sampled

    if corpus_cap and len(sampled) < corpus_cap:
        remaining = [d for d in corpus.keys() if d not in sampled]
        need = corpus_cap - len(sampled)
        for doc_id in random.sample(remaining, min(need, len(remaining))):
            sampled[doc_id] = corpus[doc_id]
    return sampled


def save_corpus(corpus: Dict[str, Dict], output_path: Path):
    """Save corpus to jsonl format."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for doc in corpus.values():
            f.write(json.dumps(doc) + '\n')


def save_queries(queries: Dict[str, Dict], sampled_query_ids: List[str], output_path: Path):
    """Save sampled queries to jsonl format."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for qid in sampled_query_ids:
            if qid in queries:
                f.write(json.dumps(queries[qid]) + '\n')


def save_qrels(qrels: Dict[str, Dict[str, int]], sampled_query_ids: List[str], output_path: Path):
    """Save filtered qrels to tsv format."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("query-id\tcorpus-id\tcorpus-id\trel\n")  # BEIR header format
        for qid in sampled_query_ids:
            if qid in qrels:
                for doc_id, relevance in qrels[qid].items():
                    f.write(f"{qid}\t{doc_id}\t{doc_id}\t{relevance}\n")


def main():
    parser = argparse.ArgumentParser(description="Sample a BEIR dataset subset")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to BEIR dataset directory")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for subset")
    parser.add_argument("--num_queries", type=int, required=True, help="Number of queries to sample")
    parser.add_argument("--corpus_cap", type=int, help="Maximum number of documents in corpus")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    
    # Validate input directory
    corpus_path = dataset_dir / "corpus.jsonl"
    queries_path = dataset_dir / "queries.jsonl"
    qrels_path = dataset_dir / "qrels" / "test.tsv"
    
    if not corpus_path.exists():
        print(f"Error: {corpus_path} not found")
        sys.exit(1)
    if not queries_path.exists():
        print(f"Error: {queries_path} not found")
        sys.exit(1)
    if not qrels_path.exists():
        print(f"Error: {qrels_path} not found")
        sys.exit(1)
    
    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "qrels").mkdir(parents=True, exist_ok=True)
    
    print(f"Loading dataset from {dataset_dir}")
    
    # Load data
    corpus = load_corpus(corpus_path)
    queries = load_queries(queries_path)
    qrels = load_qrels(qrels_path)
    
    print(f"Loaded {len(corpus)} documents, {len(queries)} queries, {len(qrels)} query-document pairs")
    
    # Sample queries
    sampled_query_ids = sample_queries(queries, args.num_queries, args.seed)
    print(f"Sampled {len(sampled_query_ids)} queries")
    
    # Get relevant documents
    relevant_docs = get_relevant_docs(sampled_query_ids, qrels)
    print(f"Found {len(relevant_docs)} relevant documents")
    
    # Sample corpus
    sampled_corpus = sample_corpus(corpus, relevant_docs, args.corpus_cap, args.seed)
    print(f"Sampled {len(sampled_corpus)} documents for corpus")
    
    # Save outputs
    save_corpus(sampled_corpus, out_dir / "corpus.jsonl")
    save_queries(queries, sampled_query_ids, out_dir / "queries.jsonl")
    save_qrels(qrels, sampled_query_ids, out_dir / "qrels" / "test.tsv")
    
    print(f"Saved subset to {out_dir}")
    print(f"- corpus.jsonl: {len(sampled_corpus)} documents")
    print(f"- queries.jsonl: {len(sampled_query_ids)} queries")
    print(f"- qrels/test.tsv: {sum(len(qrels.get(qid, {})) for qid in sampled_query_ids)} query-document pairs")


if __name__ == "__main__":
    main()

