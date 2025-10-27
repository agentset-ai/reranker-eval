#!/usr/bin/env python3
"""
Retrieve Top-K Candidates Script

Retrieves top-K candidates for each query using FAISS retrieval.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List
import logging

import numpy as np
from tqdm import tqdm
from fastembed import TextEmbedding
import faiss

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from tools.sample_beir_subset import load_corpus, load_queries


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def retrieve_candidates(query: str, index, doc_ids: List[str], embed_model: str, topk: int = 50) -> List[Dict]:
    """
    Retrieve top-k candidates for a query using FAISS.
    
    Args:
        query: The search query text
        index: FAISS index
        doc_ids: List of document IDs mapping FAISS indices
        embed_model: Embedding model name
        topk: Number of candidates to retrieve
        
    Returns:
        List of candidate dictionaries with doc_id, text, title, retrieval_score
    """
    model = TextEmbedding(model_name=embed_model)
    
    # Embed query
    query_embedding = np.array(list(model.embed([query]))[0]).astype('float32')
    faiss.normalize_L2(query_embedding.reshape(1, -1))
    
    # Search FAISS index
    scores, indices = index.search(query_embedding.reshape(1, -1), topk)
    
    # Convert to candidate format
    candidates = []
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx != -1:  # Valid index
            doc_id = doc_ids[idx]
            candidates.append({
                'doc_id': doc_id,
                'text': '',  # Will be filled from corpus
                'title': None,
                'retrieval_score': float(score),
                'rank': i + 1
            })
    
    return candidates


def main():
    parser = argparse.ArgumentParser(description="Retrieve top-K candidates for queries")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to BEIR dataset directory")
    parser.add_argument("--index_dir", type=str, required=True, help="Directory for FAISS index")
    parser.add_argument("--embed_model", type=str, required=True, help="Embedding model name")
    parser.add_argument("--topk", type=int, default=50, help="Number of candidates to retrieve")
    parser.add_argument("--num_queries", type=int, required=True, help="Number of queries to process")
    parser.add_argument("--out_file", type=str, required=True, help="Output JSONL file to save results")
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Convert paths
    dataset_dir = Path(args.dataset_dir)
    index_dir = Path(args.index_dir)
    out_file = Path(args.out_file)
    
    # Load dataset
    logging.info(f"Loading dataset from {dataset_dir}")
    corpus_path = dataset_dir / "corpus.jsonl"
    queries_path = dataset_dir / "queries.jsonl"
    
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
    if not queries_path.exists():
        raise FileNotFoundError(f"Queries file not found: {queries_path}")
    
    corpus = load_corpus(corpus_path)
    queries = load_queries(queries_path)
    
    logging.info(f"Loaded {len(corpus)} documents, {len(queries)} queries")
    
    # Load FAISS index
    index_path = index_dir / f"faiss_index_{args.embed_model.replace('/', '_')}.faiss"
    doc_ids_path = index_dir / f"doc_ids_{args.embed_model.replace('/', '_')}.json"
    
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not doc_ids_path.exists():
        raise FileNotFoundError(f"Document IDs file not found: {doc_ids_path}")
    
    logging.info(f"Loading FAISS index from {index_path}")
    index = faiss.read_index(str(index_path))
    with open(doc_ids_path, 'r') as f:
        doc_ids = json.load(f)
    
    logging.info(f"FAISS index loaded: {index.ntotal} documents")
    
    # Sample queries
    query_ids = list(queries.keys())[:args.num_queries]
    logging.info(f"Processing {len(query_ids)} queries")
    
    # Process each query
    results = []
    for query_id in tqdm(query_ids, desc="Retrieving candidates"):
        query_text = queries[query_id]['text']
        
        # Retrieve candidates
        candidates = retrieve_candidates(query_text, index, doc_ids, args.embed_model, args.topk)
        
        # Fill in document text and title from corpus
        for candidate in candidates:
            doc_id = candidate['doc_id']
            if doc_id in corpus:
                candidate['text'] = corpus[doc_id].get('text', '')
                candidate['title'] = corpus[doc_id].get('title')
        
        # Store result
        results.append({
            'query_id': query_id,
            'query_text': query_text,
            'candidates': candidates
        })
    
    # Save results
    logging.info(f"Saving results to {out_file}")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    logging.info(f"Successfully saved {len(results)} query results to {out_file}")
    logging.info(f"Each query has {args.topk} retrieved candidates")


if __name__ == "__main__":
    main()


