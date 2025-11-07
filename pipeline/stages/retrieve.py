"""
Retrieval stage: Retrieve top-k documents for queries
"""

import json
import numpy as np
import faiss
from pathlib import Path
from typing import Dict, List
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import logging

from ..config import Config
from ..paths import RunPaths


def retrieve_stage(config: Config, paths: RunPaths, logger: logging.Logger) -> Dict:
    """
    Retrieve top-k documents for queries using FAISS
    
    Args:
        config: Pipeline configuration
        paths: Run paths manager
        logger: Logger instance
    
    Returns:
        Dictionary with stage results metadata
    """
    logger.info("Starting retrieval stage...")
    
    # Check if results already exist
    retrieval_file = paths.get_retrieval_file()
    if retrieval_file.exists() and config.pipeline.skip_if_exists:
        logger.info(f"Retrieval results already exist at {retrieval_file}, skipping...")
        with open(retrieval_file, 'r') as f:
            results = json.load(f)
        return {
            'status': 'skipped',
            'retrieval_file': str(retrieval_file),
            'num_queries': len(results)
        }
    
    # Load embedding model
    logger.info(f"Loading embedding model: {config.embedding.model}")
    print(f"   📥 Loading embedding model: {config.embedding.model}")
    model = SentenceTransformer(config.embedding.model)
    print(f"   ✓ Model loaded")
    
    # Load document embeddings
    embedding_file = paths.get_embedding_file(config.embedding.model)
    logger.info(f"Loading document embeddings from {embedding_file}")
    doc_data = np.load(embedding_file)
    doc_embeddings = doc_data['embeddings']
    doc_ids = doc_data['ids']
    logger.info(f"Loaded {len(doc_ids)} document embeddings")
    
    # Create FAISS index
    logger.info(f"Creating FAISS index (type: {config.retrieval.index_type})...")
    dimension = doc_embeddings.shape[1]
    
    if config.retrieval.index_type == "flat_ip":
        index = faiss.IndexFlatIP(dimension)
    else:
        raise ValueError(f"Unsupported index type: {config.retrieval.index_type}")
    
    index.add(doc_embeddings.astype('float32'))
    logger.info("FAISS index created")
    
    # Load queries
    logger.info(f"Loading queries from {config.dataset.queries_path}")
    with open(config.dataset.queries_path, 'r') as f:
        all_queries = [json.loads(line) for line in f]
    
    # Optionally subset queries
    if config.retrieval.query_subset:
        queries = all_queries[:config.retrieval.query_subset]
        logger.info(f"Processing subset of {len(queries)} queries (total: {len(all_queries)})")
        print(f"   📊 Processing {len(queries)} queries (subset of {len(all_queries)})")
    else:
        queries = all_queries
        logger.info(f"Processing all {len(queries)} queries")
        print(f"   📊 Processing all {len(queries)} queries")
    
    # Embed queries and retrieve
    logger.info(f"Retrieving top-{config.retrieval.top_k} documents for each query...")
    results = []
    
    for query in tqdm(queries, desc="Retrieving"):
        query_text = query['text']
        query_id = query['_id']
        
        # Embed query
        query_embedding = model.encode([query_text], normalize_embeddings=True)[0]
        
        # Search
        k = min(config.retrieval.top_k, len(doc_ids))
        scores, indices = index.search(
            query_embedding.reshape(1, -1).astype('float32'),
            k=k
        )
        
        # Format results
        retrieved_docs = []
        for score, idx in zip(scores[0], indices[0]):
            doc_id = doc_ids[idx]
            retrieved_docs.append({
                'doc_id': str(doc_id),
                'rank': len(retrieved_docs) + 1,
                'score': float(score)
            })
        
        results.append({
            'query_id': query_id,
            'query_text': query_text,
            'num_retrieved': len(retrieved_docs),
            'retrieved_docs': retrieved_docs
        })
    
    # Save results
    logger.info(f"Saving retrieval results to {retrieval_file}")
    with open(retrieval_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    result = {
        'status': 'completed',
        'retrieval_file': str(retrieval_file),
        'num_queries': len(results),
        'top_k': config.retrieval.top_k
    }
    
    logger.info("Retrieval stage completed successfully")
    return result

