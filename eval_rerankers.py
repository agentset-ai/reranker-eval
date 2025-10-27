#!/usr/bin/env python3
"""
FAISS Index Creation Utility

Creates and manages FAISS indexes for document embeddings.
Used by retrieval and reranking scripts.
"""

import json
from pathlib import Path
from typing import Dict
import logging

import numpy as np
from fastembed import TextEmbedding
import faiss


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def create_or_load_faiss_index(corpus: Dict, embed_model: str, index_dir: Path, topk: int = 50):
    """Create or load FAISS index for retrieval.
    
    Args:
        corpus: Dictionary of documents (doc_id -> doc_dict)
        embed_model: Name of the embedding model
        index_dir: Directory to store/load the index
        topk: Number of candidates to retrieve (for logging)
        
    Returns:
        Tuple of (index, doc_ids)
    """
    index_path = index_dir / f"faiss_index_{embed_model.replace('/', '_')}.faiss"
    doc_ids_path = index_dir / f"doc_ids_{embed_model.replace('/', '_')}.json"
    
    # Create index directory if it doesn't exist
    index_dir.mkdir(parents=True, exist_ok=True)
    
    if index_path.exists() and doc_ids_path.exists():
        logging.info(f"Loading existing FAISS index from {index_path}")
        index = faiss.read_index(str(index_path))
        with open(doc_ids_path, 'r') as f:
            doc_ids = json.load(f)
        logging.info(f"Loaded index with {index.ntotal} documents")
        return index, doc_ids
    
    logging.info(f"Creating FAISS index with model {embed_model}")
    
    # Initialize embedding model
    model = TextEmbedding(model_name=embed_model)
    
    # Prepare documents for embedding
    documents = []
    doc_ids = []
    
    for doc_id, doc in corpus.items():
        # Combine title and text for embedding
        text = doc.get('text', '')
        if doc.get('title'):
            text = f"{doc['title']}\n{text}"
        documents.append(text)
        doc_ids.append(doc_id)
    
    logging.info(f"Embedding {len(documents)} documents...")
    
    # Generate embeddings
    embeddings = list(model.embed(documents))
    embeddings = np.array(embeddings).astype('float32')
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Add embeddings to index
    index.add(embeddings)
    
    # Save index and doc_ids
    faiss.write_index(index, str(index_path))
    with open(doc_ids_path, 'w') as f:
        json.dump(doc_ids, f)
    
    logging.info(f"Saved FAISS index to {index_path} with {index.ntotal} documents")
    return index, doc_ids


if __name__ == "__main__":
    # This file is now primarily imported as a utility
    # The main logic has been moved to final_evaluation/evaluate_reranker.py
    setup_logging()
    logging.info("This module provides the create_or_load_faiss_index function.")
    logging.info("Import it in your scripts to create or load FAISS indexes.")
