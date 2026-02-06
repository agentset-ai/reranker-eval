"""
Embedding stage: Generate embeddings for corpus
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import logging

from ..config import Config
from ..paths import RunPaths


def embed_stage(config: Config, paths: RunPaths, logger: logging.Logger) -> Dict:
    """
    Generate embeddings for corpus documents
    
    Args:
        config: Pipeline configuration
        paths: Run paths manager
        logger: Logger instance
    
    Returns:
        Dictionary with stage results metadata
    """
    logger.info("Starting embedding stage...")
    
    # Check if embeddings already exist in current run directory
    embedding_file = paths.get_embedding_file(config.embedding.model)
    if embedding_file.exists() and config.pipeline.skip_if_exists:
        logger.info(f"Embeddings already exist at {embedding_file}, skipping...")
        data = np.load(embedding_file)
        return {
            'status': 'skipped',
            'embedding_file': str(embedding_file),
            'num_documents': len(data['ids']),
            'embedding_dim': data['embeddings'].shape[1]
        }
    
    # Check if embeddings exist in dataset directory (from previous runs or old scripts)
    if config.pipeline.skip_if_exists:
        dataset_dir = Path(config.dataset.base_path)
        # Try common naming patterns
        possible_names = [
            f"embeddings_{config.embedding.model.replace('/', '_').replace('-', '_')}.npz",
            f"embeddings_bge_small_en_v1.5.npz",  # Common pattern
        ]
        # Also check for any .npz file in dataset directory
        for npz_file in dataset_dir.glob("embeddings*.npz"):
            if npz_file.exists():
                logger.info(f"Found existing embeddings at {npz_file}, copying to run directory...")
                print(f"   📋 Found existing embeddings, reusing from {npz_file.name}...")
                # Copy to new location
                import shutil
                embedding_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(npz_file, embedding_file)
                data = np.load(embedding_file)
                return {
                    'status': 'skipped',
                    'embedding_file': str(embedding_file),
                    'num_documents': len(data['ids']),
                    'embedding_dim': data['embeddings'].shape[1],
                    'copied_from': str(npz_file)
                }
    
    # Load embedding model
    logger.info(f"Loading embedding model: {config.embedding.model}")
    print(f"   📥 Loading embedding model: {config.embedding.model}")
    model = SentenceTransformer(config.embedding.model)
    logger.info("Model loaded successfully")
    print(f"   ✓ Model loaded successfully")
    
    # Load corpus
    logger.info(f"Loading corpus from {config.dataset.corpus_path}")
    corpus_texts = []
    corpus_ids = []
    
    with open(config.dataset.corpus_path, 'r') as f:
        for line in f:
            doc = json.loads(line)
            corpus_ids.append(doc['_id'])
            # Combine title and text
            text = doc.get('title', '') + ' ' + doc.get('text', '')
            corpus_texts.append(text.strip())
    
    logger.info(f"Loaded {len(corpus_texts)} documents")
    print(f"   📚 Loaded {len(corpus_texts)} documents")
    
    # Generate embeddings
    logger.info(f"Generating embeddings (batch_size={config.embedding.batch_size})...")
    print(f"   ⚙️  Generating embeddings (batch_size={config.embedding.batch_size})...")
    embeddings = []
    
    for i in tqdm(range(0, len(corpus_texts), config.embedding.batch_size), desc="Embedding"):
        batch = corpus_texts[i:i+config.embedding.batch_size]
        batch_embeddings = model.encode(
            batch,
            normalize_embeddings=config.embedding.normalize,
            show_progress_bar=False
        )
        embeddings.append(batch_embeddings)
    
    # Combine all embeddings
    embeddings = np.vstack(embeddings)
    logger.info(f"Generated embeddings with shape: {embeddings.shape}")
    
    # Save embeddings
    logger.info(f"Saving embeddings to {embedding_file}")
    np.savez_compressed(
        embedding_file,
        embeddings=embeddings,
        ids=corpus_ids
    )
    
    result = {
        'status': 'completed',
        'embedding_file': str(embedding_file),
        'num_documents': len(corpus_ids),
        'embedding_dim': embeddings.shape[1],
        'model': config.embedding.model
    }
    
    logger.info("Embedding stage completed successfully")
    return result

