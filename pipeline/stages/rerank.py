"""
Reranking stage: Rerank retrieval results using multiple reranker APIs
"""

import json
import time
import csv
import os
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
import requests
import matplotlib.pyplot as plt
import logging


from ..config import Config, RerankerConfig
from ..paths import RunPaths

# Lazy import transformers
_transformers_cache = {}
def _get_model_tokenizer(model_name: str):
    """Lazy load model and tokenizer"""
    if model_name not in _transformers_cache:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Ensure padding token is set for batching
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            elif tokenizer.unk_token is not None:
                tokenizer.pad_token = tokenizer.unk_token
                tokenizer.pad_token_id = tokenizer.unk_token_id
            else:
                # Add a new padding token
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                # Resize model embeddings to include new token
                model.resize_token_embeddings(len(tokenizer))
        
        # Ensure pad_token_id is set if pad_token exists
        if tokenizer.pad_token is not None and tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        
        _transformers_cache[model_name] = {
            'tokenizer': tokenizer,
            'model': model
        }
        _transformers_cache[model_name]['model'].eval()
    return _transformers_cache[model_name]['tokenizer'], _transformers_cache[model_name]['model']


@dataclass
class RerankResult:
    """Rerank result dataclass"""
    doc_id: str
    rank: int
    score: float


def _cohere_rerank(query: str, documents: List[str], reranker: RerankerConfig, top_k: int = 15) -> Tuple[List[RerankResult], float]:
    """Rerank using Cohere API"""
    headers = {
        'Authorization': f'Bearer {reranker.api_key}',
        'Content-Type': 'application/json'
    }
    payload = {
        'model': reranker.model,
        'query': query,
        'documents': documents,
        'top_n': top_k,
        'return_documents': False
    }
    start_time = time.time()
    try:
        response = requests.post('https://api.cohere.com/v1/rerank', json=payload, headers=headers, timeout=30)
        latency = time.time() - start_time
        if response.status_code != 200:
            logging.warning(f"Error in Cohere: {response.status_code} - {response.text[:200]}")
            return [], latency
        data = response.json()
        results = []
        for i, result in enumerate(data.get('results', [])):
            results.append(RerankResult(
                doc_id=str(result['index']),
                rank=i + 1,
                score=result.get('relevance_score', 0)
            ))
        return results, latency
    except Exception as e:
        latency = time.time() - start_time
        logging.error(f"Error in Cohere: {e}")
        return [], latency


def _zerank_rerank(query: str, documents: List[str], reranker: RerankerConfig, top_k: int = 15) -> Tuple[List[RerankResult], float]:
    """Rerank using ZeroEntropy API"""
    headers = {
        'Authorization': f'Bearer {reranker.api_key}',
        'Content-Type': 'application/json'
    }
    payload = {
        'model': reranker.model,
        'query': query,
        'documents': documents,
        'top_n': top_k
    }
    base_urls = [
        os.getenv('ZERANK_BASE', 'https://api.zeroentropy.dev'),
        'https://eu-api.zeroentropy.dev',
        'https://api.zenml.io'
    ]
    start_time = time.time()
    response = None
    last_error = None
    for base in base_urls:
        try:
            test_resp = requests.post(f'{base}/v1/models/rerank', json=payload, headers=headers, timeout=30)
            if test_resp.status_code == 200:
                response = test_resp
                break
            else:
                last_error = f"{test_resp.status_code}: {test_resp.text[:100]}"
        except Exception as e:
            last_error = str(e)
            continue
    latency = time.time() - start_time
    if response is None:
        logging.error(f"Error in {reranker.name}: Could not connect. Last error: {last_error}")
        return [], latency
    if response.status_code != 200:
        logging.error(f"Error in {reranker.name}: {response.status_code} - {response.text[:200]}")
        return [], latency
    data = response.json()
    results = []
    for i, result in enumerate(data.get('results', [])):
        results.append(RerankResult(
            doc_id=str(result['index']),
            rank=i + 1,
            score=result.get('relevance_score', result.get('score', 0))
        ))
    return results, latency


def _jina_rerank(query: str, documents: List[str], reranker: RerankerConfig, top_k: int = 15) -> Tuple[List[RerankResult], float]:
    """Rerank using Jina API"""
    url = 'https://api.jina.ai/v1/rerank'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {reranker.api_key}'
    }
    data = {
        "model": reranker.model,
        "query": query,
        "top_n": top_k,
        "documents": documents,
        "return_documents": False
    }
    start_time = time.time()
    response = requests.post(url, headers=headers, json=data, timeout=30)
    latency = time.time() - start_time
    if response.status_code != 200:
        logging.error(f"Error in Jina: {response.status_code} - {response.text[:300]}")
        return [], latency
    result_data = response.json()
    results = []
    for i, result in enumerate(result_data.get('results', [])):
        results.append(RerankResult(
            doc_id=str(result['index']),
            rank=i + 1,
            score=result.get('relevance_score', result.get('score', 0))
        ))
    return results, latency


def _voyage_rerank(query: str, documents: List[str], reranker: RerankerConfig, top_k: int = 15) -> Tuple[List[RerankResult], float]:
    """Rerank using Voyage AI API"""
    headers = {
        'Authorization': f'Bearer {reranker.api_key}',
        'Content-Type': 'application/json'
    }
    payload = {
        'model': reranker.model,
        'query': query,
        'documents': documents,
        'top_k': top_k
    }
    start = time.time()
    resp = requests.post('https://api.voyageai.com/v1/rerank', json=payload, headers=headers, timeout=30)
    latency = time.time() - start
    if resp.status_code != 200:
        logging.error(f"Error in {reranker.name}: {resp.status_code} - {resp.text[:300]}")
        return [], latency
    data = resp.json()
    results = []
    for i, item in enumerate(data.get('data') or data.get('results') or []):
        idx = item.get('index', i)
        score = item.get('score', item.get('relevance_score', 0))
        results.append(RerankResult(doc_id=str(idx), rank=i+1, score=float(score)))
    return results, latency


def _together_rerank(query: str, documents: List[str], reranker: RerankerConfig, top_k: int = 15) -> Tuple[List[RerankResult], float]:
    """Rerank using Together AI API (for custom uploaded models like Qwen3-Reranker-8B)
    
    Together AI provides OpenAI-compatible API for custom models.
    For reranking, we use the chat completion endpoint with specific prompts.
    """
    headers = {
        'Authorization': f'Bearer {reranker.api_key}',
        'Content-Type': 'application/json'
    }
    
    # Build system prompt for the reranker
    system_prompt = (
        "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
        "Note that the answer can only be \"yes\" or \"no\"."
    )
    
    start_time = time.time()
    results = []
    
    try:
        scored_indices = []
        
        # Process each document with the Qwen3-Reranker prompt format
        for i, doc in enumerate(documents):
            # Construct the reranking prompt in the Qwen format
            user_prompt = f"<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n<Query>: {query}\n<Document>: {doc}"
            
            # Send request to Together AI chat completions endpoint
            payload = {
                'model': reranker.model,
                'messages': [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                'max_tokens': 1,
                'temperature': 0,
                'stop': ['</thinking>', '</thinking\n']
            }
            
            response = requests.post(
                'https://api.together.xyz/v1/chat/completions',
                json=payload,
                headers=headers,
                timeout=120
            )
            
            if response.status_code != 200:
                logging.error(f"Error in Together AI ({reranker.name}) for doc {i}: {response.status_code} - {response.text[:300]}")
                # Assign neutral score on error
                scored_indices.append((i, 0.5))
                continue
            
            data = response.json()
            choices = data.get('choices', [])
            
            if not choices:
                logging.warning(f"No response for document {i}")
                scored_indices.append((i, 0.5))
                continue
            
            content = choices[0].get('message', {}).get('content', '').strip().lower()
            
            # Score is 1.0 for "yes", 0.0 for "no", 0.5 for ambiguous
            if content == 'yes' or 'yes' in content:
                score = 1.0
            elif content == 'no' or 'no' in content:
                score = 0.0
            else:
                score = 0.5  # Default for ambiguous responses
                logging.warning(f"Unexpected response for doc {i}: {content}")
            
            scored_indices.append((i, score))
        
        # Sort by score and take top_k
        scored_indices.sort(key=lambda x: x[1], reverse=True)
        
        for rank, (idx, score) in enumerate(scored_indices[:top_k], 1):
            results.append(RerankResult(doc_id=str(idx), rank=rank, score=score))
        
        latency = time.time() - start_time
        return results, latency
        
    except Exception as e:
        latency = time.time() - start_time
        logging.error(f"Error in Together AI ({reranker.name}): {e}")
        import traceback
        logging.error(traceback.format_exc())
        return [], latency

def _contextual_rerank(query: str, documents: List[str], reranker: RerankerConfig, top_k: int = 15) -> Tuple[List[RerankResult], float]:
    """Rerank using Contextual AI API"""
    try:
        from contextual import ContextualAI
    except ImportError:
        logging.error("contextual-client not installed. Run: pip install contextual-client")
        return [], 0.0

    client = ContextualAI(api_key=reranker.api_key)

    start_time = time.time()
    try:
        result = client.rerank.create(
            query=query,
            instruction="Given a web search query, retrieve relevant passages that answer the query",
            documents=documents,
            model=reranker.model
        )
        latency = time.time() - start_time

        # Parse results - the API returns results with relevance_score, not score
        results = []
        for i, item in enumerate(result.results[:top_k]):
            # Try different attribute names
            score = getattr(item, 'relevance_score', None) or getattr(item, 'score', None) or 0.0
            idx = getattr(item, 'index', i)

            results.append(RerankResult(
                doc_id=str(idx),
                rank=i + 1,
                score=score
            ))
        return results, latency

    except Exception as e:
        latency = time.time() - start_time
        logging.error(f"Error in Contextual AI: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return [], latency


def _replicate_rerank(query: str, documents: List[str], reranker: RerankerConfig, top_k: int = 15) -> Tuple[List[RerankResult], float]:
    """Rerank using Replicate API for BAAI/bge-reranker-v2-m3"""
    try:
        import replicate
    except ImportError:
        logging.error("replicate not installed. Run: pip install replicate")
        return [], 0.0

    start_time = time.time()
    try:
        # Format input as list of [query, document] pairs
        input_pairs = [[query, doc] for doc in documents]
        input_json = json.dumps(input_pairs)

        # Run the model
        output = replicate.run(
            reranker.model,  # Should be in format: "yxzwayne/bge-reranker-v2-m3:7f7c6e9d18336e2cbf07d88e9362d881d2fe4d6a9854ec1260f115cabc106a8c"
            input={"input_list": input_json}
        )

        latency = time.time() - start_time

        # Parse results - output is a list of scores
        scores = list(output) if hasattr(output, '__iter__') else [output]

        # Create results with scores and sort by score descending
        results = []
        for i, score in enumerate(scores):
            results.append(RerankResult(
                doc_id=str(i),
                rank=0,  # Will be set after sorting
                score=float(score)
            ))

        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)

        # Update ranks and return top_k
        for rank, result in enumerate(results[:top_k], 1):
            result.rank = rank

        return results[:top_k], latency

    except Exception as e:
        latency = time.time() - start_time
        logging.error(f"Error in Replicate: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return [], latency


def _qwen_rerank(query: str, documents: List[str], reranker: RerankerConfig, top_k: int = 15) -> Tuple[List[RerankResult], float]:
    """Rerank using Qwen3-Reranker model"""
    import torch
    import torch.nn.functional as F
    
    start_time = time.time()
    
    try:
        # Get or load model
        if reranker.model not in _transformers_cache:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            tokenizer = AutoTokenizer.from_pretrained(reranker.model, padding_side='left')
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(reranker.model).eval()
            device = next(model.parameters()).device
            
            suffix = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            max_length = 8192
            suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
            true_token = tokenizer("yes", add_special_tokens=False).input_ids[0]
            false_token = tokenizer("no", add_special_tokens=False).input_ids[0]
            
            _transformers_cache[reranker.model] = {
                'tokenizer': tokenizer,
                'model': model,
                'device': device,
                'true_token': true_token,
                'false_token': false_token,
                'max_length': max_length,
                'suffix_tokens': suffix_tokens
            }
        
        # Get cached values
        cache = _transformers_cache[reranker.model]
        tokenizer = cache['tokenizer']
        model = cache['model']
        device = cache['device']
        true_token = cache['true_token']
        false_token = cache['false_token']
        max_length = cache['max_length']
        suffix_tokens = cache['suffix_tokens']
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        
        # Format messages
        messages = []
        for doc in documents:
            msg = [
                {"role": "system", "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."},
                {"role": "user", "content": f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}"}
            ]
            messages.append(msg)
        
        # Process in smaller batches to avoid OOM on CPU
        batch_size = 5
        all_scores = []
        
        for i in range(0, len(messages), batch_size):
            batch_messages = messages[i:i+batch_size]
            
            # Apply chat template - returns list of token IDs for each message
            tokenized_messages = []
            for msg in batch_messages:
                tokens = tokenizer.apply_chat_template(
                    msg, tokenize=True, add_generation_prompt=False, enable_thinking=False
                )
                # Truncate if needed and add suffix
                if len(tokens) > max_length - len(suffix_tokens):
                    tokens = tokens[:max_length - len(suffix_tokens)]
                tokens = tokens + suffix_tokens
                tokenized_messages.append(tokens)
            
            # Pad sequences
            max_len = max(len(t) for t in tokenized_messages) if tokenized_messages else 0
            padded = []
            attention_mask = []
            for tokens in tokenized_messages:
                padding_length = max_len - len(tokens)
                padded.append(tokens + [tokenizer.pad_token_id] * padding_length)
                attention_mask.append([1] * len(tokens) + [0] * padding_length)
            
            # Convert to tensors
            input_ids = torch.tensor(padded, dtype=torch.long).to(device)
            attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long).to(device)
            
            # Get logits
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask_tensor)
                logits = outputs.logits[:, -1, :]
                
                # Get true/false logits
                true_logits = logits[:, true_token]
                false_logits = logits[:, false_token]
                
                # Stack and softmax
                stacked_logits = torch.stack([false_logits, true_logits], dim=1)
                batch_scores = F.softmax(stacked_logits, dim=1)[:, 1].cpu().numpy().tolist()
                all_scores.extend(batch_scores)
        
        scores = all_scores
        
        # Create results
        results = []
        scored_docs = list(enumerate(scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        for rank, (idx, score) in enumerate(scored_docs[:top_k], 1):
            results.append(RerankResult(doc_id=str(idx), rank=rank, score=score))
        
        latency = time.time() - start_time
        return results, latency
        
    except Exception as e:
        latency = time.time() - start_time
        logging.error(f"Error in Qwen3-Reranker ({reranker.name}): {e}")
        import traceback
        logging.error(traceback.format_exc())
        return [], latency

def _rerank_single(reranker: RerankerConfig, query: str, documents: List[str], top_k: int) -> Tuple[List[RerankResult], float]:
    """Rerank using appropriate API based on reranker type"""
    if reranker.type == 'cohere':
        return _cohere_rerank(query, documents, reranker, top_k)
    elif reranker.type == 'zerank':
        return _zerank_rerank(query, documents, reranker, top_k)
    elif reranker.type == 'jina':
        return _jina_rerank(query, documents, reranker, top_k)
    elif reranker.type == 'voyage':
        return _voyage_rerank(query, documents, reranker, top_k)
    elif reranker.type == 'together':
        return _together_rerank(query, documents, reranker, top_k)
    elif reranker.type == 'qwen':
        return _qwen_rerank(query, documents, reranker, top_k)
    elif reranker.type == 'contextual':
        return _contextual_rerank(query, documents, reranker, top_k)
    elif reranker.type == 'replicate':
        return _replicate_rerank(query, documents, reranker, top_k)
    else:
        raise ValueError(f"Unknown reranker type: {reranker.type}")


def rerank_stage(config: Config, paths: RunPaths, logger: logging.Logger) -> Dict:
    """
    Rerank retrieval results using configured rerankers
    
    Args:
        config: Pipeline configuration
        paths: Run paths manager
        logger: Logger instance
    
    Returns:
        Dictionary with stage results metadata
    """
    logger.info("Starting reranking stage...")
    
    # Load retrieval results
    retrieval_file = paths.get_retrieval_file()
    logger.info(f"Loading retrieval results from {retrieval_file}")
    with open(retrieval_file, 'r') as f:
        retrieval_results = json.load(f)
    
    # Load corpus for document texts
    logger.info(f"Loading corpus from {config.dataset.corpus_path}")
    corpus = {}
    with open(config.dataset.corpus_path, 'r') as f:
        for line in f:
            doc = json.loads(line)
            doc_id = doc['_id']
            text = doc.get('title', '') + ' ' + doc.get('text', '')
            corpus[doc_id] = text.strip()
    
    logger.info(f"Processing {len(retrieval_results)} queries with {len(config.rerankers)} rerankers")
    print(f"   📊 Processing {len(retrieval_results)} queries")
    print(f"   🔄 Using {len(config.rerankers)} rerankers")
    
    # Process each reranker
    all_latencies = {}
    model_names = {}
    
    for reranker_idx, reranker in enumerate(config.rerankers, 1):
        # Qwen rerankers don't need API keys
        if reranker.type != 'qwen' and reranker.api_key is None:
            logger.warning(f"Skipping {reranker.name} - API key not set")
            print(f"   ⚠️  [{reranker_idx}/{len(config.rerankers)}] Skipping {reranker.name} - API key not set")
            continue
        
        logger.info(f"Reranking with {reranker.name}...")
        print(f"   🔄 [{reranker_idx}/{len(config.rerankers)}] Reranking with {reranker.name}...")
        
        reranked_file = paths.get_reranked_file(reranker.name)
        latency_file = paths.get_latency_file(reranker.name)
        
        # Check if already exists
        if reranked_file.exists() and config.pipeline.skip_if_exists:
            logger.info(f"Results already exist for {reranker.name}, skipping...")
            continue
        
        latencies = []
        reranked_queries = []
        
        for retrieval in retrieval_results:
            query_id = retrieval['query_id']
            query_text = retrieval['query_text']
            
            # Get document texts in retrieval order (top-50 from retrieval)
            doc_ids = [doc['doc_id'] for doc in retrieval['retrieved_docs']]
            num_retrieved = len(doc_ids)
            doc_texts = [corpus.get(doc_id, '') for doc_id in doc_ids]
            
            # Rerank: take top-50 from retrieval, rerank to get top-15
            if len(retrieval_results) <= 5:  # Only log for first few queries to avoid spam
                logger.debug(f"Reranking query {query_id}: {num_retrieved} docs -> top-{reranker.top_k}")
            
            rerank_results, latency = _rerank_single(
                reranker, query_text, doc_texts, reranker.top_k
            )
            
            latencies.append(latency * 1000)  # Convert to ms
            
            # Map back to original doc_ids
            reranked_docs = []
            for r in rerank_results:
                idx = int(r.doc_id)
                if idx < len(doc_ids):
                    reranked_docs.append({
                        'doc_id': doc_ids[idx],
                        'rank': r.rank,
                        'score': r.score
                    })
            
            reranked_queries.append({
                'query_id': query_id,
                'model': reranker.name,
                'results': reranked_docs
            })
        
        # Save reranked results
        logger.info(f"Saving reranked results to {reranked_file}")
        with open(reranked_file, 'w') as f:
            for query_result in reranked_queries:
                f.write(json.dumps(query_result) + '\n')
        
        # Save latency per query
        logger.info(f"Saving latencies to {latency_file}")
        with open(latency_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['query_id', 'latency_ms'])
            for query_result, latency_ms in zip(reranked_queries, latencies):
                writer.writerow([query_result['query_id'], f"{latency_ms:.2f}"])
        
        all_latencies[reranker.name] = latencies
        model_names[reranker.name] = reranker.name
        
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        logger.info(f"  Average latency: {avg_latency:.2f} ms")
        print(f"      ✓ Average latency: {avg_latency:.2f} ms")
    
    # Create latency summary and plot
    if all_latencies:
        summary_file = paths.get_latency_summary_file()
        logger.info(f"Creating latency summary at {summary_file}")
        with open(summary_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['reranker', 'avg_latency_ms', 'min_latency_ms', 'max_latency_ms', 'total_queries'])
            for name, latencies in all_latencies.items():
                writer.writerow([
                    name,
                    f"{sum(latencies) / len(latencies):.2f}",
                    f"{min(latencies):.2f}",
                    f"{max(latencies):.2f}",
                    len(latencies)
                ])
        
        # Create latency plot
        plot_file = paths.get_latency_plot_file()
        logger.info(f"Creating latency plot at {plot_file}")
        fig, ax = plt.subplots(figsize=(10, 6))
        reranker_names = list(all_latencies.keys())
        avg_latencies = [sum(latencies) / len(latencies) for latencies in all_latencies.values()]
        ax.bar(reranker_names, avg_latencies, color='#4C78A8')
        ax.set_ylabel('Average Latency (ms)', fontweight='bold')
        ax.set_title('Reranker Latency Comparison', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    result = {
        'status': 'completed',
        'num_rerankers': len(all_latencies),
        'rerankers': list(all_latencies.keys())
    }
    
    logger.info("Reranking stage completed successfully")
    return result

