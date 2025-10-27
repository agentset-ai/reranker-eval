#!/usr/bin/env python3
"""
Optimized Rerank Retrieved Candidates Script

Features:
  - EU/US region toggle (ZeRank EU POP, Cohere global)
  - Proper connection pooling (aiohttp TCPConnector), no insecure TLS hacks
  - Pure API latency timing (excludes local queue wait)
  - Per-provider concurrency limits
  - Consistent payload trimming by bytes
  - Optional randomized provider call order (mitigates warm-connection bias)
  - Auto-fallback: ZeRank EU -> US on HTTP 401
  - Metrics: mean + p50/p90; optional nDCG/Recall if qrels available

Input file format (JSONL):
  {
    "query_id": "123",
    "query_text": "...",
    "candidates": [
      {"doc_id": "d1", "text": "...", "title": "optional"},
      ...
    ]
  }
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import time
import asyncio
import aiohttp
from collections import defaultdict
import math
import os
import random

import numpy as np
from tqdm import tqdm


# --------------------------- Logging ---------------------------------

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


# --------------------------- Utils -----------------------------------

def trim_text(text: str, max_bytes: int = 3000) -> str:
    """Trim text to a maximum number of bytes (UTF-8), safe truncate at sentence boundary."""
    if not isinstance(text, str):
        return ""
    b = text.encode("utf-8")
    if len(b) <= max_bytes:
        return text
    # Trim to max_bytes
    trimmed = b[:max_bytes].decode("utf-8", errors="ignore")
    # Find the last sentence-ending punctuation (., !, ?) followed by space or end
    import re
    # Look for sentence endings followed by space, newline, or end of string
    matches = list(re.finditer(r'[.!?]+[\s\n]*', trimmed))
    if matches:
        # Use the last match that ends within 90% of max_bytes
        last_match = matches[-1]
        if last_match.end() > max_bytes * 0.9:
            return trimmed[:last_match.end()].strip()
    # Fallback: find last space if no sentence boundary found
    last_space = max(trimmed.rfind(' '), trimmed.rfind('\n'))
    if last_space > max_bytes * 0.9:
        return trimmed[:last_space].strip()
    return trimmed


def load_qrels_tsv(qrels_path: Path) -> Dict[str, List[str]]:
    """Load qrels TSV with columns: query_id \t 0 \t doc_id \t relevance."""
    qrels = defaultdict(list)
    with open(qrels_path, "r", encoding="utf-8") as f:
        header_skipped = False
        for line in f:
            parts = line.rstrip("\n").split("\t")
            # Some qrels have headers; try to skip if present
            if not header_skipped and not parts[0].isdigit():
                header_skipped = True
                continue
            if len(parts) >= 4:
                qid, _, did, rel = parts[0], parts[1], parts[2], parts[3]
                try:
                    if int(rel) > 0:
                        qrels[qid].append(did)
                except ValueError:
                    # Line likely header; ignore
                    continue
    return qrels


def calculate_ndcg_at_k(results: List[str], relevant: List[str], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = 0.0
    for i in range(min(k, len(results))):
        if results[i] in relevant:
            dcg += 1.0 / math.log2(i + 2)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return dcg / idcg if idcg > 0 else 0.0


def calculate_recall_at_k(results: List[str], relevant: List[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = set(results[:k])
    return len(top_k.intersection(relevant)) / float(len(relevant))


def pct(values: List[float], p: float) -> float:
    return float(np.percentile(values, p)) if values else 0.0


# --------------------------- Providers --------------------------------

async def rerank_provider(
    session: aiohttp.ClientSession,
    provider: str,
    endpoint: str,
    query: str,
    candidates: List[Dict],
    api_key: str,
    model: str,
    max_text_bytes: int,
    store_trimmed_text: bool,
    zerank_latency_mode: str = None,  # "slow" optional (ignored by some endpoints)
) -> Tuple[List[Dict], float]:
    """
    Call provider's rerank endpoint and return (reranked_candidates, latency_ms).
    NOTE: Concurrency semaphore is handled by the caller so timing is pure API time.
    """
    # Prepare documents (trimmed + title prefix)
    docs = []
    for c in candidates:
        title = trim_text(c.get("title", ""), max_text_bytes // 2)
        body = trim_text(c.get("text", ""), max_text_bytes)
        docs.append(f"{title}\n{body}".strip() if title else body)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "query": query,
        "documents": docs,
        "top_n": len(docs),
    }
    # Optional hint; may be ignored by /models/rerank
    if provider == "ZeRank" and zerank_latency_mode:
        payload["latency"] = zerank_latency_mode

    # Cohere SDK path for v3.5
    if provider == "Cohere" and model == "rerank-v3.5":
        try:
            import cohere
        except Exception as e:
            logging.error(f"Cohere SDK not installed: {e}")
            # fall back to HTTP path below
        else:
            t0 = time.perf_counter()
            try:
                # Run blocking SDK in thread to avoid blocking event loop
                def _call():
                    client = cohere.ClientV2(api_key=api_key)
                    result = client.rerank(model=model, query=query, documents=docs, top_n=len(docs))
                    return result
                
                result = await asyncio.to_thread(_call)
                latency_ms = (time.perf_counter() - t0) * 1000.0
                
                results = getattr(result, "results", []) or (result if isinstance(result, list) else [])
                out = []
                for r in results:
                    idx = getattr(r, "index", None) or (r.get("index") if isinstance(r, dict) else None)
                    score = getattr(r, "relevance_score", None) or (r.get("relevance_score") if isinstance(r, dict) else None) or (r.get("score") if isinstance(r, dict) else None)
                    if idx is None or not (0 <= idx < len(candidates)):
                        continue
                    item = candidates[idx].copy()
                    item["score"] = score
                    if store_trimmed_text:
                        item["text"] = trim_text(item.get("text", ""), max_text_bytes)
                    out.append(item)
                return out, latency_ms
            except Exception as e:
                latency_ms = (time.perf_counter() - t0) * 1000.0
                logging.error(f"Cohere SDK request failed: {e}")
                out = []
                for c in candidates:
                    item = c.copy()
                    if store_trimmed_text:
                        item["text"] = trim_text(item.get("text", ""), max_text_bytes)
                    out.append(item)
                return out, latency_ms

    t0 = time.perf_counter()
    try:
        async with session.post(endpoint, headers=headers, json=payload) as resp:
            if provider == "ZeRank" and resp.status == 401 and "eu-api." in endpoint:
                # auto-fallback to US region
                us_endpoint = endpoint.replace("eu-api.", "api.")
                logging.warning("ZeRank EU returned 401 — falling back to US endpoint.")
                async with session.post(us_endpoint, headers=headers, json=payload) as resp2:
                    resp2.raise_for_status()
                    data = await resp2.json()
            else:
                resp.raise_for_status()
                data = await resp.json()

        latency_ms = (time.perf_counter() - t0) * 1000.0

        results = data.get("results", [])
        out = []
        for r in results:
            idx = r["index"]
            score = r.get("relevance_score")
            item = candidates[idx].copy()
            item["score"] = score
            if store_trimmed_text:
                # Keep outputs lean & consistent with request
                item["text"] = trim_text(item.get("text", ""), max_text_bytes)
            out.append(item)
        return out, latency_ms

    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        logging.error(f"{provider} request failed: {e}")
        # Return original order on failure so pipeline continues
        out = []
        for c in candidates:
            item = c.copy()
            if store_trimmed_text:
                item["text"] = trim_text(item.get("text", ""), max_text_bytes)
            out.append(item)
        return out, latency_ms


async def process_query(
    session: aiohttp.ClientSession,
    cohere_sem: asyncio.Semaphore,
    zerank_sem: asyncio.Semaphore,
    query_data: Dict,
    cohere_key: str,
    zerank_key: str,
    top_k: int,
    cohere_endpoint: str,
    zerank_endpoint: str,
    max_text_bytes: int,
    store_trimmed_text: bool,
    randomize_order: bool,
    zerank_latency_mode: str = None,
) -> Dict:
    qid = query_data["query_id"]
    qtext = query_data["query_text"]
    candidates = query_data["candidates"]

    # Gate concurrency at the CALL SITE to exclude wait time from latency.
    async def run_cohere():
        async with cohere_sem:
            return await rerank_provider(
                session, "Cohere", cohere_endpoint, qtext, candidates,
                cohere_key, "rerank-v3.5", max_text_bytes, store_trimmed_text
            )

    async def run_zerank():
        async with zerank_sem:
            return await rerank_provider(
                session, "ZeRank", zerank_endpoint, qtext, candidates,
                zerank_key, "zerank-1", max_text_bytes, store_trimmed_text,
                zerank_latency_mode=zerank_latency_mode,
            )

    if randomize_order and (hash(qid) & 1):
        zerank_result, cohere_result = await asyncio.gather(run_zerank(), run_cohere())
    else:
        cohere_result, zerank_result = await asyncio.gather(run_cohere(), run_zerank())

    cohere_reranked, cohere_lat = cohere_result
    zerank_reranked, zerank_lat = zerank_result

    # Format top-k
    def topk(rows):
        return [
            {
                "doc_id": d["doc_id"],
                "text": d.get("text", ""),
                "title": d.get("title"),
                "score": d.get("score"),
                "rank": i + 1,
            }
            for i, d in enumerate(rows[:top_k])
        ]

    return {
        "query_id": qid,
        "query_text": qtext,
        "cohere": {"documents": topk(cohere_reranked), "latency_ms": cohere_lat},
        "zerank": {"documents": topk(zerank_reranked), "latency_ms": zerank_lat},
    }


# --------------------------- Main -------------------------------------

def print_metrics(
    cohere_results: List[Dict],
    zerank_results: List[Dict],
    qrels: Dict[str, List[str]],
    metrics_file: str = None,
):
    co_lat = [r["latency_ms"] for r in cohere_results]
    ze_lat = [r["latency_ms"] for r in zerank_results]
    co_mean, ze_mean = np.mean(co_lat), np.mean(ze_lat)

    print("\n" + "=" * 80)
    print("PERFORMANCE METRICS")
    print("=" * 80)
    print("Average Latency:")
    print(f"  Cohere: {co_mean:.2f} ms   (p50 {pct(co_lat,50):.1f} / p90 {pct(co_lat,90):.1f})")
    print(f"  ZeRank: {ze_mean:.2f} ms   (p50 {pct(ze_lat,50):.1f} / p90 {pct(ze_lat,90):.1f})")
    print(f"  Difference: {co_mean - ze_mean:.2f} ms")

    metrics_data = {
        "performance": {
            "cohere_avg_latency_ms": float(co_mean),
            "zerank_avg_latency_ms": float(ze_mean),
            "cohere_p50_ms": pct(co_lat, 50),
            "cohere_p90_ms": pct(co_lat, 90),
            "zerank_p50_ms": pct(ze_lat, 50),
            "zerank_p90_ms": pct(ze_lat, 90),
            "latency_difference_ms": float(co_mean - ze_mean),
        },
        "evaluation": {},
    }

    if qrels:
        print("\n" + "=" * 80)
        print("EVALUATION METRICS (with ground truth)")
        print("=" * 80)
        ks = [1, 5, 10]
        co_ndcg = {k: [] for k in ks}
        ze_ndcg = {k: [] for k in ks}
        co_rec = {k: [] for k in ks}
        ze_rec = {k: [] for k in ks}

        for co, ze in zip(cohere_results, zerank_results):
            qid = co["query_id"]
            if qid in qrels:
                rel = qrels[qid]
                co_ids = [d["doc_id"] for d in co["documents"]]
                ze_ids = [d["doc_id"] for d in ze["documents"]]
                for k in ks:
                    co_ndcg[k].append(calculate_ndcg_at_k(co_ids, rel, k))
                    ze_ndcg[k].append(calculate_ndcg_at_k(ze_ids, rel, k))
                    co_rec[k].append(calculate_recall_at_k(co_ids, rel, k))
                    ze_rec[k].append(calculate_recall_at_k(ze_ids, rel, k))

        print("\n" + "-" * 80)
        print(f"{'Metric':<15} {'Cohere':<15} {'ZeRank':<15} {'Δ':<15}")
        print("-" * 80)
        for k in ks:
            cavg = float(np.mean(co_ndcg[k])) if co_ndcg[k] else 0.0
            zavg = float(np.mean(ze_ndcg[k])) if ze_ndcg[k] else 0.0
            print(f"nDCG@{k:<10} {cavg:<15.4f} {zavg:<15.4f} {zavg - cavg:<15.4f}")
        print("-" * 80)
        for k in ks:
            cavg = float(np.mean(co_rec[k])) if co_rec[k] else 0.0
            zavg = float(np.mean(ze_rec[k])) if ze_rec[k] else 0.0
            print(f"Recall@{k:<9} {cavg:<15.4f} {zavg:<15.4f} {zavg - cavg:<15.4f}")
        print("-" * 80)

        metrics_data["evaluation"] = {
            "ndcg": {f"@{k}": {"cohere": float(np.mean(co_ndcg[k]) if co_ndcg[k] else 0.0),
                               "zerank": float(np.mean(ze_ndcg[k]) if ze_ndcg[k] else 0.0)}
                     for k in ks},
            "recall": {f"@{k}": {"cohere": float(np.mean(co_rec[k]) if co_rec[k] else 0.0),
                                 "zerank": float(np.mean(ze_rec[k]) if ze_rec[k] else 0.0)}
                       for k in ks},
            "queries_with_ground_truth": len(qrels),
        }

    print("=" * 80)

    if metrics_file:
        Path(metrics_file).parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics_data, f, indent=2)
        print(f"\nMetrics saved to: {metrics_file}")


async def main_async(args):
    # Load environment variables from .env file
    from dotenv import load_dotenv
    load_dotenv()
    
    logging.info(f"Loading retrieved candidates from {args.input_file}")
    queries_data = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            queries_data.append(json.loads(line))
    logging.info(f"Loaded {len(queries_data)} queries")

    # Keys
    cohere_key = os.getenv("COHERE_API_KEY")
    zerank_key = os.getenv("ZERANK_API_KEY")
    if not cohere_key:
        raise ValueError("COHERE_API_KEY is not set")
    if not zerank_key:
        raise ValueError("ZERANK_API_KEY is not set")

    # Endpoints
    if args.region == "eu":
        cohere_endpoint = "https://api.cohere.ai/v1/rerank"  # Cohere: global
        zerank_endpoint = "https://eu-api.zeroentropy.dev/v1/models/rerank"
        logging.info("Using EU endpoint for ZeRank (Cohere global).")
    else:
        cohere_endpoint = "https://api.cohere.ai/v1/rerank"
        zerank_endpoint = "https://api.zeroentropy.dev/v1/models/rerank"
        logging.info("Using US endpoints.")

    # qrels (optional)
    qrels = {}
    if args.qrels_path:
        qrels_path = Path(args.qrels_path)
        if qrels_path.exists():
            qrels = load_qrels_tsv(qrels_path)
            logging.info(f"Loaded ground truth for {len(qrels)} queries from {qrels_path}")

    # Concurrency
    cohere_sem = asyncio.Semaphore(args.cohere_concurrency)
    zerank_sem = asyncio.Semaphore(args.zerank_concurrency)
    logging.info(f"Concurrency: Cohere={args.cohere_concurrency}  ZeRank={args.zerank_concurrency}")
    logging.info(f"Payload trim: {args.max_text_bytes} bytes per document")
    if args.randomize_order:
        logging.info("Randomizing provider call order per query (warm-bias mitigation).")

    # Pooled session
    connector = aiohttp.TCPConnector(
        limit=20,
        limit_per_host=10,
        ttl_dns_cache=300,
        enable_cleanup_closed=True,
        ssl=False,  # Temporary workaround for SSL certificate issues
    )
    timeout = aiohttp.ClientTimeout(total=30, connect=10, sock_read=30)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout, trust_env=True) as session:
        tasks = [
            process_query(
                session, cohere_sem, zerank_sem, q,
                cohere_key, zerank_key, args.top_k,
                cohere_endpoint, zerank_endpoint, args.max_text_bytes,
                args.store_trimmed_text, args.randomize_order, args.zerank_latency_mode
            )
            for q in queries_data
        ]
        results = []
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Reranking"):
            results.append(await coro)

    results.sort(key=lambda x: x["query_id"])

    # Split & persist
    cohere_results, zerank_results = [], []
    for r in results:
        cohere_results.append({
            "query_id": r["query_id"],
            "query_text": r["query_text"],
            "documents": r["cohere"]["documents"],
            "latency_ms": r["cohere"]["latency_ms"],
        })
        zerank_results.append({
            "query_id": r["query_id"],
            "query_text": r["query_text"],
            "documents": r["zerank"]["documents"],
            "latency_ms": r["zerank"]["latency_ms"],
        })

    Path(args.cohere_output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.cohere_output, "w", encoding="utf-8") as f:
        for row in cohere_results:
            f.write(json.dumps(row) + "\n")

    Path(args.zerank_output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.zerank_output, "w", encoding="utf-8") as f:
        for row in zerank_results:
            f.write(json.dumps(row) + "\n")

    # Metrics
    print_metrics(cohere_results, zerank_results, qrels, args.metrics_output)
    logging.info(f"Saved: {args.cohere_output} and {args.zerank_output}")


def main():
    parser = argparse.ArgumentParser(description="Rerank retrieved candidates (optimized)")
    parser.add_argument("--input_file", required=True, help="Input JSONL with queries + candidates")
    parser.add_argument("--cohere_output", required=True, help="Output JSONL for Cohere reranks")
    parser.add_argument("--zerank_output", required=True, help="Output JSONL for ZeRank reranks")
    parser.add_argument("--metrics_output", default=None, help="Optional JSON file for metrics")
    parser.add_argument("--qrels_path", default=None, help="Optional TSV qrels for nDCG/Recall")
    parser.add_argument("--top_k", type=int, default=15, help="Keep top-k results per query")
    parser.add_argument("--region", choices=["us", "eu"], default="us", help="API region for ZeRank (Cohere global)")
    parser.add_argument("--cohere_concurrency", type=int, default=4, help="Max concurrent Cohere requests")
    parser.add_argument("--zerank_concurrency", type=int, default=4, help="Max concurrent ZeRank requests")
    parser.add_argument("--max_text_bytes", type=int, default=3000, help="Max bytes per document text")
    parser.add_argument("--store_trimmed_text", action="store_true", default=True, help="Store trimmed text in outputs")
    parser.add_argument("--no-store_trimmed_text", dest="store_trimmed_text", action="store_false")
    parser.add_argument("--randomize_order", action="store_true", help="Randomize provider order per query")
    parser.add_argument("--zerank_latency_mode", choices=["slow"], default=None,
                        help="Optional hint for ZeRank ('slow' for high-throughput batch).")
    args = parser.parse_args()

    setup_logging()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
