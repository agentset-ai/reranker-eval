#!/usr/bin/env python3
"""
LLM Judge Script for Comparing Rerankers

Uses GPT-5 via Azure to judge which reranker returns more relevant results.
Uses ELO rating system with base 1500 and points 24.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import time
import os

import requests
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def call_azure_gpt(prompt: str, api_key: str, resource_name: str, deployment_id: str) -> str:
    """
    Call Azure GPT-5 API.
    
    Args:
        prompt: The prompt to send
        api_key: Azure API key
        resource_name: Azure resource name
        deployment_id: Deployment ID for GPT-5
        
    Returns:
        Response text from the API
    """
    client = AzureOpenAI(
        api_key=api_key,
        api_version="2024-02-15-preview",
        azure_endpoint=f"https://{resource_name}.openai.azure.com/"
    )
    
    try:
        response = client.chat.completions.create(
            model=deployment_id,
            messages=[
                {"role": "system", "content": "You are an expert judge evaluating search results relevance."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=100
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"API call failed: {e}")
        raise


def judge_query(cohere_results: List[Dict], zerank_results: List[Dict], 
                query_text: str, api_key: str, resource_name: str, deployment_id: str) -> Tuple[str, float]:
    """
    Judge which reranker returns more relevant results for a single query.
    
    Args:
        cohere_results: Top-5 Cohere results
        zerank_results: Top-5 ZeRank results
        query_text: The query text
        api_key: Azure API key
        resource_name: Azure resource name
        deployment_id: Deployment ID
        
    Returns:
        Tuple of (winner: 'A', 'B', or 'TIE', elo_change: float)
    """
    # Format the prompt
    prompt = f"""Given a user query and two lists of results from different rerankers, determine which reranker returns more relevant results. Return "A", "B" or "TIE".

The ordered list represents the relevance of the snippet to be, the higher the more relevant

Query: {query_text}

A list:
"""
    
    for i, doc in enumerate(cohere_results, 1):
        prompt += f"{i}. {doc['text']}\n"
    
    prompt += "\nB list:\n"
    
    for i, doc in enumerate(zerank_results, 1):
        prompt += f"{i}. {doc['text']}\n"
    
    prompt += '\nReturn only "A", "B" or "TIE":'
    
    # Call API
    try:
        response = call_azure_gpt(prompt, api_key, resource_name, deployment_id)
        winner = response.upper()
        
        if winner not in ['A', 'B', 'TIE']:
            logging.warning(f"Unexpected response: {response}, defaulting to TIE")
            winner = 'TIE'
        
        # Calculate ELO change
        if winner == 'A':
            elo_change = 24
        elif winner == 'B':
            elo_change = -24
        else:  # TIE
            elo_change = 0
        
        return winner, elo_change
        
    except Exception as e:
        logging.error(f"Failed to judge query: {e}")
        # Default to TIE on error
        return 'TIE', 0


def main():
    parser = argparse.ArgumentParser(description="Judge rerankers using LLM")
    parser.add_argument("--cohere_file", type=str, required=True, help="Cohere results JSONL file")
    parser.add_argument("--zerank_file", type=str, required=True, help="ZeRank results JSONL file")
    parser.add_argument("--api_key", type=str, default=None, help="Azure API key (or from AZURE_API_KEY env var)")
    parser.add_argument("--resource_name", type=str, default=None, help="Azure resource name (or from AZURE_RESOURCE_NAME env var)")
    parser.add_argument("--deployment_id", type=str, default=None, help="Deployment ID (or from AZURE_DEPLOYMENT_ID env var)")
    parser.add_argument("--top_k", type=int, default=5, help="Number of docs to compare per query")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSON file")
    
    args = parser.parse_args()
    
    # Load Azure credentials from environment if not provided
    api_key = args.api_key or os.getenv('AZURE_API_KEY')
    resource_name = args.resource_name or os.getenv('AZURE_RESOURCE_NAME')
    deployment_id = args.deployment_id or os.getenv('AZURE_DEPLOYMENT_ID')
    
    if not api_key:
        raise ValueError("Azure API key not provided. Set AZURE_API_KEY env var or use --api_key")
    if not resource_name:
        raise ValueError("Azure resource name not provided. Set AZURE_RESOURCE_NAME env var or use --resource_name")
    if not deployment_id:
        raise ValueError("Deployment ID not provided. Set AZURE_DEPLOYMENT_ID env var or use --deployment_id")
    
    setup_logging()
    
    # Load results
    logging.info(f"Loading results from {args.cohere_file} and {args.zerank_file}")
    
    cohere_data = []
    with open(args.cohere_file, 'r') as f:
        for line in f:
            cohere_data.append(json.loads(line))
    
    zerank_data = []
    with open(args.zerank_file, 'r') as f:
        for line in f:
            zerank_data.append(json.loads(line))
    
    if len(cohere_data) != len(zerank_data):
        raise ValueError(f"Mismatch: Cohere has {len(cohere_data)} queries, ZeRank has {len(zerank_data)}")
    
    logging.info(f"Loaded {len(cohere_data)} queries")
    
    # Initialize ELO ratings
    cohere_elo = 1500.0
    zerank_elo = 1500.0
    
    # Track results
    results = []
    
    # Judge each query
    for i, (cohere_query, zerank_query) in enumerate(zip(cohere_data, zerank_data)):
        query_id = cohere_query['query_id']
        query_text = cohere_query['query_text']
        
        if query_id != zerank_query['query_id']:
            raise ValueError(f"Query ID mismatch: {cohere_query['query_id']} vs {zerank_query['query_id']}")
        
        # Get top-k documents
        cohere_topk = cohere_query['documents'][:args.top_k]
        zerank_topk = zerank_query['documents'][:args.top_k]
        
        logging.info(f"Judging query {i+1}/{len(cohere_data)}: {query_id}")
        
        # Judge
        winner, elo_change = judge_query(
            cohere_topk, zerank_topk, query_text,
            api_key, resource_name, deployment_id
        )
        
        # Update ELO
        cohere_elo += elo_change
        zerank_elo -= elo_change
        
        # Record result with document chunks (FULL TEXT, not truncated)
        result = {
            'query_id': query_id,
            'query_text': query_text,
            'winner': winner,
            'elo_change': elo_change,
            'cohere_elo': cohere_elo,
            'zerank_elo': zerank_elo,
            'list_a_top5': [
                {
                    'rank': i+1,
                    'doc_id': doc['doc_id'],
                    'text': doc['text']  # Full text, not truncated
                }
                for i, doc in enumerate(cohere_topk)
            ],
            'list_b_top5': [
                {
                    'rank': i+1,
                    'doc_id': doc['doc_id'],
                    'text': doc['text']  # Full text, not truncated
                }
                for i, doc in enumerate(zerank_topk)
            ]
        }
        results.append(result)
        
        logging.info(f"  Winner: {winner}, Cohere ELO: {cohere_elo:.1f}, ZeRank ELO: {zerank_elo:.1f}")
        
        # Small delay to avoid rate limits
        time.sleep(0.5)
    
    # Final summary
    cohere_wins = sum(1 for r in results if r['winner'] == 'A')
    zerank_wins = sum(1 for r in results if r['winner'] == 'B')
    ties = sum(1 for r in results if r['winner'] == 'TIE')
    
    summary = {
        'total_queries': len(results),
        'cohere_wins': cohere_wins,
        'zerank_wins': zerank_wins,
        'ties': ties,
        'final_cohere_elo': cohere_elo,
        'final_zerank_elo': zerank_elo,
        'elo_difference': cohere_elo - zerank_elo,
        'final_winner': 'Cohere' if cohere_elo > zerank_elo else ('ZeRank' if zerank_elo > cohere_elo else 'TIE'),
        'results': results
    }
    
    # Save results
    logging.info(f"Saving results to {args.output_file}")
    with open(args.output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Total queries judged: {len(results)}")
    print(f"Cohere wins: {cohere_wins}")
    print(f"ZeRank wins: {zerank_wins}")
    print(f"Ties: {ties}")
    print(f"\nFinal ELO ratings:")
    print(f"  Cohere: {cohere_elo:.1f}")
    print(f"  ZeRank: {zerank_elo:.1f}")
    print(f"  Difference: {cohere_elo - zerank_elo:.1f}")
    print(f"\nWinner: {summary['final_winner']}")
    print("="*80)


if __name__ == "__main__":
    main()

