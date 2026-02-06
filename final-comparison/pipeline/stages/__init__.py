"""
Pipeline stages
"""

# Use lazy imports to avoid loading dependencies when not needed
# This allows importing individual stages (like rerank) without requiring
# all dependencies (like sentence_transformers for embed)
def __getattr__(name):
    if name == 'embed_stage':
        from .embed import embed_stage
        return embed_stage
    elif name == 'retrieve_stage':
        from .retrieve import retrieve_stage
        return retrieve_stage
    elif name == 'rerank_stage':
        from .rerank import rerank_stage
        return rerank_stage
    elif name == 'evaluate_stage':
        from .evaluate import evaluate_stage
        return evaluate_stage
    elif name == 'llm_judge_stage':
        from .llm_judge import llm_judge_stage
        return llm_judge_stage
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'embed_stage',
    'retrieve_stage',
    'rerank_stage',
    'evaluate_stage',
    'llm_judge_stage'
]

