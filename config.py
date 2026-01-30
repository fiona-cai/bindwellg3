GENERATE_FAST=True
SAMPLE_RERANK = 30 if GENERATE_FAST else 50
SLOW_RERANKER_MODEL_NAME = "BAAI/bge-reranker-base" # "BAAI/bge-reranker-large"
MODIFY_QUERY=False # Uses LLM to modify the user's query to add context (including past conversations)
