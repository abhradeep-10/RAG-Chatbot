"""Central configuration.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    index_dir: Path = Path("data/indexes")

    # ---- ingestion -----------------------------------------------------------
    max_pdf_mb: float = 50.0
    max_pages: int = 1500
    chunk_target_tokens: int = 350    
    chunk_overlap_tokens: int = 60    
    min_chunk_tokens: int = 40         

    # ---- embeddings / reranker (open-source, local) ---------------------------
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    query_instruction: str = "Represent this sentence for searching relevant passages: "
    use_reranker: bool = True
    reranker_model: str = "BAAI/bge-reranker-base"
    device: str = "cpu"

    # ---- retrieval -----------------------------------------------------------
    dense_top_k: int = 20
    sparse_top_k: int = 20
    rrf_k: int = 60
    final_top_k: int = 5
    min_rerank_score: float = 0.05      
    min_dense_score: float = 0.45  
    max_context_chars: int = 9000       


    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen2.5:7b-instruct"
    llm_api_key: SecretStr = SecretStr("not-needed-for-ollama")
    llm_temperature: float = 0.0
    llm_max_tokens: int = 700
    llm_timeout_s: float = 120.0
    llm_max_retries: int = 2
    llm_json_mode: bool = True        


    max_question_chars: int = 1000
    max_history_turns: int = 6
    max_history_turn_chars: int = 1500
    enable_query_rewrite: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
