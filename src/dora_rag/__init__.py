"""Top-level package for DORA RAG."""

from .search import BM25Indexer, save_index, load_index
from .ingestion import run_ingestion
from .models import Config, DocumentChunk, DocumentChunkMetadata, QueryRequest
from .retrieval import Retriever, VectorRetriever, BGEReranker
from .generation import Generator
from .cli import main
from .api import app

__all__ = [
    "BM25Indexer",
    "save_index",
    "load_index",
    "run_ingestion",
    "Config",
    "DocumentChunk",
    "DocumentChunkMetadata",
    "QueryRequest",
    "Retriever",
    "VectorRetriever",
    "BGEReranker",
    "Generator",
    "main",
    "app",
]
