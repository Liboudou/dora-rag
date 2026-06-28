"""Hybrid retrieval combining BM25, vector search, and reranking."""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from loguru import logger
from qdrant_client import QdrantClient

from .models import Config
from .search import BM25Indexer
from .vector_retrieval import VectorRetriever
from .utils import get_qdrant_client

try:
    from FlagEmbedding import BGEM3FlagModel
    EMBEDDING_MODEL_AVAILABLE = True
except ImportError:
    BGEM3FlagModel = None  # type: ignore
    EMBEDDING_MODEL_AVAILABLE = False
    logger.warning("BGEM3FlagModel not available; embedding will fail.")

try:
    from .reranker import BGEReranker
    RERANKER_AVAILABLE = True
except ImportError:
    BGEReranker = None  # type: ignore
    RERANKER_AVAILABLE = False
    logger.warning("BGEReranker not available; reranking will be disabled.")


class Retriever:
    """Hybrid retriever combining BM25 and vector retrieval with optional reranking."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.bm25_indexer = BM25Indexer()
        self.vector_retriever = VectorRetriever(config)
        self.use_hybrid = config.retrieval["use_hybrid_search"]
        self.bm25_weight = config.retrieval["bm25_weight"]
        self.dense_weight = config.retrieval["dense_weight"]
        self.top_k = config.retrieval["top_k"]
        # Initialize embedding model for query encoding
        self.embedding_model: Optional[BGEM3FlagModel] = None
        if EMBEDDING_MODEL_AVAILABLE:
            try:
                model_name = "BAAI/bge-m3"  # same as used in ingestion.py
                self.embedding_model = BGEM3FlagModel(
                    model_name=model_name,
                    device=config.embedding["device"],
                    batch_size=config.embedding["batch_size"],
                )
                logger.info(f"Embedding model loaded: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize embedding model: {e}")
                self.embedding_model = None
        else:
            self.embedding_model = None
        # Initialize reranker if enabled and available
        self.reranker: Optional[BGEReranker] = None
        if config.retrieval.get("use_reranker", False) and RERANKER_AVAILABLE:
            try:
                model_name = config.retrieval.get("model_name", "BAAI/bge-reranker-v2-m3")
                self.reranker = BGEReranker(
                    model_name=model_name,
                    device=config.embedding["device"],
                )
                logger.info(f"Reranker enabled with model '{model_name}'.")
            except Exception as e:
                logger.warning(f"Failed to initialize reranker: {e}")
                self.reranker = None
        else:
            self.reranker = None
        logger.info(
            f"Retriever initialized: hybrid={self.use_hybrid}, "
            f"bm25_weight={self.bm25_weight}, dense_weight={self.dense_weight}, "
            f"top_k={self.top_k}, reranker={'enabled' if self.reranker else 'disabled'}"
        )

    def _load_bm25_index(self) -> None:
        """Load BM25 index from persisted chunks if not already built.
        For simplicity, we rebuild the index from the vector store payloads.
        In a production setting, you would persist the BM25 index separately.
        """
        if self.bm25_indexer._index is not None:
            return
        # Build client using same logic as VectorRetriever (now via utils)
        client = get_qdrant_client(self.config)
        # Fetch all points from Qdrant to build BM25 corpus
        scroll_offset = None
        all_texts: List[str] = []
        while True:
            scroll_result = client.scroll(
                collection_name=self.config.qdrant["collection_name"],
                limit=1000,
                offset=scroll_offset,
                with_payload=True,
                with_vectors=False,
            )
            points, scroll_offset = scroll_result
            if not points:
                break
            for point in points:
                text = point.payload.get("text", "")
                if text:
                    all_texts.append(text)
            if scroll_offset is None:
                break
        if not all_texts:
            logger.warning("No documents found in Qdrant to build BM25 index.")
            return
        self.bm25_indexer.fit(all_texts)
        logger.info(f"Built BM25 index from {len(all_texts)} documents.")

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve top-k documents for the query using hybrid search and optional reranking."""
        k = top_k if top_k is not None else self.top_k
        self._load_bm25_index()

        # Get BM25 results
        bm25_results = self.bm25_indexer.query(query, top_k=k * 2)  # get more for fusion
        # Get vector results: embed query then search
        vector_results: List[Dict[str, Any]] = []
        if self.embedding_model is not None:
            try:
                # Encode the query to get a dense vector
                query_embedding = self.embedding_model.encode([query])[0].tolist()
                vector_results = self.vector_retriever.query(query_embedding, top_k=k * 2)
            except Exception as e:
                logger.warning(f"Failed to embed query: {e}")
                vector_results = []
        else:
            logger.warning("Embedding model not available; skipping vector search.")

        # Normalize scores to 0-1 range for each method
        def normalize_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            if not results:
                return results
            scores = [r["score"] for r in results]
            min_score = min(scores)
            max_score = max(scores)
            if max_score == min_score:
                # All same score
                for r in results:
                    r["norm_score"] = 1.0
            else:
                for r in results:
                    r["norm_score"] = (r["score"] - min_score) / (max_score - min_score)
            return results

        bm25_norm = normalize_results(bm25_results)
        vector_norm = normalize_results(vector_results)

        # Combine scores
        combined_scores: dict[int, float] = {}
        for item in bm25_norm:
            idx = item["id"]
            combined_scores[idx] = combined_scores.get(idx, 0.0) + self.bm25_weight * item["norm_score"]
        for item in vector_norm:
            idx = item["id"]
            combined_scores[idx] = combined_scores.get(idx, 0.0) + self.dense_weight * item["norm_score"]

        # Create list of unique documents with combined score
        id_to_item: dict[int, dict] = {}
        for item in bm25_norm + vector_norm:
            idx = item["id"]
            if idx not in id_to_item:
                id_to_item[idx] = item

        fused_results = []
        for idx, score in combined_scores.items():
            item = id_to_item[idx].copy()
            item["score"] = score
            fused_results.append(item)

        # Sort by combined score descending
        fused_results.sort(key=lambda x: x["score"], reverse=True)
        # Take top k*2 for reranking
        fused_results = fused_results[: k * 2]

        # Apply reranker if available
        if self.reranker is not None:
            # Reranker expects list of passages (strings)
            passages = [doc["text"] for doc in fused_results]
            scores = self.reranker.rerank(query, passages)
            # scores is a list of floats
            for doc, score in zip(fused_results, scores):
                doc["rerank_score"] = float(score)
            # Sort by rerank score descending
            fused_results.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
            # Replace score with rerank score for final ranking
            for doc in fused_results:
                doc["score"] = doc.get("rerank_score", doc["score"])

        # Return top k
        return fused_results[:k]