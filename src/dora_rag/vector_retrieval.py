"""Vector retrieval using Qdrant."""

from typing import Dict, List, Any
from loguru import logger
from qdrant_client import QdrantClient

from .models import Config
from .utils import get_qdrant_client


class VectorRetriever:
    _client: QdrantClient
    _collection_name: str

    def __init__(self, config: Config) -> None:
        self._client = get_qdrant_client(config)
        self._collection_name = config.qdrant["collection_name"]
        logger.info(f"VectorRetriever initialized for collection '{self._collection_name}'")

    def query(self, vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        results = self._client.search(
            collection_name=self._collection_name,
            query_vector=vector,
            limit=top_k,
        )
        return [
            {
                "id": hit.id,
                "score": hit.score,
                "text": hit.payload.get("text", ""),
                "metadata": {
                    k: v for k, v in hit.payload.items() if k != "text"
                },
            }
            for hit in results
        ]