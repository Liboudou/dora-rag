from typing import List, Optional, Tuple

import yaml
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models


class QdrantIndexer:
    def __init__(
        self,
        path: str = "./data/qdrant",
        collection_name: str = "dora_docs",
        vector_size: int = 1024,
    ):
        self.client = QdrantClient(path=path)
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._ensure_collection()

    @classmethod
    def from_config(cls, config_path: str = "config.yaml") -> "QdrantIndexer":
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return cls(
            path=config.get("index_path", "./data/qdrant"),
            collection_name=config.get("collection_name", "dora_docs"),
            vector_size=config.get("vector_size", 1024),
        )

    @classmethod
    def from_config_dict(cls, config: dict) -> "QdrantIndexer":
        return cls(
            path=config.get("index_path", "./data/qdrant"),
            collection_name=config.get("collection_name", "dora_docs"),
            vector_size=config.get("vector_size", 1024),
        )

    def _ensure_collection(self) -> None:
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=self.vector_size,
                    distance=qdrant_models.Distance.COSINE,
                ),
            )

    def recreate_collection(self, vector_size: Optional[int] = None) -> None:
        self.client.delete_collection(
            collection_name=self.collection_name, timeout=60
        )
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qdrant_models.VectorParams(
                size=vector_size or self.vector_size,
                distance=qdrant_models.Distance.COSINE,
            ),
        )

    def load_collection(self) -> Optional[dict]:
        try:
            info = self.client.get_collection(
                collection_name=self.collection_name
            )
            return info.dict()
        except Exception:
            return None

    def upsert_points(
        self, points: List[Tuple[str, List[float], dict]]
    ) -> None:
        point_structs = [
            qdrant_models.PointStruct(id=pid, vector=vector, payload=payload)
            for pid, vector, payload in points
        ]
        self.client.upsert(
            collection_name=self.collection_name, points=point_structs
        )
