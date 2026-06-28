from typing import List, Optional

from FlagEmbedding import BGEM3FlagModel


class Embedder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        batch_size: int = 32,
        use_fp16: bool = True,
    ):
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        self.batch_size = batch_size

    @classmethod
    def from_config(cls, config: dict) -> "Embedder":
        model_name = config.get("embedding_model", "BAAI/bge-m3")
        batch_size = config.get("batch_size") or config.get("embedding_batch_size", 32)
        return cls(model_name=model_name, batch_size=batch_size)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(
            texts, batch_size=self.batch_size
        )["dense_vecs"].tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], batch_size=self.batch_size)["dense_vecs"][0].tolist()
