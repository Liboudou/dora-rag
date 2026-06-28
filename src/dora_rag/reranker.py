from typing import List

from FlagEmbedding import FlagReranker
from loguru import logger


class BGEReranker:
    _model: FlagReranker

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "cpu", use_fp16: bool = False) -> None:
        logger.info(f"Loading reranker model: {model_name}")
        self._model = FlagReranker(model_name, device=device, use_fp16=use_fp16)
        logger.info(f"Reranker model loaded on {device}")

    def rerank(self, query: str, passages: List[str]) -> List[float]:
        pairs = [[query, passage] for passage in passages]
        scores = self._model.compute_score(pairs)
        return [float(s) for s in scores]
