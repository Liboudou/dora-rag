import pickle
from pathlib import Path
from typing import List, Optional

from loguru import logger
from rank_bm25 import BM25Okapi


class BM25Indexer:
    _index: BM25Okapi | None
    _documents: List[str]
    _tokenized_corpus: List[List[str]]

    def __init__(self) -> None:
        self._index = None
        self._documents = []
        self._tokenized_corpus = []

    def fit(self, texts: List[str]) -> None:
        self._documents = texts
        self._tokenized_corpus = [text.split() for text in texts]
        self._index = BM25Okapi(self._tokenized_corpus)
        logger.info(f"BM25 index built with {len(texts)} documents")

    def query(self, query: str, top_k: int = 5) -> List[dict]:
        if self._index is None:
            raise RuntimeError("BM25 index is not built. Call fit() first.")
        tokenized_query = query.split()
        scores = self._index.get_scores(tokenized_query)
        top_indices = sorted(
            range(len(self._documents)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]
        return [
            {
                "id": i,
                "text": self._documents[i],
                "score": float(scores[i]),
            }
            for i in top_indices
            if scores[i] > 0
        ]

    def save(self, path: str | Path) -> None:
        data = {
            "documents": self._documents,
            "tokenized_corpus": self._tokenized_corpus,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"BM25 index saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "BM25Indexer":
        with open(path, "rb") as f:
            data = pickle.load(f)
        indexer = cls()
        indexer._documents = data["documents"]
        indexer._tokenized_corpus = data["tokenized_corpus"]
        indexer._index = BM25Okapi(indexer._tokenized_corpus)
        logger.info(f"BM25 index loaded from {path} ({len(indexer._documents)} documents)")
        return indexer


def save_index(indexer: BM25Indexer, path: str | Path) -> None:
    indexer.save(path)


def load_index(path: str | Path) -> BM25Indexer:
    return BM25Indexer.load(path)
