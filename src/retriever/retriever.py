from typing import Dict, List

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

try:
    from FlagEmbedding import BGEReranker
except ImportError:
    from FlagEmbedding import FlagReranker as BGEReranker


class HybridRetriever:
    def __init__(
        self,
        embedder,
        qdrant_path: str = "./data/qdrant",
        collection_name: str = "dora_docs",
        rerank_model_name: str = "BAAI/bge-reranker-v2-m3",
    ):
        self.embedder = embedder
        self.client = QdrantClient(path=qdrant_path)
        self.collection_name = collection_name
        self.reranker = BGEReranker(rerank_model_name, use_fp16=True)
        self._scroll_and_build_bm25()

    def _scroll_and_build_bm25(self) -> None:
        texts = []
        self._metadata_map: Dict[str, dict] = {}
        next_offset = None
        while True:
            records, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=next_offset,
            )
            for r in records:
                text = r.payload.get("text", "")
                meta = {k: v for k, v in r.payload.items() if k != "text"}
                texts.append(text)
                self._metadata_map[text] = meta
            if next_offset is None or not records:
                break

        self._texts = texts
        tokenized_corpus = [t.split() for t in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        query_embedding = self.embedder.embed_query(query)

        vector_hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k * 2,
        )

        candidates = []
        for h in vector_hits:
            text = h.payload.get("text", "")
            meta = {k: v for k, v in h.payload.items() if k != "text"}
            candidates.append({
                "id": h.id,
                "text": text,
                "metadata": meta,
                "vector_score": h.score,
            })

        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        text_to_bm25 = dict(zip(self._texts, bm25_scores))

        for c in candidates:
            c["bm25_score"] = text_to_bm25.get(c["text"], 0.0)

        vec_scores = [c["vector_score"] for c in candidates]
        bm_scores = [c["bm25_score"] for c in candidates]
        vec_min, vec_max = min(vec_scores), max(vec_scores)
        bm_min, bm_max = min(bm_scores), max(bm_scores)

        for c in candidates:
            nv = 0.0 if vec_max == vec_min else (c["vector_score"] - vec_min) / (vec_max - vec_min)
            nb = 0.0 if bm_max == bm_min else (c["bm25_score"] - bm_min) / (bm_max - bm_min)
            c["combined_score"] = 0.5 * nv + 0.5 * nb

        candidates.sort(key=lambda x: x["combined_score"], reverse=True)

        rerank_candidates = candidates[:top_k]
        if rerank_candidates:
            pairs = [[query, c["text"]] for c in rerank_candidates]
            scores = self.reranker.compute_score(pairs)
            for c, s in zip(rerank_candidates, scores):
                c["rerank_score"] = s
            rerank_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        return rerank_candidates
