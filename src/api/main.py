import os

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.embedder.embedder import Embedder
from src.generator.generator import OpenRouterGenerator
from src.indexer.indexer import QdrantIndexer
from src.ingest import processor
from src.models import IngestRequest, QueryRequest, QueryResponse
from src.retriever.retriever import HybridRetriever

load_dotenv()

config_path = os.environ.get("CONFIG_PATH", "config.yaml")

app = FastAPI(title="DORA RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    with open(config_path) as f:
        config = yaml.safe_load(f)
    app.state.config = config
    app.state.embedder = Embedder.from_config(config)
    app.state.indexer = QdrantIndexer.from_config_dict(config)
    app.state.retriever = HybridRetriever(
        embedder=app.state.embedder,
        qdrant_path=config.get("index_path", "./data/qdrant"),
        collection_name=config.get("collection_name", "dora_docs"),
        rerank_model_name=config.get("rerank_model", "BAAI/bge-reranker-v2-m3"),
    )
    app.state.generator = OpenRouterGenerator.from_config(config)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(req: IngestRequest):
    try:
        processor.process_pdf(pdf_path=req.pdf_path, config_path=config_path)
        return {"message": f"Ingested {req.pdf_path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    try:
        results = app.state.retriever.retrieve(query=req.query, top_k=req.top_k)
        citations = [
            {
                "id": r.get("id"),
                "text": r.get("text"),
                "metadata": r.get("metadata"),
            }
            for r in results
        ]
        answer = app.state.generator.generate(query=req.query, context=results)
        return QueryResponse(answer=answer, citations=citations)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
