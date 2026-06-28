"""FastAPI application for DORA RAG."""

from __future__ import annotations

import asyncio
import logging
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from loguru import logger

from .models import Config
from .ingestion import run_ingestion
from .retrieval import Retriever
from .generation import Generator


async def lifespan(app: FastAPI):
    # Startup
    global config, retriever, generator
    config = Config.from_yaml("config.yaml")
    retriever = Retriever(config)
    generator = Generator(config)
    logger.info("API startup completed.")
    yield
    # Shutdown (if needed)
    logger.info("API shutdown.")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="DORA RAG API",
    description="API for querying the DORA regulation using RAG.",
    lifespan=lifespan,
)

# Global instances (initialized via lifespan)
config: Config | None = None
retriever: Retriever | None = None
generator: Generator | None = None


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]


class IngestRequest(BaseModel):
    pdf_path: str | None = None


@app.get("/health")
async def health() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(req: IngestRequest) -> Dict[str, str]:
    """Ingest a PDF file into the vector store."""
    if config is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    pdf_path = req.pdf_path or config.data["pdf_path"]
    from pathlib import Path
    if not Path(pdf_path).exists():
        raise HTTPException(status_code=404, detail=f"PDF file not found: {pdf_path}")
    # Run ingestion in a background thread to avoid blocking the event loop
    logger.info(f"Ingesting PDF: {pdf_path}")
    await asyncio.to_thread(run_ingestion, pdf_path=pdf_path)
    # Reload retriever to pick up new data
    global retriever
    retriever = Retriever(config)
    return {"status": "ingested", "pdf_path": pdf_path}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    """Answer a question using the RAG pipeline."""
    if retriever is None or generator is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    # Retrieve documents
    retrieved = retriever.retrieve(req.question, top_k=req.top_k)
    # Generate answer
    result = generator.generate(req.question, retrieved)
    return QueryResponse(answer=result["answer"], citations=result["citations"])


# To run: uvicorn api:app --reload