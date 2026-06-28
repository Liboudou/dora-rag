import argparse
import uuid
from pathlib import Path

import fitz
import yaml

from src.chunker.chunker import chunk_text
from src.embedder.embedder import Embedder
from src.indexer.indexer import QdrantIndexer
from src.models import DocumentChunk


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def convert_pdf_to_text(pdf_path: str) -> tuple[str, list[dict]]:
    doc = fitz.open(pdf_path)
    pages = []
    full_text_parts = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = page.get_text()
        pages.append({"text": page_text, "page_num": page_num + 1})
        full_text_parts.append(page_text)
    doc.close()
    return "\n".join(full_text_parts), pages


def process_pdf(pdf_path: str, config_path: str = "config.yaml") -> None:
    config = load_config(config_path)
    full_text, pages = convert_pdf_to_text(pdf_path)

    chunks_data = chunk_text(full_text, base_metadata={})
    texts = [text for text, meta in chunks_data]

    embedder = Embedder.from_config(config)
    embeddings = embedder.embed_documents(texts)

    chunks = [
        DocumentChunk(
            id=str(uuid.uuid4()),
            text=text,
            metadata=meta,
            embedding=emb,
        )
        for (text, meta), emb in zip(chunks_data, embeddings)
    ]

    vector_size = len(chunks[0].embedding) if chunks else 1024
    indexer = QdrantIndexer.from_config_dict(config)
    indexer.vector_size = vector_size
    indexer._ensure_collection()

    points = [
        (chunk.id, chunk.embedding, {"text": chunk.text, **chunk.metadata})
        for chunk in chunks
    ]
    indexer.upsert_points(points)
    print(f"Ingested {len(chunks)} chunks from {pdf_path} into {config['collection_name']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="Path to the PDF file to ingest")
    args = parser.parse_args()
    process_pdf(args.pdf)
