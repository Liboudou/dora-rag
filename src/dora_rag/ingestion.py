# Ingestion pipeline for DORA regulation PDF.

import yaml
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import DoclingDocument
from docling.datamodel.base_models import DocItemLabel  # Corrected import
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

from .models import DocumentChunk, DocumentChunkMetadata, Config
from .utils import get_qdrant_client


def load_config(config_path: str = 'config.yaml') -> Config:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    return Config(**config_data)


def load_pdf(pdf_path: str, config: Config) -> DoclingDocument:
    """Load PDF using Docling converter."""
    logger.info(f'Loading PDF from {pdf_path}')
    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
    )
    result = converter.convert(pdf_path)
    return result.document


def _extract_metadata(text: str) -> dict:
    """Extract article and chapter numbers from text."""
    article_match = _pattern_article.search(text)
    chapter_match = _pattern_chapter.search(text)
    meta = {}
    if article_match:
        meta['num_article'] = int(article_match.group(1))
    if chapter_match:
        meta['chapitre'] = int(chapter_match.group(1))  # keep French key as in metadata model
    return meta


_pattern_article = re.compile(r'Article\s+(\d+)', re.IGNORECASE)
_pattern_chapter = re.compile(r'Chapter\s+(\d+)', re.IGNORECASE)


def chunk_document(doc: DoclingDocument, config: Config) -> List[dict]:
    """Chunk document into pieces with metadata."""
    chunks = []
    current_chunk = []
    current_length = 0
    current_meta = {'num_article': None, 'chapitre': None, 'page': None}

    # Iterate over document items
    for item in doc.iterate_items():
        # Update metadata based on item labels and text
        if item.label in (DocItemLabel.SECTION_HEADER,
                          DocItemLabel.HEADING_LEVEL_1,
                          DocItemLabel.HEADING_LEVEL_2,
                          DocItemLabel.HEADING_LEVEL_3):
            # When we hit a heading, finalize current chunk if any
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'metadata': current_meta.copy()
                })
                current_chunk = []
                current_length = 0
            # Update metadata from heading text
            heading_text = item.text or ''
            mu = _extract_metadata(heading_text)
            current_meta.update(mu)
            # Also try to get page from provenance
            if hasattr(item, 'prov') and item.prov:
                # prov is a list of provenance items; take first page number
                if hasattr(item.prov[0], 'page_no'):
                    current_meta['page'] = item.prov[0].page_no
        # Add item text to current chunk
        if item.text:
            current_chunk.append(item.text.strip())
            current_length += len(item.text)
            # Update page number if available
            if hasattr(item, 'prov') and item.prov:
                if hasattr(item.prov[0], 'page_no'):
                    current_meta['page'] = item.prov[0].page_no
            # If chunk exceeds max size, finalize it
            if current_length >= config.data["chunk_size"]:
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'metadata': current_meta.copy()
                })
                # Overlap: keep last overlap characters
                overlap_text = ' '.join(current_chunk)[-config.data["chunk_overlap"]:]
                current_chunk = [overlap_text] if overlap_text else []
                current_length = len(overlap_text)
    # Add remaining chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        chunks.append({
            'text': chunk_text,
            'metadata': current_meta.copy()
        })
    return chunks


def embed_chunks(chunks: List[dict], config: Config) -> List[List[float]]:
    """Embed text chunks using BGE-M3 model."""
    logger.info(f'Embedding {len(chunks)} chunks')
    model = BGEM3FlagModel(
        config.embedding["model_name"],
        device=config.embedding["device"],
        batch_size=config.embedding["batch_size"],
    )
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(
        texts,
        batch_size=config.embedding["batch_size"],
        max_length=8192,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )["dense_vecs"]
    return embeddings.tolist()
def index_chunks(chunks: List[dict], embeddings: List[List[float]], config: Config) -> None:
    """Index chunks and embeddings into Qdrant."""
    logger.info(f'Indexing {len(chunks)} chunks into Qdrant')
    client = get_qdrant_client(config)
    collection_name = config.qdrant["collection_name"]
    vector_size = config.qdrant["vector_size"]

    # Check if collection exists
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]
    if collection_name not in collection_names:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        logger.info(f'Created collection {collection_name}')

    # Prepare points
    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        point = PointStruct(
            id=i,
            vector=embedding,
            payload={
                "text": chunk["text"],
                **chunk["metadata"],
            },
        )
        points.append(point)

    # Upload in batches
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(collection_name=collection_name, points=batch)
        logger.info(f'Uploaded batch {i // batch_size + 1}/{(len(points) - 1) // batch_size + 1}')


def run_ingestion(pdf_path: str = None) -> None:
    """Run the full ingestion pipeline."""
    config = load_config()
    if pdf_path is None:
        pdf_path = config.data["pdf_path"]
    else:
        # Override with provided path
        pass

    logger.info('Starting DORA RAG ingestion pipeline')

    # Step 1: Load PDF
    doc = load_pdf(pdf_path, config)

    # Step 2: Chunk document
    chunks = chunk_document(doc, config)
    logger.info(f'Created {len(chunks)} chunks')

    # Step 3: Embed chunks
    embeddings = embed_chunks(chunks, config)
    logger.info(f'Generated {len(embeddings)} embeddings')

    # Step 4: Index chunks
    index_chunks(chunks, embeddings, config)

    logger.info('Ingestion pipeline completed successfully')