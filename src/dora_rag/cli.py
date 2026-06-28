"""Command-line interface for DORA RAG."""

import argparse
import sys
from pathlib import Path
from loguru import logger

from . import run_ingestion, Config, Retriever, Generator

def ingest_command(args):
    """Run the ingestion pipeline."""
    logger.info("Starting ingestion from command line")
    pdf_path = args.pdf or None
    run_ingestion(pdf_path=pdf_path)
    logger.info("Ingestion completed.")

def query_command(args):
    """Answer a question using the RAG pipeline."""
    config_path = args.config or "config.yaml"
    config = Config.from_yaml(config_path)

    # Initialize retriever and generator
    retriever = Retriever(config)
    generator = Generator(config)

    # Retrieve relevant documents
    logger.info(f"Retrieving documents for question: {args.question}")
    retrieved = retriever.retrieve(args.question, top_k=config.retrieval["top_k"])
    logger.info(f"Retrieved {len(retrieved)} documents.")

    # Generate answer
    result = generator.generate(args.question, retrieved)
    answer = result["answer"]
    citations = result["citations"]

    # Output
    print("\nAnswer:")
    print(answer)
    print("\nCitations:")
    for i, cite in enumerate(citations, start=1):
        meta = cite.get("metadata", {})
        article = meta.get("num_article", "?")
        chapter = meta.get("chapitre", "?")
        page = meta.get("page", "?")
        print(f"[{i}] Article {article}, Chapter {chapter}, Page {page}")
        # Optionally print a snippet
        snippet = cite["text"][:200].replace("\n", " ")
        print(f'    "{snippet}..."')
    print()

def main() -> None:
    parser = argparse.ArgumentParser(description="DORA RAG CLI")
    parser.add_argument("--config", type=str, help="Path to config.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ingest subcommand
    ingest_parser = subparsers.add_parser("ingest", help="Ingest PDF into the vector store")
    ingest_parser.add_argument("--pdf", type=str, help="Path to PDF file (overrides config)")
    ingest_parser.set_defaults(func=ingest_command)

    # query subcommand
    query_parser = subparsers.add_parser("query", help="Ask a question")
    query_parser.add_argument("question", type=str, help="The question to ask")
    query_parser.set_defaults(func=query_command)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    # Configure logger to stderr
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    main()