"""Generation module for DORA RAG: uses LLM to produce answers with citations."""

from __future__ import annotations

import os
from typing import List, Dict, Any, Optional
from loguru import logger

from .models import Config

try:
    import openai
except ImportError:  # pragma: no cover
    openai = None
    logger.warning("OpenAI package not installed; generation will fail.")


class Generator:
    """Generate answers using an LLM, grounded in retrieved documents."""

    def __init__(self, config: Config) -> None:
        self.config = config
        if openai is None:
            raise ImportError("OpenAI package is required for generation. Install with `pip install openai`.")
        # Configure OpenAI client to use OpenRouter
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set.")
        self.client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model = config.llm["model_name"]
        self.temperature = config.llm["temperature"]
        self.max_tokens = config.llm["max_tokens"]
        logger.info(f"Initialized Generator with model '{self.model}'")

    def generate(self, question: str, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate an answer to the question based on the provided contexts."""
        # Prepare context string with citations
        context_parts = []
        citations = []
        for i, ctx in enumerate(contexts, start=1):
            # Each context dict should have 'text' and maybe 'metadata' with article/chapter/page
            text = ctx.get("text", "")
            meta = ctx.get("metadata", {})
            # Build citation string
            citation_parts = []
            if meta.get("num_article") is not None:
                citation_parts.append(f"Article {meta['num_article']}")
            if meta.get("chapitre") is not None:
                citation_parts.append(f"Chapter {meta['chapitre']}")
            if meta.get("page") is not None:
                citation_parts.append(f"p. {meta['page']}")
            citation = ", ".join(citation_parts) if citation_parts else f"Source [{i}]"
            context_parts.append(f"[{i}] {text}")
            citations.append({
                "id": i,
                "text": text[:200] + ("..." if len(text) > 200 else ""),  # truncate for brevity
                "metadata": meta,
                "citation": citation,
            })
        context_str = "\n\n".join(context_parts)

        # Construct prompt
        system_prompt = (
            "You are an expert on the EU Digital Operational Resilience Act (DORA) Regulation (EU) 2022/2554. "
            "Answer the user's question based solely on the provided context excerpts. "
            "If the answer cannot be determined from the context, say you do not have enough information. "
            "When you use information from a source, cite it using the bracket numbers provided, e.g., [1]. "
            "Do not make up information."
        )
        user_prompt = f"Context:\n{context_str}\n\nQuestion: {question}\n\nAnswer:"

        # Call the LLM
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:  # pragma: no cover
            logger.error(f"LLM generation failed: {e}")
            answer = f"Error generating answer: {e}"

        return {
            "answer": answer,
            "citations": citations,
        }