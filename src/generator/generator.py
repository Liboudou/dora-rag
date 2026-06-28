import os
from typing import List, Dict

import httpx
from dotenv import load_dotenv

load_dotenv()


class OpenRouterGenerator:
    def __init__(
        self,
        model: str = "deepseek/deepseek-v4-flash",
        api_base: str = "https://openrouter.ai/api/v1",
    ):
        self.model = model
        self.api_base = api_base
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set in environment or .env file")

    @classmethod
    def from_config(cls, config: dict) -> "OpenRouterGenerator":
        return cls(
            model=config.get("llm_model", "deepseek/deepseek-v4-flash"),
            api_base=config.get("llm_api_base", "https://openrouter.ai/api/v1"),
        )

    def generate(self, query: str, context: List[Dict]) -> str:
        context_blocks = []
        for i, doc in enumerate(context):
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            page = metadata.get("page", "")
            source = f" (source: page {page})" if page else ""
            context_blocks.append(f"[{i + 1}] {text}{source}")

        context_text = "\n\n".join(context_blocks)

        system_prompt = (
            "You are a helpful assistant. Answer the user's question based strictly on the provided context. "
            "Cite your sources using the citation numbers in brackets like [1], [2], etc."
        )

        user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer with citations:"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        with httpx.Client(timeout=60) as client:
            response = client.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
