"""Embedding providers with swappable backend.

Supports:
- OpenAI API (default, best quality)
- Local models via sentence-transformers (cost-free, faster)

Usage:
    provider = get_provider()
    embeddings = await provider.embed(["text1", "text2"])
"""

import asyncio
from abc import ABC, abstractmethod

from .config import settings


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the embedding dimension size."""
        pass

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of strings to embed (max ~8000 tokens per string for OpenAI)

        Returns:
            List of embedding vectors, one per input text
        """
        pass

    async def embed_one(self, text: str) -> list[float]:
        """Convenience method for single text embedding."""
        results = await self.embed([text])
        return results[0]


class OpenAIEmbeddings(EmbeddingProvider):
    """OpenAI embeddings via API.

    Best quality, ~$0.02 per 1M tokens.
    Supports batching up to 2048 inputs per call.
    """

    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None):
        import openai

        self.model = model
        self.client = openai.AsyncOpenAI(api_key=api_key or settings.openai_api_key)

        # Model dimensions
        self._dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }.get(model, 1536)

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        # OpenAI supports batching up to 2048 inputs
        response = await self.client.embeddings.create(model=self.model, input=texts)

        # Response comes back in same order as input
        return [item.embedding for item in response.data]


class LocalEmbeddings(EmbeddingProvider):
    """Local embeddings via sentence-transformers.

    Zero cost, faster for high volume, but slightly lower quality.
    Runs on CPU by default, GPU if available.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Lazy import - only load if actually used
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self._dimensions = self.model.get_sentence_embedding_dimension()

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        # sentence-transformers is synchronous and CPU-bound
        # Run in thread pool to not block event loop
        embeddings = await asyncio.to_thread(
            self.model.encode, texts, normalize_embeddings=True, show_progress_bar=False
        )

        return embeddings.tolist()


class MockEmbeddings(EmbeddingProvider):
    """Mock embeddings for testing.

    Returns deterministic vectors based on text hash.
    """

    def __init__(self, dimensions: int = 1536):
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        import hashlib

        results = []
        for text in texts:
            # Create deterministic but varied embedding from text hash
            hash_bytes = hashlib.sha256(text.encode()).digest()
            # Expand hash to fill dimensions
            vector = []
            for i in range(self._dimensions):
                byte_idx = i % len(hash_bytes)
                # Normalize to [-1, 1] range
                value = (hash_bytes[byte_idx] / 255.0) * 2 - 1
                vector.append(value)
            results.append(vector)
        return results


# Provider factory
_provider: EmbeddingProvider | None = None


def get_provider() -> EmbeddingProvider:
    """Get the configured embedding provider (singleton)."""
    global _provider
    if _provider is None:
        _provider = create_provider()
    return _provider


def create_provider(provider_type: str | None = None) -> EmbeddingProvider:
    """Create an embedding provider based on config or explicit type.

    Args:
        provider_type: "openai", "local", or "mock". Defaults to settings.embedding_provider.
    """
    provider_type = provider_type or settings.embedding_provider

    if provider_type == "openai":
        return OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)
    elif provider_type == "local":
        return LocalEmbeddings(model_name=settings.local_model_name)
    elif provider_type == "mock":
        return MockEmbeddings(dimensions=settings.embedding_dimensions)
    else:
        raise ValueError(f"Unknown embedding provider: {provider_type}")


def format_for_embedding(message: dict) -> str:
    """Format a decision message for embedding.

    Combines context, reasoning, and action into embeddable text.
    More context = better semantic search, but more tokens = more cost.

    Args:
        message: Parsed JSON message with optional fields:
            - context: What prompted this decision
            - reasoning: Agent's thought process
            - action: What action was taken
            - anchor: File/location reference
    """
    parts = []

    # High-signal fields (always include if present)
    if context := message.get("context"):
        parts.append(f"Context: {context}")

    if reasoning := message.get("reasoning"):
        parts.append(f"Reasoning: {reasoning}")

    if action := message.get("action"):
        parts.append(f"Action: {action}")

    # Lower-signal but useful for anchor queries
    if anchor := message.get("anchor"):
        parts.append(f"File: {anchor}")

    # Fallback: if no structured fields, embed the whole thing
    if not parts:
        import json

        return json.dumps(message)

    return "\n".join(parts)
