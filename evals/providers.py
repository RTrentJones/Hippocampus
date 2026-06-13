"""Eval-only embedding providers.

FastEmbedProvider gives real neural embeddings (ONNX all-MiniLM-L6-v2 via
fastembed) without torch and without HuggingFace access — fastembed falls
back to a GCS mirror. Useful for CI-adjacent environments where
sentence-transformers can't download models.

PaddedProvider zero-pads vectors up to the schema's fixed dimension
(vector(1536)). Zero-padding preserves cosine similarity exactly —
cos(pad(a), pad(b)) == cos(a, b) — so rankings are unaffected; it only
exists so smaller models fit the column.
"""

import asyncio

from hippocampus.embeddings import EmbeddingProvider

SCHEMA_DIMENSIONS = 1536  # hippocampus.embeddings is vector(1536)


class FastEmbedProvider(EmbeddingProvider):
    """all-MiniLM-L6-v2 via fastembed (ONNX, no torch)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name)
        self._dimensions = 384

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = await asyncio.to_thread(lambda: list(self.model.embed(texts)))
        return [v.tolist() for v in vectors]


class PaddedProvider(EmbeddingProvider):
    """Zero-pad a smaller provider's vectors to the schema dimension."""

    def __init__(self, inner: EmbeddingProvider, target: int = SCHEMA_DIMENSIONS):
        if inner.dimensions > target:
            raise ValueError(f"Provider dims {inner.dimensions} exceed target {target}")
        self.inner = inner
        self.target = target

    @property
    def dimensions(self) -> int:
        return self.target

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = await self.inner.embed(texts)
        pad = self.target - self.inner.dimensions
        return [v + [0.0] * pad for v in vectors]


def make_provider(name: str) -> EmbeddingProvider:
    """Create an eval provider by name, padded to the schema dimension."""
    from hippocampus.embeddings import create_provider

    provider = FastEmbedProvider() if name == "fastembed" else create_provider(name)
    if provider.dimensions < SCHEMA_DIMENSIONS:
        provider = PaddedProvider(provider)
    return provider
