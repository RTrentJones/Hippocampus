"""Tests for embedding providers."""

import pytest

from hippocampus.embeddings import (
    MockEmbeddings,
    format_for_embedding,
    create_provider,
)


class TestFormatForEmbedding:
    """Tests for message formatting."""

    def test_full_message(self):
        """Test formatting a complete message."""
        msg = {
            "context": "User asked to refactor",
            "reasoning": "Current code is messy",
            "action": "Extract class",
            "anchor": "src/auth.py:45",
        }
        result = format_for_embedding(msg)

        assert "Context: User asked to refactor" in result
        assert "Reasoning: Current code is messy" in result
        assert "Action: Extract class" in result
        assert "File: src/auth.py:45" in result

    def test_partial_message(self):
        """Test formatting a message with missing fields."""
        msg = {
            "context": "User request",
            "action": "Do something",
        }
        result = format_for_embedding(msg)

        assert "Context: User request" in result
        assert "Action: Do something" in result
        assert "Reasoning" not in result

    def test_empty_message(self):
        """Test formatting an empty message falls back to JSON."""
        msg = {"custom_field": "value"}
        result = format_for_embedding(msg)

        # Should JSON-encode the whole thing
        assert "custom_field" in result


class TestMockEmbeddings:
    """Tests for mock embedding provider."""

    @pytest.mark.asyncio
    async def test_embed_single(self):
        """Test embedding a single text."""
        provider = MockEmbeddings(dimensions=384)
        result = await provider.embed_one("test text")

        assert len(result) == 384
        assert all(isinstance(v, float) for v in result)

    @pytest.mark.asyncio
    async def test_embed_batch(self):
        """Test embedding multiple texts."""
        provider = MockEmbeddings(dimensions=384)
        texts = ["text one", "text two", "text three"]
        results = await provider.embed(texts)

        assert len(results) == 3
        assert all(len(r) == 384 for r in results)

    @pytest.mark.asyncio
    async def test_embed_deterministic(self):
        """Test that same input produces same output."""
        provider = MockEmbeddings(dimensions=384)
        result1 = await provider.embed_one("deterministic test")
        result2 = await provider.embed_one("deterministic test")

        assert result1 == result2

    @pytest.mark.asyncio
    async def test_embed_different_inputs(self):
        """Test that different inputs produce different outputs."""
        provider = MockEmbeddings(dimensions=384)
        result1 = await provider.embed_one("text one")
        result2 = await provider.embed_one("text two")

        assert result1 != result2

    @pytest.mark.asyncio
    async def test_embed_empty_list(self):
        """Test embedding an empty list."""
        provider = MockEmbeddings(dimensions=384)
        results = await provider.embed([])

        assert results == []


class TestProviderFactory:
    """Tests for provider factory."""

    def test_create_mock_provider(self):
        """Test creating mock provider."""
        provider = create_provider("mock")
        assert provider.dimensions == 1536  # default

    def test_invalid_provider(self):
        """Test that invalid provider raises error."""
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            create_provider("invalid")
