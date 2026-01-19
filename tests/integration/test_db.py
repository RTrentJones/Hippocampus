"""Integration tests for database operations.

These tests require a running PostgreSQL instance with pg_kafka and pgvector extensions.
Run with: pytest tests/integration/test_db.py -v
"""

import pytest


@pytest.mark.integration
class TestEmbeddingStorage:
    """Tests for embedding storage operations."""

    async def test_store_embedding_success(self, patched_db_pool, seeded_topic, embedding_provider):
        """Test storing a single embedding."""
        from hippocampus.db import store_embedding

        topic_id = seeded_topic["topic_id"]
        embedding = await embedding_provider.embed_one("Test message for embedding")

        result = await store_embedding(
            topic_id=topic_id,
            partition_id=0,
            partition_offset=100,
            embedding=embedding,
        )

        assert result is True

        # Verify it was stored
        row = await patched_db_pool.fetchrow(
            """
            SELECT * FROM hippocampus.embeddings
            WHERE topic_id = $1 AND partition_id = 0 AND partition_offset = 100
            """,
            topic_id,
        )
        assert row is not None

    async def test_store_embedding_idempotent(
        self, patched_db_pool, seeded_topic, embedding_provider
    ):
        """Test that duplicate embedding insert is handled gracefully."""
        from hippocampus.db import store_embedding

        topic_id = seeded_topic["topic_id"]
        embedding = await embedding_provider.embed_one("Duplicate test")

        # First insert
        result1 = await store_embedding(
            topic_id=topic_id,
            partition_id=0,
            partition_offset=200,
            embedding=embedding,
        )
        assert result1 is True

        # Second insert (duplicate) - should not raise
        result2 = await store_embedding(
            topic_id=topic_id,
            partition_id=0,
            partition_offset=200,
            embedding=embedding,
        )
        # Note: Current implementation always returns True due to ON CONFLICT DO NOTHING
        assert result2 is True

    async def test_store_embeddings_batch(
        self, patched_db_pool, seeded_topic, embedding_provider
    ):
        """Test batch embedding storage."""
        from hippocampus.db import store_embeddings_batch

        topic_id = seeded_topic["topic_id"]
        texts = ["Batch text 1", "Batch text 2", "Batch text 3"]
        embeddings = await embedding_provider.embed(texts)

        records = [
            (topic_id, 0, 300 + i, emb) for i, emb in enumerate(embeddings)
        ]

        result = await store_embeddings_batch(records)
        assert result == 3

        # Verify all were stored
        count = await patched_db_pool.fetchval(
            """
            SELECT COUNT(*) FROM hippocampus.embeddings
            WHERE topic_id = $1 AND partition_offset >= 300 AND partition_offset < 303
            """,
            topic_id,
        )
        assert count == 3

    async def test_store_embeddings_batch_empty(self, patched_db_pool):
        """Test batch storage with empty list."""
        from hippocampus.db import store_embeddings_batch

        result = await store_embeddings_batch([])
        assert result == 0


@pytest.mark.integration
class TestConsumerState:
    """Tests for consumer offset tracking."""

    async def test_get_consumer_offset_not_exists(self, patched_db_pool):
        """Test getting offset for non-existent consumer returns 0."""
        from hippocampus.db import get_consumer_offset

        offset = await get_consumer_offset("nonexistent-consumer")
        assert offset == 0

    async def test_update_and_get_consumer_offset(self, patched_db_pool):
        """Test updating and retrieving consumer offset."""
        from hippocampus.db import get_consumer_offset, update_consumer_offset

        consumer_id = "test-consumer-001"

        # Update offset
        await update_consumer_offset(consumer_id, 100)

        # Retrieve
        offset = await get_consumer_offset(consumer_id)
        assert offset == 100

    async def test_update_consumer_offset_upsert(self, patched_db_pool):
        """Test that consumer offset updates existing record."""
        from hippocampus.db import get_consumer_offset, update_consumer_offset

        consumer_id = "test-consumer-002"

        # Initial
        await update_consumer_offset(consumer_id, 50)
        assert await get_consumer_offset(consumer_id) == 50

        # Update
        await update_consumer_offset(consumer_id, 150)
        assert await get_consumer_offset(consumer_id) == 150


@pytest.mark.integration
class TestSemanticSearch:
    """Tests for semantic search (find_similar)."""

    async def test_find_similar_returns_results(
        self, patched_db_pool, seeded_embeddings, embedding_provider
    ):
        """Test that find_similar returns results."""
        from hippocampus.db import find_similar

        # Query for something similar to seeded data
        query_embedding = await embedding_provider.embed_one("authentication bug fix")

        results = await find_similar(query_embedding, limit=10)

        assert len(results) > 0
        assert "value" in results[0]
        assert "topic" in results[0]
        assert "distance" in results[0]

    async def test_find_similar_returns_ranked(
        self, patched_db_pool, seeded_embeddings, embedding_provider
    ):
        """Test that results are ranked by distance (ascending)."""
        from hippocampus.db import find_similar

        query_embedding = await embedding_provider.embed_one("authentication")

        results = await find_similar(query_embedding, limit=10)

        if len(results) > 1:
            # Distance should be in ascending order (most similar first)
            distances = [r["distance"] for r in results]
            assert distances == sorted(distances)

    async def test_find_similar_respects_limit(
        self, patched_db_pool, seeded_embeddings, embedding_provider
    ):
        """Test that limit parameter is respected."""
        from hippocampus.db import find_similar

        query_embedding = await embedding_provider.embed_one("test query")

        results = await find_similar(query_embedding, limit=2)
        assert len(results) <= 2

    async def test_find_similar_with_topic_pattern(
        self, patched_db_pool, seeded_embeddings, embedding_provider
    ):
        """Test filtering by topic pattern."""
        from hippocampus.db import find_similar

        query_embedding = await embedding_provider.embed_one("test query")

        results = await find_similar(
            query_embedding, limit=10, topic_pattern="decisions.test.%"
        )

        # All results should match the pattern
        for result in results:
            assert result["topic"].startswith("decisions.test.")

    async def test_find_similar_no_results(
        self, patched_db_pool, seeded_embeddings, embedding_provider
    ):
        """Test find_similar with non-matching topic pattern."""
        from hippocampus.db import find_similar

        query_embedding = await embedding_provider.embed_one("test query")

        results = await find_similar(
            query_embedding, limit=10, topic_pattern="nonexistent.%"
        )

        assert len(results) == 0


@pytest.mark.integration
class TestTemporalTraversal:
    """Tests for temporal traversal queries."""

    async def test_replay_causal_chain_returns_results(
        self, patched_db_pool, seeded_messages
    ):
        """Test replaying causal chain."""
        from hippocampus.db import replay_causal_chain

        # Get the last message's global offset
        last_msg = seeded_messages["messages"][-1]
        anchor_offset = last_msg["global_offset"]

        results = await replay_causal_chain(anchor_offset, lookback=10)

        assert len(results) > 0

    async def test_replay_causal_chain_chronological_order(
        self, patched_db_pool, seeded_messages
    ):
        """Test that causal chain returns events in chronological order (oldest first)."""
        from hippocampus.db import replay_causal_chain

        last_msg = seeded_messages["messages"][-1]
        anchor_offset = last_msg["global_offset"]

        results = await replay_causal_chain(anchor_offset, lookback=10)

        if len(results) > 1:
            offsets = [r["global_offset"] for r in results]
            assert offsets == sorted(offsets), "Results should be in chronological order"

    async def test_replay_causal_chain_with_topic_pattern(
        self, patched_db_pool, seeded_messages
    ):
        """Test causal chain with topic filtering."""
        from hippocampus.db import replay_causal_chain

        last_msg = seeded_messages["messages"][-1]
        anchor_offset = last_msg["global_offset"]

        results = await replay_causal_chain(
            anchor_offset, lookback=10, topic_pattern="decisions.test.%"
        )

        for result in results:
            assert result["topic"].startswith("decisions.test.")


@pytest.mark.integration
class TestReplayTopic:
    """Tests for topic replay."""

    async def test_replay_topic_returns_results(self, patched_db_pool, seeded_messages):
        """Test replaying a topic."""
        from hippocampus.db import replay_topic

        topic_name = seeded_messages["topic_name"]

        results = await replay_topic(topic_name, from_offset=0, limit=100)

        assert len(results) > 0
        assert len(results) == len(seeded_messages["messages"])

    async def test_replay_topic_from_offset(self, patched_db_pool, seeded_messages):
        """Test replaying from a specific offset."""
        from hippocampus.db import replay_topic

        topic_name = seeded_messages["topic_name"]

        results = await replay_topic(topic_name, from_offset=1, limit=100)

        # Should skip the first message
        assert len(results) == len(seeded_messages["messages"]) - 1

    async def test_replay_topic_respects_limit(self, patched_db_pool, seeded_messages):
        """Test that limit is respected."""
        from hippocampus.db import replay_topic

        topic_name = seeded_messages["topic_name"]

        results = await replay_topic(topic_name, from_offset=0, limit=1)

        assert len(results) == 1

    async def test_replay_topic_chronological_order(self, patched_db_pool, seeded_messages):
        """Test that results are in chronological order."""
        from hippocampus.db import replay_topic

        topic_name = seeded_messages["topic_name"]

        results = await replay_topic(topic_name, from_offset=0, limit=100)

        if len(results) > 1:
            offsets = [r["partition_offset"] for r in results]
            assert offsets == sorted(offsets)


@pytest.mark.integration
class TestTemporalContext:
    """Tests for temporal context queries."""

    async def test_temporal_context_returns_results(self, patched_db_pool, seeded_messages):
        """Test getting temporal context."""
        from hippocampus.db import temporal_context

        # Use middle message's offset
        mid_msg = seeded_messages["messages"][1]
        center_offset = mid_msg["global_offset"]

        results = await temporal_context(center_offset, window=10)

        assert len(results) > 0

    async def test_temporal_context_window(self, patched_db_pool, seeded_messages):
        """Test that window parameter works correctly."""
        from hippocampus.db import temporal_context

        mid_msg = seeded_messages["messages"][1]
        center_offset = mid_msg["global_offset"]

        results = await temporal_context(center_offset, window=1)

        # All results should be within window
        for result in results:
            assert abs(result["global_offset"] - center_offset) <= 1


@pytest.mark.integration
class TestWhatTouched:
    """Tests for what_touched queries."""

    async def test_what_touched_finds_anchors(self, patched_db_pool, seeded_messages):
        """Test finding decisions by anchor."""
        from hippocampus.db import what_touched

        # Search for auth.py which is in seeded messages
        results = await what_touched("auth.py", limit=10)

        assert len(results) > 0
        # All results should have matching anchor
        for result in results:
            assert "auth.py" in result["value"].get("anchor", "")

    async def test_what_touched_partial_match(self, patched_db_pool, seeded_messages):
        """Test partial anchor matching."""
        from hippocampus.db import what_touched

        # Partial match
        results = await what_touched("auth", limit=10)

        assert len(results) > 0

    async def test_what_touched_respects_limit(self, patched_db_pool, seeded_messages):
        """Test that limit is respected."""
        from hippocampus.db import what_touched

        results = await what_touched("auth", limit=1)

        assert len(results) <= 1

    async def test_what_touched_no_results(self, patched_db_pool, seeded_messages):
        """Test what_touched with non-matching anchor."""
        from hippocampus.db import what_touched

        results = await what_touched("nonexistent-file.xyz", limit=10)

        assert len(results) == 0

    async def test_what_touched_newest_first(self, patched_db_pool, seeded_messages):
        """Test that results are ordered newest first."""
        from hippocampus.db import what_touched

        results = await what_touched("auth", limit=10)

        if len(results) > 1:
            offsets = [r["global_offset"] for r in results]
            assert offsets == sorted(offsets, reverse=True)
