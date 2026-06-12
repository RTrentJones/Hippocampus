"""Integration tests for causal edges, hybrid search, and the embedding watermark.

These exercise the Phase 1 features end to end against the database:
- explicit causal edges declared via parent_id/caused_by payload fields
- recursive causal chain traversal with temporal fallback
- hybrid (vector + full-text) retrieval with RRF fusion
- consumer embedding watermark updates
- indexed anchor lookups for what_touched
"""

import json

import pytest


async def _insert_message(db_pool, topic_id: int, partition_offset: int, value: dict) -> int:
    """Insert a kafka message and return its global_offset."""
    row = await db_pool.fetchrow(
        """
        INSERT INTO kafka.messages (topic_id, partition_id, partition_offset, value)
        VALUES ($1, 0, $2, $3)
        RETURNING global_offset
        """,
        topic_id,
        partition_offset,
        json.dumps(value).encode("utf-8"),
    )
    return row["global_offset"]


def _record(topic: str, offset: int, value: dict):
    """Build a mock ConsumerRecord."""
    from unittest.mock import MagicMock

    record = MagicMock()
    record.topic = topic
    record.partition = 0
    record.offset = offset
    record.value = value
    return record


@pytest.mark.integration
class TestCausalEdges:
    """Tests for causal edge storage and traversal."""

    async def test_consumer_records_causal_edges(
        self, patched_db_pool, seeded_topic, patched_provider
    ):
        """A chain declared via parent_id becomes causal edges."""
        from hippocampus.consumer import EmbeddingConsumer

        topic_id = seeded_topic["topic_id"]
        topic_name = seeded_topic["topic_name"]

        decisions = [
            {"id": "d1", "context": "saw failure", "action": "open logs"},
            {"id": "d2", "parent_id": "d1", "context": "found error", "action": "git blame"},
            {"id": "d3", "parent_id": "d2", "context": "found commit", "action": "write fix"},
        ]
        for i, decision in enumerate(decisions):
            await _insert_message(patched_db_pool, topic_id, 2000 + i, decision)

        consumer = EmbeddingConsumer(consumer_id="test-causal")
        batch = [
            (_record(topic_name, 2000 + i, decision), decision)
            for i, decision in enumerate(decisions)
        ]
        assert await consumer._process_batch(batch) is True
        assert consumer.stats["edges"] == 2

        edge_count = await patched_db_pool.fetchval("SELECT COUNT(*) FROM hippocampus.causal_edges")
        assert edge_count == 2

    async def test_replay_causal_chain_follows_edges(
        self, patched_db_pool, seeded_topic, patched_provider
    ):
        """replay_causal_chain walks the graph, skipping unrelated noise."""
        from hippocampus.consumer import EmbeddingConsumer
        from hippocampus.db import replay_causal_chain

        topic_id = seeded_topic["topic_id"]
        topic_name = seeded_topic["topic_name"]

        # d1 -> d2 -> d3 with an unrelated decision interleaved
        decisions = [
            {"id": "c1", "context": "root cause event", "action": "change config"},
            {"id": "noise", "context": "unrelated chatter", "action": "noop"},
            {"id": "c2", "parent_id": "c1", "context": "config applied", "action": "restart"},
            {"id": "c3", "parent_id": "c2", "context": "server crashed", "action": "investigate"},
        ]
        offsets = {}
        for i, decision in enumerate(decisions):
            offsets[decision["id"]] = await _insert_message(
                patched_db_pool, topic_id, 2100 + i, decision
            )

        consumer = EmbeddingConsumer(consumer_id="test-chain")
        batch = [
            (_record(topic_name, 2100 + i, decision), decision)
            for i, decision in enumerate(decisions)
        ]
        await consumer._process_batch(batch)

        chain = await replay_causal_chain(offsets["c3"], lookback=10)

        assert [event["retrieval"] for event in chain] == ["causal_graph"] * 3
        chain_ids = [event["value"]["id"] for event in chain]
        assert chain_ids == ["c1", "c2", "c3"]  # chronological, noise excluded

    async def test_replay_causal_chain_falls_back_to_temporal(
        self, patched_db_pool, seeded_messages
    ):
        """Without edges, replay_causal_chain degrades to the temporal window."""
        from hippocampus.db import replay_causal_chain

        anchor = seeded_messages["messages"][-1]["global_offset"]
        results = await replay_causal_chain(anchor, lookback=10)

        assert len(results) > 0
        assert all(event["retrieval"] == "temporal_window" for event in results)

    async def test_unresolvable_parent_skipped(
        self, patched_db_pool, seeded_topic, patched_provider
    ):
        """An unknown parent_id is skipped without failing the batch."""
        from hippocampus.consumer import EmbeddingConsumer

        topic_id = seeded_topic["topic_id"]
        topic_name = seeded_topic["topic_name"]

        decision = {"id": "orphan", "parent_id": "never-seen", "action": "act"}
        await _insert_message(patched_db_pool, topic_id, 2200, decision)

        consumer = EmbeddingConsumer(consumer_id="test-orphan")
        assert await consumer._process_batch([(_record(topic_name, 2200, decision), decision)])
        assert consumer.stats["edges"] == 0

    async def test_cross_batch_parent_resolution(
        self, patched_db_pool, seeded_topic, patched_provider
    ):
        """A parent embedded in an earlier batch is resolved via the database."""
        from hippocampus.consumer import EmbeddingConsumer

        topic_id = seeded_topic["topic_id"]
        topic_name = seeded_topic["topic_name"]

        parent = {"id": "p1", "context": "first batch", "action": "start"}
        child = {"id": "ch1", "parent_id": "p1", "context": "second batch", "action": "finish"}
        await _insert_message(patched_db_pool, topic_id, 2300, parent)
        await _insert_message(patched_db_pool, topic_id, 2301, child)

        consumer = EmbeddingConsumer(consumer_id="test-cross-batch")
        await consumer._process_batch([(_record(topic_name, 2300, parent), parent)])
        await consumer._process_batch([(_record(topic_name, 2301, child), child)])

        assert consumer.stats["edges"] == 1


@pytest.mark.integration
class TestHybridSearch:
    """Tests for hybrid (vector + full-text) retrieval."""

    async def test_hybrid_search_returns_scored_results(
        self, patched_db_pool, seeded_topic, embedding_provider
    ):
        """Hybrid search returns RRF-scored results."""
        from hippocampus.db import hybrid_search, store_embedding

        topic_id = seeded_topic["topic_id"]
        text = "Fixed NullPointerException in AuthService token validation"
        await _insert_message(patched_db_pool, topic_id, 2400, {"action": text})
        embedding = await embedding_provider.embed_one(text)
        await store_embedding(topic_id, 0, 2400, embedding, content_text=text)

        query_embedding = await embedding_provider.embed_one("token validation error")
        results = await hybrid_search("token validation error", query_embedding, limit=5)

        assert len(results) > 0
        assert "rrf_score" in results[0]
        scores = [r["rrf_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    async def test_hybrid_keyword_match_boosts_exact_terms(
        self, patched_db_pool, seeded_topic, embedding_provider
    ):
        """A document containing the exact query term ranks first even when
        embeddings (mock = content hash) carry no semantic signal."""
        from hippocampus.db import hybrid_search, store_embedding

        topic_id = seeded_topic["topic_id"]
        texts = {
            2500: "Refactored the billing module for clarity",
            2501: "Fixed crash in QuantumFluxCapacitor initialization",
            2502: "Updated documentation for the API gateway",
        }
        for offset, text in texts.items():
            await _insert_message(patched_db_pool, topic_id, offset, {"action": text})
            embedding = await embedding_provider.embed_one(text)
            await store_embedding(topic_id, 0, offset, embedding, content_text=text)

        query = "QuantumFluxCapacitor crash"
        query_embedding = await embedding_provider.embed_one(query)
        results = await hybrid_search(query, query_embedding, limit=3)

        assert len(results) > 0
        assert "QuantumFluxCapacitor" in results[0]["value"]["action"]

    async def test_hybrid_search_topic_filter(
        self, patched_db_pool, seeded_topic, embedding_provider
    ):
        """Topic pattern filtering applies to hybrid results."""
        from hippocampus.db import hybrid_search, store_embedding

        topic_id = seeded_topic["topic_id"]
        text = "topic filter test entry"
        await _insert_message(patched_db_pool, topic_id, 2600, {"action": text})
        embedding = await embedding_provider.embed_one(text)
        await store_embedding(topic_id, 0, 2600, embedding, content_text=text)

        query_embedding = await embedding_provider.embed_one(text)
        results = await hybrid_search(text, query_embedding, limit=5, topic_pattern="nonexistent.%")
        assert results == []


@pytest.mark.integration
class TestEmbeddingWatermark:
    """Tests for consumer_state watermark maintenance."""

    async def test_process_batch_advances_watermark(
        self, patched_db_pool, seeded_topic, patched_provider
    ):
        """Processing a batch records the max embedded global_offset."""
        from hippocampus.consumer import EmbeddingConsumer
        from hippocampus.db import get_consumer_offset

        topic_id = seeded_topic["topic_id"]
        topic_name = seeded_topic["topic_name"]

        decision = {"id": "w1", "context": "watermark test", "action": "act"}
        global_offset = await _insert_message(patched_db_pool, topic_id, 2700, decision)

        consumer = EmbeddingConsumer(consumer_id="test-watermark")
        await consumer._process_batch([(_record(topic_name, 2700, decision), decision)])

        assert await get_consumer_offset("test-watermark") == global_offset

    async def test_watermark_never_regresses(self, patched_db_pool):
        """update_consumer_offset keeps the high-water mark."""
        from hippocampus.db import get_consumer_offset, update_consumer_offset

        await update_consumer_offset("test-monotonic", 100)
        await update_consumer_offset("test-monotonic", 50)

        assert await get_consumer_offset("test-monotonic") == 100


@pytest.mark.integration
class TestWhatTouchedIndexedPath:
    """Tests for the indexed anchor column path of what_touched."""

    async def test_anchor_column_used_when_populated(
        self, patched_db_pool, seeded_topic, patched_provider
    ):
        """Embedded decisions are found via the anchor column."""
        from hippocampus.consumer import EmbeddingConsumer
        from hippocampus.db import what_touched

        topic_id = seeded_topic["topic_id"]
        topic_name = seeded_topic["topic_name"]

        decision = {
            "id": "a1",
            "context": "touched a file",
            "action": "edit",
            "anchor": "src/payments/stripe_client.py:88",
        }
        await _insert_message(patched_db_pool, topic_id, 2800, decision)

        consumer = EmbeddingConsumer(consumer_id="test-anchor")
        await consumer._process_batch([(_record(topic_name, 2800, decision), decision)])

        results = await what_touched("stripe_client.py")
        assert len(results) == 1
        assert results[0]["value"]["id"] == "a1"
