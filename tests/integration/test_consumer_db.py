"""Integration tests for consumer with database operations.

Tests the consumer's ability to process messages and store embeddings.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.integration
class TestConsumerProcessing:
    """Tests for consumer message processing."""

    async def test_format_for_embedding_integration(self, sample_decision):
        """Test message formatting produces valid embedding input."""
        from hippocampus.embeddings import format_for_embedding

        text = format_for_embedding(sample_decision)

        assert isinstance(text, str)
        assert len(text) > 0
        assert "Context:" in text
        assert "Reasoning:" in text
        assert "Action:" in text

    async def test_process_and_embed(
        self, patched_db_pool, seeded_topic, embedding_provider, sample_decision
    ):
        """Test that a message can be formatted, embedded, and stored."""
        from hippocampus.db import store_embedding
        from hippocampus.embeddings import format_for_embedding

        topic_id = seeded_topic["topic_id"]

        # Format message
        text = format_for_embedding(sample_decision)

        # Generate embedding
        embedding = await embedding_provider.embed_one(text)

        # Store
        result = await store_embedding(
            topic_id=topic_id,
            partition_id=0,
            partition_offset=500,
            embedding=embedding,
        )

        assert result is True

        # Verify storage
        row = await patched_db_pool.fetchrow(
            """
            SELECT * FROM hippocampus.embeddings
            WHERE topic_id = $1 AND partition_offset = 500
            """,
            topic_id,
        )
        assert row is not None

    async def test_batch_process_and_embed(
        self, patched_db_pool, seeded_topic, embedding_provider, sample_decisions
    ):
        """Test batch processing of multiple messages."""
        from hippocampus.db import store_embeddings_batch
        from hippocampus.embeddings import format_for_embedding

        topic_id = seeded_topic["topic_id"]

        # Format all messages
        texts = [format_for_embedding(msg) for msg in sample_decisions]

        # Batch embed
        embeddings = await embedding_provider.embed(texts)
        assert len(embeddings) == len(sample_decisions)

        # Store batch
        records = [(topic_id, 0, 600 + i, emb) for i, emb in enumerate(embeddings)]
        inserted, _offsets = await store_embeddings_batch(records)

        assert inserted == len(sample_decisions)

        # Verify all stored
        count = await patched_db_pool.fetchval(
            """
            SELECT COUNT(*) FROM hippocampus.embeddings
            WHERE topic_id = $1 AND partition_offset >= 600
            """,
            topic_id,
        )
        assert count == len(sample_decisions)


@pytest.mark.integration
class TestEmbeddingConsumer:
    """Tests for the EmbeddingConsumer class."""

    async def test_consumer_init(self, patched_settings, patched_provider):
        """Test consumer initialization."""
        from hippocampus.consumer import EmbeddingConsumer

        consumer = EmbeddingConsumer(consumer_id="test-consumer")

        assert consumer.consumer_id == "test-consumer"
        assert consumer.topic_pattern == patched_settings.kafka_topic_pattern
        assert consumer.batch_size == patched_settings.batch_size
        assert consumer._running is False
        assert consumer.stats == {"processed": 0, "errors": 0, "batches": 0, "edges": 0}

    async def test_consumer_stats(self, patched_settings, patched_provider):
        """Test consumer stats are independent copies."""
        from hippocampus.consumer import EmbeddingConsumer

        consumer = EmbeddingConsumer()
        stats1 = consumer.stats
        stats2 = consumer.stats

        # Should be copies, not the same object
        assert stats1 is not stats2
        assert stats1 == stats2

    async def test_get_topic_id(self, patched_db_pool, seeded_topic, patched_provider):
        """Test topic ID lookup."""
        from hippocampus.consumer import EmbeddingConsumer

        consumer = EmbeddingConsumer()

        topic_id = await consumer._get_topic_id(seeded_topic["topic_name"])

        assert topic_id == seeded_topic["topic_id"]

    async def test_get_topic_id_not_found(self, patched_db_pool, patched_provider):
        """Test topic ID lookup for non-existent topic."""
        from hippocampus.consumer import EmbeddingConsumer

        consumer = EmbeddingConsumer()

        with pytest.raises(ValueError, match="Topic not found"):
            await consumer._get_topic_id("nonexistent-topic")


@pytest.mark.integration
class TestConsumerProcessSingle:
    """Tests for single message processing."""

    async def test_process_single_calls_embed(
        self, patched_db_pool, seeded_topic, mock_consumer_record, patched_provider
    ):
        """Test that _process_single generates embedding."""
        from hippocampus.consumer import EmbeddingConsumer

        consumer = EmbeddingConsumer()

        # Create a mock record
        record = mock_consumer_record(
            topic=seeded_topic["topic_name"],
            partition=0,
            offset=700,
            value={"context": "Test", "action": "Test action"},
        )

        # Process
        await consumer._process_single(record)

        # Verify embedding was stored
        row = await patched_db_pool.fetchrow(
            """
            SELECT * FROM hippocampus.embeddings
            WHERE topic_id = $1 AND partition_offset = 700
            """,
            seeded_topic["topic_id"],
        )
        assert row is not None

    async def test_process_single_skips_empty_value(
        self, patched_db_pool, seeded_topic, mock_consumer_record, patched_provider
    ):
        """Test that messages with no value are skipped."""
        from hippocampus.consumer import EmbeddingConsumer

        consumer = EmbeddingConsumer()

        record = mock_consumer_record(
            topic=seeded_topic["topic_name"],
            partition=0,
            offset=701,
            value=None,
            use_default_value=False,  # Explicitly test None value handling
        )

        # Should not raise
        await consumer._process_single(record)

        # Should not store anything
        row = await patched_db_pool.fetchrow(
            """
            SELECT * FROM hippocampus.embeddings
            WHERE topic_id = $1 AND partition_offset = 701
            """,
            seeded_topic["topic_id"],
        )
        assert row is None


@pytest.mark.integration
class TestConsumerProcessBatch:
    """Tests for batch message processing."""

    async def test_process_batch_stores_all(
        self, patched_db_pool, seeded_topic, mock_consumer_record, patched_provider
    ):
        """Test that _process_batch stores all embeddings."""
        from hippocampus.consumer import EmbeddingConsumer

        consumer = EmbeddingConsumer()

        # Create batch
        batch = [
            (
                mock_consumer_record(
                    topic=seeded_topic["topic_name"],
                    partition=0,
                    offset=800 + i,
                    value={"context": f"Test {i}", "action": f"Action {i}"},
                ),
                {"context": f"Test {i}", "action": f"Action {i}"},
            )
            for i in range(3)
        ]

        await consumer._process_batch(batch)

        # Verify all stored
        count = await patched_db_pool.fetchval(
            """
            SELECT COUNT(*) FROM hippocampus.embeddings
            WHERE topic_id = $1 AND partition_offset >= 800 AND partition_offset < 803
            """,
            seeded_topic["topic_id"],
        )
        assert count == 3

    async def test_process_batch_empty(self, patched_db_pool, patched_provider):
        """Test that empty batch is handled."""
        from hippocampus.consumer import EmbeddingConsumer

        consumer = EmbeddingConsumer()

        # Should not raise
        await consumer._process_batch([])

    async def test_process_batch_handles_embedding_error(
        self, patched_db_pool, seeded_topic, mock_consumer_record, patched_provider
    ):
        """Test that embedding errors are handled gracefully."""
        from hippocampus.consumer import EmbeddingConsumer

        # patched_provider keeps EmbeddingConsumer() from building a real
        # (credentialed) provider; the failing mock below replaces it anyway.
        consumer = EmbeddingConsumer()

        # Mock provider to raise error
        consumer.provider = MagicMock()
        consumer.provider.embed = AsyncMock(side_effect=Exception("API error"))

        batch = [
            (
                mock_consumer_record(
                    topic=seeded_topic["topic_name"],
                    partition=0,
                    offset=900,
                    value={"context": "Test", "action": "Action"},
                ),
                {"context": "Test", "action": "Action"},
            )
        ]

        # Should not raise
        await consumer._process_batch(batch)

        # Should increment error counter
        assert consumer.stats["errors"] == 1


@pytest.mark.integration
class TestConsumerLoopMocked:
    """Tests for consumer loops with mocked Kafka."""

    async def test_consumer_stop_signal(self, patched_settings, patched_provider):
        """Test that stop() signal stops the consumer."""
        from hippocampus.consumer import EmbeddingConsumer

        consumer = EmbeddingConsumer()

        # Start would run forever, so we just test the flag
        assert consumer._running is False
        consumer._running = True
        consumer.stop()
        assert consumer._running is False

    async def test_sync_loop_processes_message(
        self, patched_db_pool, seeded_topic, patched_provider
    ):
        """Test sync loop processes a single message."""

        from hippocampus.consumer import EmbeddingConsumer

        consumer = EmbeddingConsumer()

        # Mock Kafka consumer
        mock_kafka = MagicMock()
        mock_record = MagicMock()
        mock_record.topic = seeded_topic["topic_name"]
        mock_record.partition = 0
        mock_record.offset = 950
        mock_record.value = {"context": "Sync test", "action": "Test action"}

        # First poll returns message, second poll triggers stop
        call_count = [0]

        def poll_side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return {("topic", 0): [mock_record]}
            else:
                consumer.stop()
                return {}

        mock_kafka.poll.side_effect = poll_side_effect
        mock_kafka.commit.return_value = None
        mock_kafka.close.return_value = None

        consumer._consumer = mock_kafka
        consumer._running = True

        # Run sync loop
        await consumer._sync_loop()

        # Verify message was processed
        assert consumer.stats["processed"] == 1
        mock_kafka.commit.assert_called()
