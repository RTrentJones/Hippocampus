"""Kafka consumer for embedding generation.

Subscribes to agent decision topics and generates embeddings.
Supports both synchronous (v0.1) and batched (v0.2) modes.
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass

from kafka import KafkaConsumer
from kafka.consumer.fetcher import ConsumerRecord

from .config import settings
from .db import get_consumer_offset, store_embeddings_batch, update_consumer_offset
from .embeddings import format_for_embedding, get_provider

logger = logging.getLogger(__name__)


@dataclass
class ProcessedMessage:
    """A message ready for embedding storage."""

    topic_id: int
    partition_id: int
    partition_offset: int
    global_offset: int
    text: str


class EmbeddingConsumer:
    """Kafka consumer that generates embeddings for messages.

    Modes:
        - sync: Process one message at a time (simple, <100 msg/sec)
        - batch: Batch messages before embedding (efficient, ~1000 msg/sec)
    """

    def __init__(
        self,
        consumer_id: str = "hippocampus-main",
        topic_pattern: str | None = None,
        batch_size: int | None = None,
        batch_timeout_ms: int | None = None,
    ):
        self.consumer_id = consumer_id
        self.topic_pattern = topic_pattern or settings.kafka_topic_pattern
        self.batch_size = batch_size or settings.batch_size
        self.batch_timeout_ms = batch_timeout_ms or settings.batch_timeout_ms

        self.provider = get_provider()
        self._consumer: KafkaConsumer | None = None
        self._running = False
        self._stats = {"processed": 0, "errors": 0, "batches": 0}

    def _create_consumer(self) -> KafkaConsumer:
        """Create and configure the Kafka consumer."""
        consumer = KafkaConsumer(
            bootstrap_servers=settings.kafka_bootstrap_servers,
            group_id=settings.kafka_consumer_group,
            auto_offset_reset="earliest",
            enable_auto_commit=False,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")) if m else None,
            key_deserializer=lambda k: k.decode("utf-8") if k else None,
        )
        # Subscribe to pattern (e.g., decisions.*)
        consumer.subscribe(pattern=self.topic_pattern)
        return consumer

    async def start(self, mode: str = "batch"):
        """Start the consumer loop.

        Args:
            mode: "sync" for one-at-a-time, "batch" for batched processing
        """
        self._consumer = self._create_consumer()
        self._running = True

        logger.info(
            f"Starting consumer '{self.consumer_id}' in {mode} mode, "
            f"pattern='{self.topic_pattern}'"
        )

        try:
            if mode == "sync":
                await self._sync_loop()
            else:
                await self._batch_loop()
        finally:
            self._consumer.close()

    def stop(self):
        """Signal the consumer to stop."""
        self._running = False

    async def _sync_loop(self):
        """Process messages one at a time (simple mode)."""
        while self._running:
            # Poll with short timeout to allow stop signal
            records = self._consumer.poll(timeout_ms=1000, max_records=1)

            for topic_partition, messages in records.items():
                for msg in messages:
                    try:
                        await self._process_single(msg)
                        self._consumer.commit()
                        self._stats["processed"] += 1
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                        self._stats["errors"] += 1

            # Yield to event loop
            await asyncio.sleep(0)

    async def _batch_loop(self):
        """Process messages in batches (efficient mode)."""
        batch: list[tuple[ConsumerRecord, dict]] = []
        last_flush = time.monotonic()

        while self._running:
            # Poll for messages
            records = self._consumer.poll(timeout_ms=100, max_records=self.batch_size)

            # Collect messages
            for topic_partition, messages in records.items():
                for msg in messages:
                    if msg.value:
                        batch.append((msg, msg.value))

            # Check if we should flush
            elapsed_ms = (time.monotonic() - last_flush) * 1000
            should_flush = len(batch) >= self.batch_size or (
                batch and elapsed_ms > self.batch_timeout_ms
            )

            if should_flush:
                await self._process_batch(batch)
                self._consumer.commit()
                self._stats["batches"] += 1
                self._stats["processed"] += len(batch)
                batch = []
                last_flush = time.monotonic()

            # Yield to event loop
            await asyncio.sleep(0)

        # Flush remaining
        if batch:
            await self._process_batch(batch)
            self._consumer.commit()

    async def _process_single(self, msg: ConsumerRecord):
        """Process a single message."""
        if not msg.value:
            return

        text = format_for_embedding(msg.value)
        embedding = await self.provider.embed_one(text)

        # Get topic_id from topic name (need to look up in pg_kafka)
        topic_id = await self._get_topic_id(msg.topic)

        await store_embeddings_batch([
            (topic_id, msg.partition, msg.offset, embedding)
        ])

    async def _process_batch(self, batch: list[tuple[ConsumerRecord, dict]]):
        """Process a batch of messages."""
        if not batch:
            return

        # Format all texts
        texts = [format_for_embedding(value) for _, value in batch]

        # Single API call for all embeddings
        try:
            embeddings = await self.provider.embed(texts)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            self._stats["errors"] += len(batch)
            return

        # Look up topic IDs (cache these in production)
        topic_ids = {}
        for msg, _ in batch:
            if msg.topic not in topic_ids:
                topic_ids[msg.topic] = await self._get_topic_id(msg.topic)

        # Build records for batch insert
        records = []
        for (msg, _), embedding in zip(batch, embeddings):
            topic_id = topic_ids[msg.topic]
            records.append((topic_id, msg.partition, msg.offset, embedding))

        # Batch insert
        await store_embeddings_batch(records)

        logger.debug(f"Processed batch of {len(batch)} messages")

    async def _get_topic_id(self, topic_name: str) -> int:
        """Look up topic ID from pg_kafka's kafka.topics table."""
        from .db import get_pool

        pool = await get_pool()
        row = await pool.fetchrow(
            "SELECT id FROM kafka.topics WHERE name = $1", topic_name
        )
        if row:
            return row["id"]
        raise ValueError(f"Topic not found: {topic_name}")

    @property
    def stats(self) -> dict:
        """Get consumer statistics."""
        return self._stats.copy()


async def run_consumer(mode: str = "batch"):
    """Run the embedding consumer (entry point)."""
    consumer = EmbeddingConsumer()
    await consumer.start(mode=mode)
