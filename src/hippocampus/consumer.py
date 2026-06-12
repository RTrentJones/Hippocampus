"""Kafka consumer for embedding generation.

Subscribes to agent decision topics, generates embeddings, and records causal
edges declared in decision payloads (`parent_id` / `caused_by`).

Failure semantics: offsets are committed to Kafka only after a batch has been
fully embedded and stored. A batch that keeps failing after
`consumer_max_retries` attempts stops the consumer (fail-stop) rather than
being skipped, so no message is ever committed past without being processed.
"""

import asyncio
import json
import logging
import time

from kafka import KafkaConsumer
from kafka.consumer.fetcher import ConsumerRecord

from .config import settings
from .db import (
    get_consumer_offset,
    resolve_decision_offsets,
    store_causal_edges,
    store_embeddings_batch,
    update_consumer_offset,
)
from .embeddings import format_for_embedding, get_provider

logger = logging.getLogger(__name__)


def extract_cause_ids(value: dict) -> list[str]:
    """Extract declared causal parent decision IDs from a payload.

    Producers declare causality with either `parent_id` (single ID) or
    `caused_by` (single ID or list of IDs). IDs are producer-assigned
    (e.g., UUIDs) because producers cannot know broker offsets at publish time.
    """
    causes: list[str] = []
    parent_id = value.get("parent_id")
    if isinstance(parent_id, str) and parent_id:
        causes.append(parent_id)

    caused_by = value.get("caused_by")
    if isinstance(caused_by, str) and caused_by:
        causes.append(caused_by)
    elif isinstance(caused_by, list):
        causes.extend(c for c in caused_by if isinstance(c, str) and c)

    # Preserve order, drop duplicates
    return list(dict.fromkeys(causes))


class EmbeddingConsumer:
    """Kafka consumer that generates embeddings for messages.

    Modes:
        - sync: Process one message at a time (simple, lower throughput)
        - batch: Batch messages before embedding (one API call per batch)
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
        self.max_retries = settings.consumer_max_retries
        self.retry_backoff_s = settings.consumer_retry_backoff_ms / 1000

        self.provider = get_provider()
        self._consumer: KafkaConsumer | None = None
        self._running = False
        self._topic_id_cache: dict[str, int] = {}
        self._stats = {"processed": 0, "errors": 0, "batches": 0, "edges": 0}

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

    async def _poll(self, timeout_ms: int, max_records: int) -> dict:
        """Poll Kafka without blocking the event loop.

        kafka-python is synchronous; run it in a worker thread so the MCP
        server can keep serving when both run in one process (`hippocampus both`).
        """
        assert self._consumer is not None
        return await asyncio.to_thread(
            self._consumer.poll, timeout_ms=timeout_ms, max_records=max_records
        )

    async def _commit(self):
        """Commit Kafka offsets without blocking the event loop."""
        assert self._consumer is not None
        await asyncio.to_thread(self._consumer.commit)

    async def start(self, mode: str = "batch"):
        """Start the consumer loop.

        Args:
            mode: "sync" for one-at-a-time, "batch" for batched processing
        """
        self._consumer = self._create_consumer()
        self._running = True

        watermark = await get_consumer_offset(self.consumer_id)
        logger.info(
            f"Starting consumer '{self.consumer_id}' in {mode} mode, "
            f"pattern='{self.topic_pattern}', embedding watermark={watermark}"
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

    async def _flush_with_retry(self, batch: list[tuple[ConsumerRecord, dict]]) -> bool:
        """Process a batch, retrying with backoff; commit only on success.

        Returns True if the batch was stored and committed. On persistent
        failure, stops the consumer and returns False — offsets stay
        uncommitted so the messages are redelivered on restart.
        """
        for attempt in range(self.max_retries + 1):
            if await self._process_batch(batch):
                await self._commit()
                self._stats["batches"] += 1
                self._stats["processed"] += len(batch)
                return True
            if attempt < self.max_retries:
                delay = self.retry_backoff_s * (2**attempt)
                logger.warning(
                    f"Batch of {len(batch)} failed (attempt {attempt + 1}/"
                    f"{self.max_retries + 1}), retrying in {delay:.1f}s"
                )
                await asyncio.sleep(delay)

        logger.critical(
            f"Batch of {len(batch)} failed after {self.max_retries + 1} attempts; "
            "stopping consumer without committing (messages will be redelivered)"
        )
        self.stop()
        return False

    async def _sync_loop(self):
        """Process messages one at a time (simple mode)."""
        while self._running:
            # Poll with short timeout to allow stop signal
            records = await self._poll(timeout_ms=1000, max_records=1)

            for _topic_partition, messages in records.items():
                for msg in messages:
                    if not msg.value:
                        continue
                    if not await self._flush_with_retry([(msg, msg.value)]):
                        return

    async def _batch_loop(self):
        """Process messages in batches (efficient mode)."""
        batch: list[tuple[ConsumerRecord, dict]] = []
        last_flush = time.monotonic()

        while self._running:
            # Poll for messages
            records = await self._poll(timeout_ms=100, max_records=self.batch_size)

            # Collect messages
            for _topic_partition, messages in records.items():
                for msg in messages:
                    if msg.value:
                        batch.append((msg, msg.value))

            # Check if we should flush
            elapsed_ms = (time.monotonic() - last_flush) * 1000
            should_flush = len(batch) >= self.batch_size or (
                batch and elapsed_ms > self.batch_timeout_ms
            )

            if should_flush:
                if not await self._flush_with_retry(batch):
                    return
                batch = []
                last_flush = time.monotonic()

        # Flush remaining on shutdown
        if batch:
            await self._flush_with_retry(batch)

    async def _process_single(self, msg: ConsumerRecord):
        """Process a single message (thin wrapper over batch processing)."""
        if not msg.value:
            return
        await self._process_batch([(msg, msg.value)])

    async def _process_batch(self, batch: list[tuple[ConsumerRecord, dict]]) -> bool:
        """Embed and store a batch of messages, plus any declared causal edges.

        Returns True on success, False on failure (nothing committed).
        """
        if not batch:
            return True

        # Format all texts
        texts = [format_for_embedding(value) for _, value in batch]

        # Single API call for all embeddings
        try:
            embeddings = await self.provider.embed(texts)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            self._stats["errors"] += len(batch)
            return False

        try:
            for msg, _ in batch:
                if msg.topic not in self._topic_id_cache:
                    self._topic_id_cache[msg.topic] = await self._get_topic_id(msg.topic)

            records = []
            for (msg, value), embedding, text in zip(batch, embeddings, texts):
                decision_id = value.get("id") if isinstance(value.get("id"), str) else None
                records.append(
                    (
                        self._topic_id_cache[msg.topic],
                        msg.partition,
                        msg.offset,
                        embedding,
                        text,
                        value.get("anchor"),
                        decision_id,
                    )
                )

            _, offsets = await store_embeddings_batch(records)

            await self._store_edges(batch, offsets)

            # Advance the embedding watermark for observability/lag monitoring.
            # (Kafka group offsets — committed by the caller — are the actual
            # resume point; see consumer_state docs.)
            if offsets:
                await update_consumer_offset(self.consumer_id, max(offsets.values()))
        except Exception as e:
            logger.error(f"Storing batch failed: {e}")
            self._stats["errors"] += len(batch)
            return False

        logger.debug(f"Processed batch of {len(batch)} messages")
        return True

    async def _store_edges(
        self, batch: list[tuple[ConsumerRecord, dict]], batch_offsets: dict[str, int]
    ):
        """Record causal edges declared in payloads.

        Cause IDs are resolved to global offsets first within the current
        batch, then against previously embedded decisions. Unresolvable causes
        (unknown ID, or parent not yet embedded) are logged and skipped — a
        missing edge degrades to temporal fallback, it does not block ingest.
        """
        wanted: list[tuple[str, list[str]]] = []  # (effect decision_id, cause ids)
        for _, value in batch:
            effect_id = value.get("id")
            if not isinstance(effect_id, str):
                continue
            causes = extract_cause_ids(value)
            if causes:
                wanted.append((effect_id, causes))

        if not wanted:
            return

        unresolved = {
            cause for _, causes in wanted for cause in causes if cause not in batch_offsets
        }
        resolved = dict(batch_offsets)
        resolved.update(await resolve_decision_offsets(list(unresolved)))

        edges = []
        for effect_id, causes in wanted:
            effect_offset = resolved.get(effect_id)
            if effect_offset is None:
                # Effect row was a duplicate (already embedded) — edges for it
                # were recorded when it was first processed.
                continue
            for cause in causes:
                cause_offset = resolved.get(cause)
                if cause_offset is None:
                    logger.warning(
                        f"Cannot resolve causal parent '{cause}' for decision "
                        f"'{effect_id}'; skipping edge"
                    )
                    continue
                edges.append((effect_offset, cause_offset, "caused_by"))

        if edges:
            self._stats["edges"] += await store_causal_edges(edges)

    async def _get_topic_id(self, topic_name: str) -> int:
        """Look up topic ID from pg_kafka's kafka.topics table."""
        from .db import get_pool

        pool = await get_pool()
        row = await pool.fetchrow("SELECT id FROM kafka.topics WHERE name = $1", topic_name)
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
