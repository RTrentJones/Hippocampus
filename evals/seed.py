"""Seed the database with a generated eval dataset.

Inserts messages directly into pg_kafka's tables (the same write path the
broker uses) and runs them through the same storage path the consumer uses:
embeddings with content/anchor/decision_id, plus causal edges resolved from
declared parent_id links.
"""

import json
import logging

from hippocampus.consumer import extract_cause_ids
from hippocampus.db import (
    get_pool,
    resolve_decision_offsets,
    store_causal_edges,
    store_embeddings_batch,
)
from hippocampus.embeddings import EmbeddingProvider, format_for_embedding

from .dataset import Session

logger = logging.getLogger(__name__)

EVAL_TOPIC_PREFIX = "decisions.eval."


async def clean_eval_data() -> None:
    """Remove data from previous eval runs (eval topics only)."""
    pool = await get_pool()
    await pool.execute(
        """
        DELETE FROM hippocampus.causal_edges ce
        USING kafka.messages m, kafka.topics t
        WHERE m.global_offset = ce.effect_global_offset
          AND t.id = m.topic_id AND t.name LIKE $1
        """,
        EVAL_TOPIC_PREFIX + "%",
    )
    await pool.execute(
        """
        DELETE FROM hippocampus.embeddings e
        USING kafka.topics t
        WHERE t.id = e.topic_id AND t.name LIKE $1
        """,
        EVAL_TOPIC_PREFIX + "%",
    )
    await pool.execute(
        """
        DELETE FROM kafka.messages m
        USING kafka.topics t
        WHERE t.id = m.topic_id AND t.name LIKE $1
        """,
        EVAL_TOPIC_PREFIX + "%",
    )
    await pool.execute("DELETE FROM kafka.topics WHERE name LIKE $1", EVAL_TOPIC_PREFIX + "%")


async def seed_session(session: Session, provider: EmbeddingProvider) -> dict[str, int]:
    """Insert one session's decisions and return decision_id -> global_offset."""
    pool = await get_pool()
    topic_name = EVAL_TOPIC_PREFIX + session.name

    # pg_kafka owns kafka.topics; don't assume a unique constraint exists
    topic_id = await pool.fetchval("SELECT id FROM kafka.topics WHERE name = $1", topic_name)
    if topic_id is None:
        topic_id = await pool.fetchval(
            "INSERT INTO kafka.topics (name) VALUES ($1) RETURNING id", topic_name
        )

    payloads = [decision.payload() for decision in session.decisions]
    for offset, payload in enumerate(payloads):
        await pool.execute(
            """
            INSERT INTO kafka.messages (topic_id, partition_id, partition_offset, value)
            VALUES ($1, 0, $2, $3)
            ON CONFLICT DO NOTHING
            """,
            topic_id,
            offset,
            json.dumps(payload).encode("utf-8"),
        )

    # Same storage path as the consumer: embed, store with metadata, link causes
    texts = [format_for_embedding(payload) for payload in payloads]
    embeddings = await provider.embed(texts)

    records = [
        (topic_id, 0, offset, embedding, text, payload.get("anchor"), payload["id"])
        for offset, (payload, embedding, text) in enumerate(zip(payloads, embeddings, texts))
    ]
    _, offsets = await store_embeddings_batch(records)

    edges = []
    for payload in payloads:
        for cause in extract_cause_ids(payload):
            effect_offset = offsets.get(payload["id"])
            cause_offset = offsets.get(cause)
            if cause_offset is None:
                resolved = await resolve_decision_offsets([cause])
                cause_offset = resolved.get(cause)
            if effect_offset is not None and cause_offset is not None:
                edges.append((effect_offset, cause_offset, "caused_by"))
    await store_causal_edges(edges)

    return offsets


async def seed_dataset(sessions: list[Session], provider: EmbeddingProvider) -> dict[str, int]:
    """Seed all sessions; returns a combined decision_id -> global_offset map."""
    offsets: dict[str, int] = {}
    for session in sessions:
        offsets.update(await seed_session(session, provider))
    logger.info(f"Seeded {len(sessions)} sessions, {len(offsets)} decisions")
    return offsets
