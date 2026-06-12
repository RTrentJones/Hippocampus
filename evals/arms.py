"""Retrieval arms compared by the eval.

Every arm answers the same question — "what led to this failure?" — and
returns a ranked list of decision IDs. Arms differ in what information they
use:

    semantic         vector similarity to the query, nothing else (baseline RAG)
    hybrid           vector + keyword RRF (the find_similar default)
    temporal         the k events immediately before the failure (needs the anchor)
    causal           walk explicit causal edges back from the failure (needs the anchor)
    anchor_causal    full pipeline: hybrid search finds the anchor, then walk edges
                     (no ground-truth anchor given — this is the README thesis)
"""

import json

from hippocampus.db import (
    find_similar,
    get_causal_chain,
    get_pool,
    hybrid_search,
)
from hippocampus.embeddings import EmbeddingProvider

from .seed import EVAL_TOPIC_PREFIX

TOPIC_PATTERN = EVAL_TOPIC_PREFIX + "%"


def _ids(rows: list[dict]) -> list[str]:
    """Extract decision IDs from query results, dropping rows without one."""
    ids = []
    for row in rows:
        value = row.get("value") or {}
        decision_id = value.get("id")
        if isinstance(decision_id, str):
            ids.append(decision_id)
    return ids


async def semantic_arm(query: str, provider: EmbeddingProvider, k: int) -> list[str]:
    embedding = await provider.embed_one(query)
    rows = await find_similar(embedding, limit=k, topic_pattern=TOPIC_PATTERN)
    return _ids(rows)


async def hybrid_arm(query: str, provider: EmbeddingProvider, k: int) -> list[str]:
    embedding = await provider.embed_one(query)
    rows = await hybrid_search(query, embedding, limit=k, topic_pattern=TOPIC_PATTERN)
    return _ids(rows)


async def temporal_arm(anchor_offset: int, k: int) -> list[str]:
    """The k events at or before the anchor, most recent first."""
    pool = await get_pool()
    rows = await pool.fetch(
        """
        SELECT m.value
        FROM kafka.messages m
        JOIN kafka.topics t ON m.topic_id = t.id
        WHERE m.global_offset <= $1 AND t.name LIKE $3
        ORDER BY m.global_offset DESC
        LIMIT $2
        """,
        anchor_offset,
        k,
        TOPIC_PATTERN,
    )
    ids = []
    for row in rows:
        value = json.loads(row["value"].decode("utf-8"))
        if isinstance(value.get("id"), str):
            ids.append(value["id"])
    return ids


async def causal_arm(anchor_offset: int, k: int) -> list[str]:
    """Walk causal edges back from the anchor; anchor first, then nearest causes."""
    chain = await get_causal_chain(anchor_offset, max_depth=k)
    # get_causal_chain returns chronological order (root first); rank with the
    # anchor and its nearest causes first, the way a debugger walks backward.
    return list(reversed(_ids(chain)))[:k]


async def anchor_causal_arm(query: str, provider: EmbeddingProvider, k: int) -> list[str]:
    """Full pipeline: hybrid search to find the anchor, then causal walk.

    This arm gets no ground truth at all — it must locate the failure event
    from the query before it can traverse, exactly as an agent would.
    """
    embedding = await provider.embed_one(query)
    rows = await hybrid_search(query, embedding, limit=1, topic_pattern=TOPIC_PATTERN)
    if not rows:
        return []
    anchor_offset = rows[0]["global_offset"]
    return await causal_arm(anchor_offset, k)
