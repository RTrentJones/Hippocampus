"""Database connection and queries for Hippocampus."""

import json

import asyncpg
from pgvector.asyncpg import register_vector

from .config import settings


def _deserialize_row(row: asyncpg.Record, **extra) -> dict:
    """Convert a database row to dict, deserializing bytea 'value' field if present."""
    result = dict(row)
    if "value" in result and isinstance(result["value"], bytes):
        result["value"] = json.loads(result["value"].decode("utf-8"))
    result.update(extra)
    return result


# Global connection pool
_pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    """Get or create the database connection pool."""
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            settings.database_url,
            min_size=2,
            max_size=10,
            init=_init_connection,
        )
    return _pool


async def _init_connection(conn: asyncpg.Connection):
    """Initialize each connection with pgvector support."""
    await register_vector(conn)


async def close_pool():
    """Close the connection pool."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


# =============================================================================
# Embedding Storage
# =============================================================================

# The scalar subquery resolves the message's global_offset at insert time so
# that causal-edge and watermark queries never need the composite-key join.
# It is NULL when the message row does not (yet) exist in kafka.messages.
_INSERT_EMBEDDING_SQL = """
    INSERT INTO hippocampus.embeddings
        (topic_id, partition_id, partition_offset, global_offset,
         embedding, content_text, anchor, decision_id)
    VALUES (
        $1, $2, $3,
        (SELECT m.global_offset FROM kafka.messages m
         WHERE m.topic_id = $1 AND m.partition_id = $2 AND m.partition_offset = $3),
        $4, $5, $6, $7
    )
    ON CONFLICT DO NOTHING
    RETURNING global_offset
"""


async def store_embedding(
    topic_id: int,
    partition_id: int,
    partition_offset: int,
    embedding: list[float],
    content_text: str | None = None,
    anchor: str | None = None,
    decision_id: str | None = None,
) -> bool:
    """Store an embedding for a message.

    Returns True if a new row was inserted, False if the row already existed.
    Database errors propagate to the caller.
    """
    pool = await get_pool()
    row = await pool.fetchrow(
        _INSERT_EMBEDDING_SQL,
        topic_id,
        partition_id,
        partition_offset,
        embedding,
        content_text,
        anchor,
        decision_id,
    )
    return row is not None


async def store_embeddings_batch(
    records: list[tuple],
) -> tuple[int, dict[str, int]]:
    """Store multiple embeddings in a single transaction.

    Args:
        records: List of tuples:
            (topic_id, partition_id, partition_offset, embedding,
             content_text | None, anchor | None, decision_id | None)
            Short 4-tuples (without text/anchor/decision_id) are accepted.

    Returns:
        (inserted_count, offsets) where offsets maps decision_id ->
        global_offset for every inserted row that had both values resolved.
    """
    if not records:
        return 0, {}

    pool = await get_pool()
    inserted = 0
    offsets: dict[str, int] = {}
    async with pool.acquire() as conn:
        async with conn.transaction():
            stmt = await conn.prepare(_INSERT_EMBEDDING_SQL)
            for record in records:
                padded = tuple(record) + (None,) * (7 - len(record))
                row = await stmt.fetchrow(*padded)
                if row is not None:
                    inserted += 1
                    decision_id = padded[6]
                    if decision_id is not None and row["global_offset"] is not None:
                        offsets[decision_id] = row["global_offset"]
    return inserted, offsets


async def resolve_decision_offsets(decision_ids: list[str]) -> dict[str, int]:
    """Resolve decision IDs to global offsets via previously stored embeddings.

    Returns a mapping for the IDs that could be resolved; unknown IDs are
    silently absent from the result.
    """
    if not decision_ids:
        return {}

    pool = await get_pool()
    rows = await pool.fetch(
        """
        SELECT decision_id, global_offset
        FROM hippocampus.embeddings
        WHERE decision_id = ANY($1::text[])
          AND global_offset IS NOT NULL
        """,
        decision_ids,
    )
    return {row["decision_id"]: row["global_offset"] for row in rows}


# =============================================================================
# Causal Edges
# =============================================================================


async def store_causal_edges(edges: list[tuple[int, int, str]]) -> int:
    """Store causal edges between messages.

    Args:
        edges: List of (effect_global_offset, cause_global_offset, relation)

    Returns:
        Number of edges inserted (duplicates are skipped).
    """
    if not edges:
        return 0

    pool = await get_pool()
    inserted = 0
    async with pool.acquire() as conn:
        async with conn.transaction():
            stmt = await conn.prepare(
                """
                INSERT INTO hippocampus.causal_edges
                    (effect_global_offset, cause_global_offset, relation)
                VALUES ($1, $2, $3)
                ON CONFLICT DO NOTHING
                RETURNING 1
                """
            )
            for effect, cause, relation in edges:
                row = await stmt.fetchrow(effect, cause, relation)
                if row is not None:
                    inserted += 1
    return inserted


# =============================================================================
# Consumer State
# =============================================================================


async def get_consumer_offset(consumer_id: str) -> int:
    """Get the embedding watermark (last embedded global_offset) for a consumer."""
    pool = await get_pool()
    row = await pool.fetchrow(
        "SELECT last_global_offset FROM hippocampus.consumer_state WHERE consumer_id = $1",
        consumer_id,
    )
    return row["last_global_offset"] if row else 0


async def update_consumer_offset(consumer_id: str, global_offset: int):
    """Advance the consumer's embedding watermark (never moves backward)."""
    pool = await get_pool()
    await pool.execute(
        """
        INSERT INTO hippocampus.consumer_state (consumer_id, last_global_offset, updated_at)
        VALUES ($1, $2, NOW())
        ON CONFLICT (consumer_id) DO UPDATE
        SET last_global_offset = GREATEST(
                hippocampus.consumer_state.last_global_offset, EXCLUDED.last_global_offset
            ),
            updated_at = NOW()
        """,
        consumer_id,
        global_offset,
    )


# =============================================================================
# Semantic + Hybrid Search (MCP Tools)
# =============================================================================


async def find_similar(
    query_embedding: list[float],
    limit: int = 10,
    topic_pattern: str | None = None,
) -> list[dict]:
    """Find messages most similar to the query embedding.

    Args:
        query_embedding: The query vector
        limit: Max results to return
        topic_pattern: Optional LIKE pattern to filter topics (e.g., 'decisions.%')

    Returns:
        List of messages with similarity scores
    """
    pool = await get_pool()

    base_sql = """
        SELECT
            m.value,
            t.name AS topic,
            m.global_offset,
            m.created_at,
            e.embedding <=> $1 AS distance
        FROM kafka.messages m
        JOIN kafka.topics t ON m.topic_id = t.id
        JOIN hippocampus.embeddings e
            ON e.topic_id = m.topic_id
            AND e.partition_id = m.partition_id
            AND e.partition_offset = m.partition_offset
        {where}
        ORDER BY e.embedding <=> $1
        LIMIT $2
    """

    if topic_pattern:
        rows = await pool.fetch(
            base_sql.format(where="WHERE t.name LIKE $3"),
            query_embedding,
            limit,
            topic_pattern,
        )
    else:
        rows = await pool.fetch(base_sql.format(where=""), query_embedding, limit)

    return [_deserialize_row(row) for row in rows]


async def hybrid_search(
    query_text: str,
    query_embedding: list[float],
    limit: int = 10,
    topic_pattern: str | None = None,
) -> list[dict]:
    """Hybrid retrieval: vector similarity + full-text search, fused with RRF.

    Reciprocal Rank Fusion combines the two rankings without needing to
    calibrate their incomparable scores: score = sum(1 / (60 + rank)).
    Rows that appear in only one ranking still score via that ranking.

    Args:
        query_text: Natural-language query (used for full-text search)
        query_embedding: Embedding of the same query (used for vector search)
        limit: Max results to return
        topic_pattern: Optional LIKE pattern to filter topics

    Returns:
        List of messages with an `rrf_score` field, best match first
    """
    pool = await get_pool()
    # Pull a larger candidate pool from each ranking before fusing so a result
    # ranked just below `limit` in both lists can still win overall.
    candidates = limit * 4

    sql = """
        WITH semantic AS (
            SELECT e.topic_id, e.partition_id, e.partition_offset,
                   ROW_NUMBER() OVER (ORDER BY e.embedding <=> $1) AS rank
            FROM hippocampus.embeddings e
            ORDER BY e.embedding <=> $1
            LIMIT $3
        ),
        keyword AS (
            SELECT e.topic_id, e.partition_id, e.partition_offset,
                   ROW_NUMBER() OVER (
                       ORDER BY ts_rank_cd(e.content_tsv, websearch_to_tsquery('english', $2)) DESC
                   ) AS rank
            FROM hippocampus.embeddings e
            WHERE e.content_tsv @@ websearch_to_tsquery('english', $2)
            LIMIT $3
        ),
        fused AS (
            SELECT
                COALESCE(s.topic_id, k.topic_id) AS topic_id,
                COALESCE(s.partition_id, k.partition_id) AS partition_id,
                COALESCE(s.partition_offset, k.partition_offset) AS partition_offset,
                COALESCE(1.0 / (60 + s.rank), 0) + COALESCE(1.0 / (60 + k.rank), 0) AS rrf_score
            FROM semantic s
            FULL OUTER JOIN keyword k
                ON s.topic_id = k.topic_id
                AND s.partition_id = k.partition_id
                AND s.partition_offset = k.partition_offset
        )
        SELECT
            m.value,
            t.name AS topic,
            m.global_offset,
            m.created_at,
            f.rrf_score
        FROM fused f
        JOIN kafka.messages m
            ON m.topic_id = f.topic_id
            AND m.partition_id = f.partition_id
            AND m.partition_offset = f.partition_offset
        JOIN kafka.topics t ON m.topic_id = t.id
        {where}
        ORDER BY f.rrf_score DESC
        LIMIT $4
    """

    if topic_pattern:
        rows = await pool.fetch(
            sql.format(where="WHERE t.name LIKE $5"),
            query_embedding,
            query_text,
            candidates,
            limit,
            topic_pattern,
        )
    else:
        rows = await pool.fetch(
            sql.format(where=""), query_embedding, query_text, candidates, limit
        )

    return [_deserialize_row(row) for row in rows]


# =============================================================================
# Temporal + Causal Traversal (MCP Tools)
# =============================================================================


async def get_causal_chain(
    anchor_offset: int,
    max_depth: int = 10,
) -> list[dict]:
    """Walk explicit causal edges backward from an anchor message.

    Follows hippocampus.causal_edges transitively (effect -> cause) up to
    max_depth hops. Returns only the anchor itself when no edges exist.

    Returns:
        Events in chronological order (oldest first), each tagged with
        retrieval='causal_graph' and its hop depth from the anchor.
    """
    pool = await get_pool()
    rows = await pool.fetch(
        """
        WITH RECURSIVE chain AS (
            SELECT $1::bigint AS global_offset, 0 AS depth
            UNION ALL
            SELECT ce.cause_global_offset, chain.depth + 1
            FROM hippocampus.causal_edges ce
            JOIN chain ON ce.effect_global_offset = chain.global_offset
            WHERE chain.depth < $2
        )
        SELECT DISTINCT ON (m.global_offset)
            m.value,
            t.name AS topic,
            m.global_offset,
            m.created_at,
            chain.depth
        FROM chain
        JOIN kafka.messages m ON m.global_offset = chain.global_offset
        JOIN kafka.topics t ON m.topic_id = t.id
        ORDER BY m.global_offset, chain.depth
        """,
        anchor_offset,
        max_depth,
    )
    return [_deserialize_row(row, retrieval="causal_graph") for row in rows]


async def replay_causal_chain(
    anchor_offset: int,
    lookback: int = 10,
    topic_pattern: str | None = None,
) -> list[dict]:
    """Replay events leading up to an anchor point.

    Prefers explicit causal edges (declared via `parent_id`/`caused_by` in
    decision payloads) and falls back to a temporal window over global_offset
    when the anchor has no recorded edges. Each returned event carries a
    `retrieval` field ('causal_graph' or 'temporal_window') so callers know
    which guarantee they got.

    Args:
        anchor_offset: The global_offset to walk back from
        lookback: Max events (graph hops or window size) to retrieve
        topic_pattern: Optional LIKE pattern to filter topics
            (applies to the temporal fallback only; causal edges cross topics)

    Returns:
        List of events in chronological order (oldest first)
    """
    chain = await get_causal_chain(anchor_offset, max_depth=lookback)
    if len(chain) > 1:
        return chain

    pool = await get_pool()

    base_sql = """
        SELECT
            m.value,
            t.name AS topic,
            m.global_offset,
            m.created_at
        FROM kafka.messages m
        JOIN kafka.topics t ON m.topic_id = t.id
        WHERE m.global_offset <= $1
        {extra}
        ORDER BY m.global_offset DESC
        LIMIT $2
    """

    if topic_pattern:
        rows = await pool.fetch(
            base_sql.format(extra="AND t.name LIKE $3"),
            anchor_offset,
            lookback,
            topic_pattern,
        )
    else:
        rows = await pool.fetch(base_sql.format(extra=""), anchor_offset, lookback)

    # Return in chronological order (oldest first)
    return [_deserialize_row(row, retrieval="temporal_window") for row in reversed(rows)]


async def replay_topic(
    topic_name: str,
    from_offset: int = 0,
    limit: int = 100,
) -> list[dict]:
    """Replay all messages from a specific topic.

    Useful for viewing a single agent's decision history.

    Args:
        topic_name: Exact topic name
        from_offset: Starting partition_offset (0 for beginning)
        limit: Max messages to return

    Returns:
        List of messages in chronological order
    """
    pool = await get_pool()
    rows = await pool.fetch(
        """
        SELECT
            m.value,
            m.partition_offset,
            m.global_offset,
            m.created_at
        FROM kafka.messages m
        JOIN kafka.topics t ON m.topic_id = t.id
        WHERE t.name = $1
          AND m.partition_offset >= $2
        ORDER BY m.partition_offset
        LIMIT $3
        """,
        topic_name,
        from_offset,
        limit,
    )
    return [_deserialize_row(row) for row in rows]


async def temporal_context(
    global_offset: int,
    window: int = 10,
) -> list[dict]:
    """Get events around a specific global_offset.

    Shows what was happening across all agents at a point in time.

    Args:
        global_offset: Center point
        window: Events before and after (total = window * 2 + 1)

    Returns:
        List of events in chronological order
    """
    pool = await get_pool()
    rows = await pool.fetch(
        """
        SELECT
            m.value,
            t.name AS topic,
            m.global_offset,
            m.created_at
        FROM kafka.messages m
        JOIN kafka.topics t ON m.topic_id = t.id
        WHERE m.global_offset BETWEEN $1::bigint - $2::int AND $1::bigint + $2::int
        ORDER BY m.global_offset
        """,
        global_offset,
        window,
    )
    return [_deserialize_row(row) for row in rows]


async def what_touched(
    anchor: str,
    limit: int = 20,
) -> list[dict]:
    """Find all decisions that affected a specific file/location.

    Primary path uses the trigram-indexed `anchor` column on
    hippocampus.embeddings (populated by the consumer). Falls back to scanning
    raw message payloads for messages that were never embedded; the fallback
    guards against non-JSON payloads instead of erroring on them.

    Args:
        anchor: File path or location to search for (partial match)
        limit: Max results

    Returns:
        List of decisions (newest first)
    """
    pool = await get_pool()
    rows = await pool.fetch(
        """
        SELECT
            m.value,
            t.name AS topic,
            m.global_offset,
            m.created_at
        FROM hippocampus.embeddings e
        JOIN kafka.messages m
            ON m.topic_id = e.topic_id
            AND m.partition_id = e.partition_id
            AND m.partition_offset = e.partition_offset
        JOIN kafka.topics t ON m.topic_id = t.id
        WHERE e.anchor ILIKE '%' || $1 || '%'
        ORDER BY m.global_offset DESC
        LIMIT $2
        """,
        anchor,
        limit,
    )
    if rows:
        return [_deserialize_row(row) for row in rows]

    # Fallback: payload scan for messages without embeddings (e.g., before the
    # consumer caught up). Sequential scan; bounded by LIMIT.
    rows = await pool.fetch(
        """
        SELECT
            m.value,
            t.name AS topic,
            m.global_offset,
            m.created_at
        FROM kafka.messages m
        JOIN kafka.topics t ON m.topic_id = t.id
        WHERE pg_input_is_valid(convert_from(m.value, 'UTF8'), 'jsonb')
          AND (convert_from(m.value, 'UTF8')::jsonb)->>'anchor' LIKE $1
        ORDER BY m.global_offset DESC
        LIMIT $2
        """,
        f"%{anchor}%",
        limit,
    )
    return [_deserialize_row(row) for row in rows]
