"""Database connection and queries for Hippocampus."""

import json

import asyncpg
from pgvector.asyncpg import register_vector

from .config import settings


def _deserialize_row(row: asyncpg.Record) -> dict:
    """Convert a database row to dict, deserializing bytea 'value' field if present."""
    result = dict(row)
    if "value" in result and isinstance(result["value"], bytes):
        result["value"] = json.loads(result["value"].decode("utf-8"))
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


async def store_embedding(
    topic_id: int,
    partition_id: int,
    partition_offset: int,
    embedding: list[float],
) -> bool:
    """Store an embedding for a message.

    Returns True if inserted, False if already exists.
    """
    pool = await get_pool()
    try:
        await pool.execute(
            """
            INSERT INTO hippocampus.embeddings
                (topic_id, partition_id, partition_offset, embedding)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT DO NOTHING
            """,
            topic_id,
            partition_id,
            partition_offset,
            embedding,
        )
        return True
    except Exception:
        return False


async def store_embeddings_batch(
    records: list[tuple[int, int, int, list[float]]],
) -> int:
    """Store multiple embeddings in a batch.

    Args:
        records: List of (topic_id, partition_id, partition_offset, embedding) tuples

    Returns:
        Number of records inserted
    """
    if not records:
        return 0

    pool = await get_pool()
    # Use copy for bulk insert (faster than executemany)
    result = await pool.executemany(
        """
        INSERT INTO hippocampus.embeddings 
            (topic_id, partition_id, partition_offset, embedding)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT DO NOTHING
        """,
        records,
    )
    return len(records)


# =============================================================================
# Consumer State
# =============================================================================


async def get_consumer_offset(consumer_id: str) -> int:
    """Get the last processed global_offset for a consumer."""
    pool = await get_pool()
    row = await pool.fetchrow(
        "SELECT last_global_offset FROM hippocampus.consumer_state WHERE consumer_id = $1",
        consumer_id,
    )
    return row["last_global_offset"] if row else 0


async def update_consumer_offset(consumer_id: str, global_offset: int):
    """Update the consumer's last processed offset."""
    pool = await get_pool()
    await pool.execute(
        """
        INSERT INTO hippocampus.consumer_state (consumer_id, last_global_offset, updated_at)
        VALUES ($1, $2, NOW())
        ON CONFLICT (consumer_id) DO UPDATE 
        SET last_global_offset = $2, updated_at = NOW()
        """,
        consumer_id,
        global_offset,
    )


# =============================================================================
# Semantic Search (MCP Tools)
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

    if topic_pattern:
        rows = await pool.fetch(
            """
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
            WHERE t.name LIKE $3
            ORDER BY e.embedding <=> $1
            LIMIT $2
            """,
            query_embedding,
            limit,
            topic_pattern,
        )
    else:
        rows = await pool.fetch(
            """
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
            ORDER BY e.embedding <=> $1
            LIMIT $2
            """,
            query_embedding,
            limit,
        )

    return [_deserialize_row(row) for row in rows]


# =============================================================================
# Temporal Traversal (MCP Tools)
# =============================================================================


async def replay_causal_chain(
    anchor_offset: int,
    lookback: int = 10,
    topic_pattern: str | None = None,
) -> list[dict]:
    """Replay events leading up to an anchor point.

    Uses pg_kafka's global_offset for strict temporal ordering.

    Args:
        anchor_offset: The global_offset to walk back from
        lookback: Number of events to retrieve
        topic_pattern: Optional LIKE pattern to filter topics

    Returns:
        List of events in chronological order (oldest first)
    """
    pool = await get_pool()

    if topic_pattern:
        rows = await pool.fetch(
            """
            SELECT 
                m.value,
                t.name AS topic,
                m.global_offset,
                m.created_at
            FROM kafka.messages m
            JOIN kafka.topics t ON m.topic_id = t.id
            WHERE m.global_offset <= $1
              AND t.name LIKE $3
            ORDER BY m.global_offset DESC
            LIMIT $2
            """,
            anchor_offset,
            lookback,
            topic_pattern,
        )
    else:
        rows = await pool.fetch(
            """
            SELECT 
                m.value,
                t.name AS topic,
                m.global_offset,
                m.created_at
            FROM kafka.messages m
            JOIN kafka.topics t ON m.topic_id = t.id
            WHERE m.global_offset <= $1
            ORDER BY m.global_offset DESC
            LIMIT $2
            """,
            anchor_offset,
            lookback,
        )

    # Return in chronological order (oldest first)
    return [_deserialize_row(row) for row in reversed(rows)]


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

    Searches the 'anchor' field in decision payloads.

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
        FROM kafka.messages m
        JOIN kafka.topics t ON m.topic_id = t.id
        WHERE (convert_from(m.value, 'UTF8')::jsonb)->>'anchor' LIKE $1
        ORDER BY m.global_offset DESC
        LIMIT $2
        """,
        f"%{anchor}%",
        limit,
    )
    return [_deserialize_row(row) for row in rows]
