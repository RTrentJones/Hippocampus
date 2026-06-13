-- Hippocampus: Temporal RAG for AI Agents
-- Migration 001: Embeddings table
--
-- Prerequisites:
--   CREATE EXTENSION IF NOT EXISTS pg_kafka;
--   CREATE EXTENSION IF NOT EXISTS vector;

CREATE SCHEMA IF NOT EXISTS hippocampus;

-- Embeddings table - FK to pg_kafka's messages
-- Uses composite key to join with kafka.messages
CREATE TABLE hippocampus.embeddings (
    topic_id INT NOT NULL,
    partition_id INT NOT NULL,
    partition_offset BIGINT NOT NULL,
    embedding vector(1536) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (topic_id, partition_id, partition_offset)
);

-- Semantic search: exact scans at this scale (no ANN index here on purpose).
-- ivfflat trains its centroids at CREATE INDEX time, so an index built on an
-- empty table can MISS inserted rows entirely at low probe counts (measured:
-- a stored row invisible to `ORDER BY embedding <=> $1 LIMIT k`). When data
-- volume outgrows exact scans, build the index AFTER loading data — prefer
-- HNSW, which builds incrementally and has no empty-table training problem:
--   CREATE INDEX idx_embeddings_vector ON hippocampus.embeddings
--       USING hnsw (embedding vector_cosine_ops);

-- Consumer state tracking
-- Tracks where the embedding consumer left off
CREATE TABLE hippocampus.consumer_state (
    consumer_id TEXT PRIMARY KEY,
    last_global_offset BIGINT NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Helper view: Join embeddings with message content
CREATE VIEW hippocampus.embedded_messages AS
SELECT 
    m.topic_id,
    m.partition_id,
    m.partition_offset,
    m.global_offset,
    m.key,
    m.value,
    m.headers,
    m.created_at AS message_created_at,
    e.embedding,
    e.created_at AS embedded_at,
    t.name AS topic_name
FROM kafka.messages m
JOIN hippocampus.embeddings e 
    ON e.topic_id = m.topic_id 
    AND e.partition_id = m.partition_id 
    AND e.partition_offset = m.partition_offset
JOIN kafka.topics t ON m.topic_id = t.id;

COMMENT ON TABLE hippocampus.embeddings IS 
    'Vector embeddings for kafka.messages, enabling semantic search';
COMMENT ON TABLE hippocampus.consumer_state IS 
    'Tracks embedding generation progress for consumers';
COMMENT ON VIEW hippocampus.embedded_messages IS 
    'Convenience view joining messages with their embeddings';
