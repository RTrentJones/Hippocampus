-- Hippocampus: Temporal RAG for AI Agents
-- Migration 002: Causal edges + hybrid search support
--
-- Prerequisites:
--   migrations/001_embeddings.sql
--   CREATE EXTENSION IF NOT EXISTS pg_trgm;  (created below; requires superuser)

CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Denormalized columns on embeddings:
--   global_offset: resolved from kafka.messages at insert time so causal-edge
--                  and watermark queries never need the composite-key join
--   content_text:  the exact text that was embedded, enabling full-text search
--   anchor:        file/location reference extracted from the payload
--   decision_id:   producer-assigned ID (producers cannot know broker offsets
--                  at publish time, so causality is declared via these IDs)
ALTER TABLE hippocampus.embeddings
    ADD COLUMN global_offset BIGINT,
    ADD COLUMN content_text TEXT,
    ADD COLUMN anchor TEXT,
    ADD COLUMN decision_id TEXT;

ALTER TABLE hippocampus.embeddings
    ADD COLUMN content_tsv tsvector
        GENERATED ALWAYS AS (to_tsvector('english', coalesce(content_text, ''))) STORED;

-- Backfill global_offset for rows embedded before this migration
UPDATE hippocampus.embeddings e
SET global_offset = m.global_offset
FROM kafka.messages m
WHERE m.topic_id = e.topic_id
  AND m.partition_id = e.partition_id
  AND m.partition_offset = e.partition_offset
  AND e.global_offset IS NULL;

-- Full-text search (keyword half of hybrid retrieval)
CREATE INDEX idx_embeddings_tsv ON hippocampus.embeddings USING gin (content_tsv);

-- Trigram index so what_touched's substring match (ILIKE '%...%') is indexed
CREATE INDEX idx_embeddings_anchor_trgm ON hippocampus.embeddings
    USING gin (anchor gin_trgm_ops);

-- Causal-edge resolution: decision_id -> global_offset
CREATE INDEX idx_embeddings_decision_id ON hippocampus.embeddings (decision_id)
    WHERE decision_id IS NOT NULL;

-- Explicit causal links between messages, declared by producers via
-- `parent_id` / `caused_by` payload fields and resolved by the consumer.
-- replay_causal_chain walks these edges transitively (effect -> cause);
-- when a message has no edges it falls back to a temporal window.
CREATE TABLE hippocampus.causal_edges (
    effect_global_offset BIGINT NOT NULL,
    cause_global_offset BIGINT NOT NULL,
    relation TEXT NOT NULL DEFAULT 'caused_by',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (effect_global_offset, cause_global_offset)
);

-- Forward traversal (what did this decision cause?)
CREATE INDEX idx_causal_edges_cause ON hippocampus.causal_edges (cause_global_offset);

-- Extend the convenience view with the new columns (appended to keep
-- CREATE OR REPLACE compatible with the existing column order)
CREATE OR REPLACE VIEW hippocampus.embedded_messages AS
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
    t.name AS topic_name,
    e.content_text,
    e.anchor,
    e.decision_id
FROM kafka.messages m
JOIN hippocampus.embeddings e
    ON e.topic_id = m.topic_id
    AND e.partition_id = m.partition_id
    AND e.partition_offset = m.partition_offset
JOIN kafka.topics t ON m.topic_id = t.id;

COMMENT ON TABLE hippocampus.causal_edges IS
    'Explicit causal links between messages (effect <- cause), by global_offset';
COMMENT ON COLUMN hippocampus.embeddings.content_text IS
    'Text that was embedded; also drives full-text (hybrid) search';
COMMENT ON COLUMN hippocampus.embeddings.decision_id IS
    'Producer-assigned decision ID used to resolve declared causality';
