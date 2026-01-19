-- Initialize extensions before running migrations
-- This must run BEFORE 001_embeddings.sql
CREATE EXTENSION IF NOT EXISTS pg_kafka;
CREATE EXTENSION IF NOT EXISTS vector;
