# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status

**Learning/exploration project** — not production-ready

Hippocampus demonstrates minimal infrastructure patterns for AI agent memory. It prioritizes simplicity over scale, using only Postgres extensions where production systems might use dedicated services.

## Project Overview

Hippocampus is a Temporal RAG (Retrieval Augmented Generation) layer for AI agents, built on top of [pg_kafka](https://github.com/RTrentJones/pg_kafka).

**Problem it solves:** Standard RAG returns disconnected "bags of chunks" based on semantic similarity. When debugging AI agents, you need to know *what happened before* an event—the causal chain, not just similar events.

**Solution:** Combine semantic search (find an anchor point) with temporal traversal (walk backward through the event log).

## Architecture

```
Agent (Cline, Claude Code, etc.)
    │
    │ Kafka produce (decisions.{agent}.{session})
    ▼
┌─────────────────────────────────────────────────────┐
│  Postgres with pg_kafka + pgvector                   │
│                                                      │
│  kafka.messages (event log)                          │
│       ↓                                              │
│  hippocampus.embeddings (vector layer)              │
│       ↓                                              │
│  MCP Server (semantic + temporal tools)             │
└─────────────────────────────────────────────────────┘
```

### Key Architectural Patterns

**Resume point vs. watermark:** Kafka consumer-group offsets (committed only after a batch is fully stored) are the actual resume point after a restart. `hippocampus.consumer_state` tracks `last_global_offset` as an *embedding watermark* — the highest globally-ordered message that has been embedded — used for observability and lag/backfill checks, not for resuming.

**Failure semantics (fail-stop):** Offsets are committed only after a batch is embedded and stored. A batch that keeps failing after `CONSUMER_MAX_RETRIES` attempts (exponential backoff) stops the consumer rather than being skipped, so messages are never committed past without being processed; they are redelivered on restart. Inserts are idempotent (`ON CONFLICT DO NOTHING`), so redelivery is safe.

**Composite key join:** The embeddings table uses `(topic_id, partition_id, partition_offset)` as its primary key, matching `kafka.messages` row identity for efficient joins while preserving Kafka's partitioning semantics. (It is a join key, not an enforced FOREIGN KEY constraint — pg_kafka owns that table.) `global_offset` is denormalized onto embeddings at insert time so causal-edge and watermark queries skip the join.

**Causal edges:** Producers declare causality via `parent_id` / `caused_by` payload fields holding producer-assigned decision IDs (producers can't know broker offsets at publish time). The consumer resolves IDs to global offsets — first within the batch, then against previously embedded decisions — and stores edges in `hippocampus.causal_edges`. `replay_causal_chain` walks edges with a recursive CTE and falls back to a temporal window when the anchor has no edges; results are tagged with `retrieval: causal_graph | temporal_window`.

**Hybrid retrieval:** `find_similar` defaults to hybrid mode: pgvector cosine ranking fused with Postgres full-text search (`content_tsv` over the embedded text) using Reciprocal Rank Fusion. `mode: "semantic"` gives vector-only ranking.

**Consumer Modes:** The embedding consumer supports two modes:
- `sync`: Process one message at a time (simple, lower throughput)
- `batch`: Batch messages before sending to embedding API (one API call per batch)

The `embedded_messages` view provides a convenient join between `kafka.messages` and `hippocampus.embeddings`, including the topic name for easier querying.

## Development Setup

```bash
# Install package in editable mode with dev dependencies
pip install -e ".[dev]"

# Install with local embeddings support (sentence-transformers)
pip install -e ".[dev,local]"

# Copy environment template
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Run migrations in order (requires Postgres with pg_kafka + pgvector extensions)
psql -f migrations/001_embeddings.sql
psql -f migrations/002_causal_edges_hybrid_search.sql
```

## Common Commands

```bash
# Run MCP server (for Claude Desktop integration)
hippocampus server

# Run embedding consumer (batch mode, default)
hippocampus consumer

# Run consumer in sync mode (simpler, slower)
hippocampus consumer --mode sync

# Run both consumer and server (dev mode)
hippocampus both

# Linting
ruff check .

# Auto-fix linting issues
ruff check --fix .

# Format code
ruff format .

# Type checking
mypy

# Run all tests
pytest

# Run specific test file
pytest tests/test_embeddings.py

# Run specific test function
pytest tests/test_embeddings.py::test_openai_provider

# Run tests with verbose output
pytest -v

# Run tests with print statements visible
pytest -s
```

## Testing Infrastructure

### Build Test Database Image

The test suite requires PostgreSQL 17 with pg_kafka and pgvector extensions. We provide a custom Docker image for reproducible testing.

```bash
# Build custom test image (takes 5-10 minutes on first build)
./scripts/build-test-image.sh

# Verify image works correctly
./scripts/verify-test-image.sh
```

### Run Tests

```bash
# Start test environment
docker-compose -f docker-compose.test.yml up -d

# Wait for healthy status (may take up to 50 seconds for extension loading)
docker-compose -f docker-compose.test.yml ps

# Run all tests
pytest

# Run specific test suites
pytest tests/integration/ -v  # Integration tests (requires database)
pytest tests/e2e/ -v          # E2E tests (requires database)
pytest tests/unit/ -v         # Unit tests (no database required)

# Run with coverage
pytest --cov=hippocampus --cov-report=html

# Stop environment
docker-compose -f docker-compose.test.yml down -v  # -v removes volumes
```

### Test Environment Details

- **PostgreSQL:** 17 with pg_kafka + pgvector extensions
- **Port:** 5433 (avoids conflict with dev environment on 5432)
- **Kafka Protocol:** Port 9093 (pg_kafka embedded broker)
- **Database:** `hippocampus_test`
- **Connection:** `postgresql://postgres:postgres@localhost:5433/hippocampus_test`

### Rebuild Image

Rebuild when pg_kafka updates or dependencies change:

```bash
# Full rebuild without cache
./scripts/build-test-image.sh --no-cache
```

### Troubleshooting

**Container won't start or healthcheck fails:**
- Check logs: `docker-compose -f docker-compose.test.yml logs postgres-test`
- Extensions take 5-10 seconds to load on first startup
- Healthcheck allows up to 10 retries (50 seconds total)

**Tests still skipping:**
- Verify container is healthy: `docker-compose -f docker-compose.test.yml ps`
- Test database connection: `docker-compose -f docker-compose.test.yml exec postgres-test psql -U postgres -c '\dx'`
- Should see `pg_kafka` and `vector` extensions listed

### Running Tests from Devcontainer

When running tests from inside the devcontainer, the test database is accessible via container name:

```bash
# Set the environment variable for devcontainer testing
export TEST_DATABASE_URL=postgresql://postgres:postgres@hippocampus-test-db:5432/hippocampus_test

# Or source the .env.test file
source .env.test && pytest tests/
```

**Network Configuration:** The test container automatically joins both `hippocampus-test-network` and `hippocampus_devcontainer_default` to allow connections from inside the devcontainer.

**Quick Start (Devcontainer):**
```bash
# 1. Start the test database
docker-compose -f docker-compose.test.yml up -d

# 2. Wait for healthy status (~15 seconds)
docker-compose -f docker-compose.test.yml ps

# 3. Run tests with correct database URL
TEST_DATABASE_URL=postgresql://postgres:postgres@hippocampus-test-db:5432/hippocampus_test pytest tests/
```

## Kafka Commands

```bash
# List topics (verify pg_kafka is working)
kcat -L -b localhost:9092

# Produce a test decision
echo '{"context":"test","reasoning":"testing","action":"log"}' | \
  kcat -P -b localhost:9092 -t decisions.test.001

# Consume from a topic
kcat -C -b localhost:9092 -t decisions.test.001

# Watch messages in real-time
kcat -C -b localhost:9092 -t decisions.test.001 -o end
```

## MCP Tools

| Tool | Purpose |
|------|---------|
| `find_similar(query, limit?, mode?)` | Search agent decisions: `hybrid` (default, vector + keyword RRF) or `semantic` (vector only) |
| `replay_causal_chain(anchor_offset, lookback?)` | Walk causal edges backward from an anchor; temporal-window fallback when no edges exist |
| `replay_topic(topic_name, from_offset?, limit?)` | View single agent's decision history |
| `temporal_context(global_offset, window?)` | See all agents at a point in time |
| `what_touched(anchor, limit?)` | Find decisions affecting a file (trigram-indexed) |

## Configuration

Environment variables (or `.env` file):

```bash
# Required
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
OPENAI_API_KEY=sk-...  # Required if EMBEDDING_PROVIDER=openai

# Optional
EMBEDDING_PROVIDER=openai  # or "local" for sentence-transformers
EMBEDDING_MODEL=text-embedding-3-small
LOCAL_MODEL_NAME=all-MiniLM-L6-v2

# Consumer settings
KAFKA_CONSUMER_GROUP=hippocampus
KAFKA_TOPIC_PATTERN=decisions.*
BATCH_SIZE=100
BATCH_TIMEOUT_MS=500
CONSUMER_MAX_RETRIES=5        # retries before fail-stop on a failing batch
CONSUMER_RETRY_BACKOFF_MS=1000
```

## Key Files

- [src/hippocampus/embeddings.py](src/hippocampus/embeddings.py) - Swappable embedding providers (OpenAI, local, mock)
- [src/hippocampus/consumer.py](src/hippocampus/consumer.py) - Kafka consumer: sync/batch modes, causal-edge extraction, fail-stop retries
- [src/hippocampus/mcp_server.py](src/hippocampus/mcp_server.py) - MCP tools for AI agent queries
- [src/hippocampus/db.py](src/hippocampus/db.py) - Database queries (hybrid search, causal/temporal traversal)
- [src/hippocampus/main.py](src/hippocampus/main.py) - CLI entry point (with graceful shutdown)
- [migrations/001_embeddings.sql](migrations/001_embeddings.sql) - Schema (embeddings table + consumer state)
- [migrations/002_causal_edges_hybrid_search.sql](migrations/002_causal_edges_hybrid_search.sql) - Causal edges, full-text/trigram indexes
- [.github/workflows/ci.yml](.github/workflows/ci.yml) - CI: lint, mypy, unit, then integration/e2e against the Docker test database

## Dependencies

- **pg_kafka**: Kafka-compatible message broker in Postgres
- **pgvector**: Vector similarity search extension
- **MCP SDK**: Model Context Protocol for AI tool integration
- **OpenAI API**: Embeddings (swappable for local models via sentence-transformers)
- **Hatch**: Build/packaging (configured via pyproject.toml)
- **Ruff**: Fast Python linter and formatter
- **pytest**: Testing framework with asyncio support

## Scaling Roadmap

This project is intentionally minimal. When you outgrow it:

### Storage Graduation

| Current | Future | When to Graduate |
|---------|--------|------------------|
| pgvector | Pinecone/Weaviate | >1M vectors, need managed service |
| Postgres tables | Neo4j | Complex graph traversals, relationship queries |
| Single Postgres | Read replicas | Query latency requirements |

### Messaging Graduation

| Current | Future | When to Graduate |
|---------|--------|------------------|
| pg_kafka | Real Kafka | >10k msg/sec, multi-datacenter |
| Single topic pattern | Topic partitioning | Need parallel consumers |

### Why This Design Scales

The MCP tool interface is stable. Graduation means swapping implementations:

- `find_similar()` → calls pgvector today, Pinecone tomorrow
- `replay_causal_chain()` → SQL recursive CTE today, Neo4j Cypher tomorrow
- Consumer → reads from pg_kafka today, real Kafka tomorrow

Same tools, same agent integration, different backends.
