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

**Consumer State Tracking:** The embedding consumer uses `hippocampus.consumer_state` to track `last_global_offset`, enabling resumable processing. When the consumer restarts, it queries this table to determine where to resume, ensuring no messages are skipped or double-processed.

**Composite Key FK:** The embeddings table uses `(topic_id, partition_id, partition_offset)` as a composite FK to `kafka.messages`. This enables efficient joins while preserving Kafka's partitioning semantics.

**Consumer Modes:** The embedding consumer supports two modes:
- `sync`: Process one message at a time (simple, ~100 msg/sec)
- `batch`: Batch messages before sending to embedding API (efficient, ~1000 msg/sec)

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

# Run migrations (requires Postgres with pg_kafka + pgvector extensions)
psql -f migrations/001_embeddings.sql
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
| `find_similar(query, limit?)` | Semantic search across all agent decisions |
| `replay_causal_chain(anchor_offset, lookback?)` | Walk backward from an anchor point |
| `replay_topic(topic_name, from_offset?, limit?)` | View single agent's decision history |
| `temporal_context(global_offset, window?)` | See all agents at a point in time |
| `what_touched(anchor, limit?)` | Find decisions affecting a file |

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
```

## Key Files

- [src/hippocampus/embeddings.py](src/hippocampus/embeddings.py) - Swappable embedding providers (OpenAI, local, mock)
- [src/hippocampus/consumer.py](src/hippocampus/consumer.py) - Kafka consumer with sync/batch modes
- [src/hippocampus/mcp_server.py](src/hippocampus/mcp_server.py) - MCP tools for AI agent queries
- [src/hippocampus/db.py](src/hippocampus/db.py) - Database queries (semantic search, temporal traversal)
- [src/hippocampus/main.py](src/hippocampus/main.py) - CLI entry point
- [migrations/001_embeddings.sql](migrations/001_embeddings.sql) - Schema (embeddings table + consumer state)

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
