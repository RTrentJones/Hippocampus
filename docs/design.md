# Hippocampus: Design Document

> **Provenance note:** This document began as the project's initial design
> scaffold (originally `scaffold.md`, used to bootstrap development with
> Claude Code). It is preserved as a design reference. Where it diverges from
> the implementation, the code and [README](../README.md) are authoritative —
> notably, causal links were initially aspirational and are now implemented
> via `hippocampus.causal_edges` (see `migrations/002_causal_edges_hybrid_search.sql`).

## Overview

Build **Hippocampus**, a temporal RAG system for AI agents. It provides causal memory—agents can understand *why* they made decisions, not just retrieve similar contexts.

This is a learning/exploration project with research goals. Not production-ready.

## Architecture

```
Agents publish to Kafka (fire-and-forget)
              │
              ▼
┌─────────────────────────────────────────┐
│         Kafka Topic: "decisions"        │
│         (pg_kafka on port 9092)         │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│       Hippocampus Consumer              │
│                                         │
│  • Consumes from Kafka                  │
│  • Generates embeddings (OpenAI)        │
│  • Stores in Postgres + pgvector        │
│  • Builds causal links                  │
└─────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│       Hippocampus MCP Server            │
│                                         │
│  • Query-only tools for agents          │
│  • replay_causal_chain()                │
│  • find_similar()                       │
│  • what_led_to()                        │
└─────────────────────────────────────────┘
```

**Key insight:** Agents write via Kafka (async, fire-and-forget). Agents read via MCP (sync queries). Clean separation.

## Tech Stack

- **Language:** Python 3.11+
- **MCP:** `mcp` package (Anthropic's official SDK)
- **Kafka client:** `kafka-python` (connects to pg_kafka or real Kafka)
- **Database:** PostgreSQL 17 with pgvector extension
- **Embeddings:** OpenAI API (`text-embedding-3-small`)
- **Async:** `asyncio` + `asyncpg` for non-blocking DB access

## Project Structure

```
hippocampus/
├── .devcontainer/
│   ├── devcontainer.json       # VS Code devcontainer config
│   ├── docker-compose.yml      # Multi-container setup (Python + Postgres)
│   ├── Dockerfile              # Python dev environment
│   ├── Dockerfile.postgres     # Postgres + pgvector + pg_kafka
│   └── post-create.sh          # Setup script (migrations, etc.)
├── pyproject.toml
├── README.md
├── .env.example
├── migrations/
│   └── 001_initial.sql
├── src/
│   └── hippocampus/
│       ├── __init__.py
│       ├── main.py             # Entry point
│       ├── config.py           # Settings from env
│       ├── consumer.py         # Kafka consumer + processing
│       ├── mcp_server.py       # MCP server + tool definitions
│       ├── storage/
│       │   ├── __init__.py
│       │   ├── base.py         # Abstract interface
│       │   └── postgres.py     # Postgres + pgvector implementation
│       ├── causal/
│       │   ├── __init__.py
│       │   └── graph.py        # Causal chain logic
│       └── embedding/
│           ├── __init__.py
│           └── openai.py       # OpenAI embedding client
├── tests/
│   ├── __init__.py
│   ├── test_consumer.py
│   ├── test_causal.py
│   └── test_storage.py
└── examples/
    └── simple_agent.py         # Demo agent that publishes decisions
```

## Database Schema

Create `migrations/001_initial.sql`:

```sql
-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Core decisions table
CREATE TABLE decisions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    kafka_offset BIGINT UNIQUE NOT NULL,
    session_id TEXT NOT NULL,
    parent_id UUID REFERENCES decisions(id),
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Decision content
    context TEXT NOT NULL,
    reasoning TEXT NOT NULL,
    action TEXT NOT NULL,
    anchor TEXT,

    -- Embedding for semantic search
    embedding vector(1536)
);

-- Indexes
CREATE INDEX idx_decisions_session ON decisions(session_id, kafka_offset);
CREATE INDEX idx_decisions_parent ON decisions(parent_id);
CREATE INDEX idx_decisions_created ON decisions(created_at);
CREATE INDEX idx_decisions_anchor ON decisions(anchor) WHERE anchor IS NOT NULL;

-- Vector similarity search (IVFFlat for now, can tune later)
CREATE INDEX idx_decisions_embedding ON decisions
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- View for causal chain traversal
CREATE OR REPLACE FUNCTION get_causal_chain(decision_uuid UUID)
RETURNS TABLE (
    id UUID,
    depth INT,
    session_id TEXT,
    context TEXT,
    reasoning TEXT,
    action TEXT,
    anchor TEXT,
    created_at TIMESTAMPTZ
) AS $$
    WITH RECURSIVE chain AS (
        -- Base case: the decision we're starting from
        SELECT
            d.id,
            0 as depth,
            d.session_id,
            d.parent_id,
            d.context,
            d.reasoning,
            d.action,
            d.anchor,
            d.created_at
        FROM decisions d
        WHERE d.id = decision_uuid

        UNION ALL

        -- Recursive case: walk up the parent chain
        SELECT
            d.id,
            c.depth + 1,
            d.session_id,
            d.parent_id,
            d.context,
            d.reasoning,
            d.action,
            d.anchor,
            d.created_at
        FROM decisions d
        INNER JOIN chain c ON d.id = c.parent_id
        WHERE c.depth < 100  -- Safety limit
    )
    SELECT
        id, depth, session_id, context, reasoning, action, anchor, created_at
    FROM chain
    ORDER BY depth DESC;  -- Root first, then children
$$ LANGUAGE SQL;
```

## Kafka Message Schema

Messages published to the `decisions` topic:

```json
{
    "id": "uuid-v4",
    "session_id": "string",
    "parent_id": "uuid-v4 | null",
    "context": "string",
    "reasoning": "string",
    "action": "string",
    "anchor": "string | null"
}
```

**Note:** `id` is generated client-side (UUID v4) so agents can reference it as `parent_id` in subsequent decisions without waiting for server response.

## Configuration

Create `src/hippocampus/config.py`:

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Kafka
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic: str = "decisions"
    kafka_consumer_group: str = "hippocampus"

    # Postgres
    database_url: str = "postgresql://postgres:postgres@localhost:5432/hippocampus"

    # OpenAI
    openai_api_key: str
    embedding_model: str = "text-embedding-3-small"

    # MCP Server
    mcp_server_name: str = "hippocampus"

    class Config:
        env_file = ".env"

settings = Settings()
```

## Implementation Order

### Phase 1: Project Setup (Week 1)

1. Initialize project with `pyproject.toml` (use `hatch` or `poetry`)
2. Create docker-compose.yml with:
   - Postgres 17 + pgvector
   - pg_kafka extension loaded
3. Run migrations
4. Verify pg_kafka is accepting connections on 9092

### Phase 2: Kafka Consumer (Week 2)

Implement `src/hippocampus/consumer.py`:

1. Connect to Kafka using `kafka-python`
2. Consume from `decisions` topic
3. For each message:
   - Parse JSON
   - Generate embedding via OpenAI
   - Insert into Postgres with embedding
4. Commit offsets after successful processing

Key considerations:
- Handle OpenAI rate limits gracefully
- Batch embedding requests if possible (OpenAI supports batching)
- Log kafka_offset for debugging

### Phase 3: Storage Layer (Week 3)

Implement `src/hippocampus/storage/postgres.py`:

```python
class PostgresStorage:
    async def insert_decision(self, decision: Decision, embedding: list[float]) -> None
    async def get_decision(self, id: UUID) -> Decision | None
    async def get_causal_chain(self, id: UUID) -> list[Decision]
    async def find_similar(self, embedding: list[float], limit: int) -> list[Decision]
    async def find_by_anchor(self, anchor: str) -> list[Decision]
```

Use `asyncpg` for async database access.

### Phase 4: MCP Server (Week 4)

Implement `src/hippocampus/mcp_server.py`:

```python
from mcp.server import Server
from mcp.types import Tool

server = Server("hippocampus")

@server.tool()
async def replay_causal_chain(decision_id: str) -> list[dict]:
    """
    Replay the chain of decisions that led to this decision.
    Returns decisions in causal order (root cause first).
    """
    ...

@server.tool()
async def find_similar(query: str, limit: int = 10) -> list[dict]:
    """
    Find decisions with similar context using semantic search.
    """
    ...

@server.tool()
async def what_led_to(anchor: str) -> list[dict]:
    """
    Find the causal chain for decisions affecting a specific code location.
    """
    ...

@server.tool()
async def get_decision(decision_id: str) -> dict:
    """
    Get a specific decision by ID.
    """
    ...
```

### Phase 5: Demo Agent (Week 5)

Create `examples/simple_agent.py`:

A simple coding agent that:
1. Receives a task
2. Makes decisions (publishes to Kafka)
3. Links decisions via parent_id
4. Queries Hippocampus for similar past decisions

This validates the full loop.

## Dev Container Setup

### `.devcontainer/devcontainer.json`

```json
{
    "name": "Hippocampus",
    "dockerComposeFile": "docker-compose.yml",
    "service": "dev",
    "workspaceFolder": "/workspace",

    "features": {
        "ghcr.io/devcontainers/features/python:1": {
            "version": "3.11"
        }
    },

    "customizations": {
        "vscode": {
            "extensions": [
                "ms-python.python",
                "ms-python.vscode-pylance",
                "charliermarsh.ruff",
                "ms-python.black-formatter",
                "mtxr.sqltools",
                "mtxr.sqltools-driver-pg"
            ],
            "settings": {
                "python.defaultInterpreterPath": "/usr/local/bin/python",
                "python.formatting.provider": "black",
                "python.linting.enabled": true,
                "python.linting.ruffEnabled": true,
                "[python]": {
                    "editor.formatOnSave": true,
                    "editor.codeActionsOnSave": {
                        "source.organizeImports": "explicit"
                    }
                },
                "sqltools.connections": [
                    {
                        "name": "Hippocampus DB",
                        "driver": "PostgreSQL",
                        "server": "postgres",
                        "port": 5432,
                        "database": "hippocampus",
                        "username": "postgres",
                        "password": "postgres"
                    }
                ]
            }
        }
    },

    "forwardPorts": [5432, 9092],
    "portsAttributes": {
        "5432": { "label": "PostgreSQL" },
        "9092": { "label": "Kafka (pg_kafka)" }
    },

    "postCreateCommand": "bash .devcontainer/post-create.sh",

    "remoteEnv": {
        "DATABASE_URL": "postgresql://postgres:postgres@postgres:5432/hippocampus",
        "KAFKA_BOOTSTRAP_SERVERS": "postgres:9092",
        "PYTHONPATH": "/workspace/src"
    }
}
```

### `.devcontainer/docker-compose.yml`

```yaml
services:
  dev:
    build:
      context: .
      dockerfile: Dockerfile
    volumes:
      - ..:/workspace:cached
      - pip-cache:/root/.cache/pip
    command: sleep infinity
    depends_on:
      postgres:
        condition: service_healthy
    environment:
      DATABASE_URL: postgresql://postgres:postgres@postgres:5432/hippocampus
      KAFKA_BOOTSTRAP_SERVERS: postgres:9092
      OPENAI_API_KEY: ${OPENAI_API_KEY}

  postgres:
    build:
      context: .
      dockerfile: Dockerfile.postgres
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: hippocampus
    ports:
      - "5432:5432"
      - "9092:9092"
    volumes:
      - postgres-data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres && psql -U postgres -c 'SELECT 1'"]
      interval: 5s
      timeout: 5s
      retries: 10

volumes:
  postgres-data:
  pip-cache:
```

### `.devcontainer/Dockerfile` (Python dev environment)

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python tools
RUN pip install --upgrade pip setuptools wheel
RUN pip install hatch pytest pytest-asyncio black ruff

# Install kcat for Kafka testing
RUN apt-get update && apt-get install -y kafkacat && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
```

### `.devcontainer/Dockerfile.postgres` (Postgres + pgvector + pg_kafka)

```dockerfile
# Stage 1: Build pg_kafka
FROM rust:latest AS builder

RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    libclang-dev \
    postgresql-server-dev-17 \
    && rm -rf /var/lib/apt/lists/*

# Install nightly Rust (required by pgrx)
RUN rustup default nightly

# Install pgrx
RUN cargo install cargo-pgrx --version 0.16.1 --locked
RUN cargo pgrx init --pg17 /usr/bin/pg_config

# Clone and build pg_kafka
RUN git clone https://github.com/RTrentJones/pg_kafka.git /pg_kafka
WORKDIR /pg_kafka
RUN cargo pgrx package --pg-config /usr/bin/pg_config

# Stage 2: Runtime image
FROM pgvector/pgvector:pg17

# Copy pg_kafka extension from builder
COPY --from=builder /pg_kafka/target/release/pg_kafka-pg17/usr/share/postgresql/17/extension/* \
    /usr/share/postgresql/17/extension/
COPY --from=builder /pg_kafka/target/release/pg_kafka-pg17/usr/lib/postgresql/17/lib/* \
    /usr/lib/postgresql/17/lib/

# Configure PostgreSQL
RUN echo "shared_preload_libraries = 'pg_kafka'" >> /usr/share/postgresql/postgresql.conf.sample
RUN echo "pg_kafka.port = 9092" >> /usr/share/postgresql/postgresql.conf.sample
RUN echo "pg_kafka.host = '0.0.0.0'" >> /usr/share/postgresql/postgresql.conf.sample

# Expose both Postgres and Kafka ports
EXPOSE 5432 9092
```

### `.devcontainer/post-create.sh`

```bash
#!/bin/bash
set -e

echo "=== Hippocampus Dev Container Setup ==="

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -e ".[dev]"

# Wait for Postgres to be ready
echo "Waiting for PostgreSQL..."
until pg_isready -h postgres -U postgres; do
    sleep 1
done

# Run migrations
echo "Running migrations..."
psql -h postgres -U postgres -d hippocampus -f /workspace/migrations/001_initial.sql || true

# Create pg_kafka schema if not exists
echo "Ensuring pg_kafka extension..."
psql -h postgres -U postgres -d hippocampus -c "CREATE EXTENSION IF NOT EXISTS pg_kafka;" || true

# Verify setup
echo ""
echo "=== Verification ==="
echo "PostgreSQL: $(psql -h postgres -U postgres -d hippocampus -t -c 'SELECT version();' | head -1)"
echo "pgvector: $(psql -h postgres -U postgres -d hippocampus -t -c "SELECT extversion FROM pg_extension WHERE extname='vector';" | head -1)"
echo "pg_kafka: $(psql -h postgres -U postgres -d hippocampus -t -c "SELECT extversion FROM pg_extension WHERE extname='pg_kafka';" | head -1)"

# Test Kafka connectivity
echo ""
echo "Testing Kafka (pg_kafka) connectivity..."
timeout 5 bash -c 'cat < /dev/null > /dev/tcp/postgres/9092' && echo "✅ Kafka port 9092 is open" || echo "⚠️ Kafka port not ready yet"

echo ""
echo "=== Setup Complete ==="
echo "Database URL: postgresql://postgres:postgres@postgres:5432/hippocampus"
echo "Kafka Bootstrap: postgres:9092"
echo ""
echo "Quick start:"
echo "  python -m hippocampus.main     # Start the service"
echo "  pytest                          # Run tests"
echo "  kcat -L -b postgres:9092        # List Kafka topics"
```

### `.env.example`

```bash
# Required
OPENAI_API_KEY=sk-your-key-here

# Optional (defaults shown)
DATABASE_URL=postgresql://postgres:postgres@postgres:5432/hippocampus
KAFKA_BOOTSTRAP_SERVERS=postgres:9092
KAFKA_TOPIC=decisions
KAFKA_CONSUMER_GROUP=hippocampus
EMBEDDING_MODEL=text-embedding-3-small
```

## Using the Dev Container

### VS Code (Recommended)

1. Open project in VS Code
2. Install "Dev Containers" extension
3. Press `F1` → "Dev Containers: Reopen in Container"
4. Wait for build (first time takes ~5-10 min for pg_kafka build)
5. Terminal opens with everything ready

### Manual Docker

```bash
# Build and start
cd .devcontainer
docker-compose up -d

# Exec into dev container
docker exec -it hippocampus-dev-1 bash

# Run the service
python -m hippocampus.main
```

### Verify Setup

```bash
# Check Postgres + extensions
psql -h postgres -U postgres -d hippocampus -c "\dx"

# Test Kafka via kcat
kcat -L -b postgres:9092

# Produce a test message
echo '{"id":"test","session_id":"s1","context":"test","reasoning":"test","action":"test"}' | \
  kcat -P -b postgres:9092 -t decisions

# Consume it back
kcat -C -b postgres:9092 -t decisions -o beginning -c 1
```

## pyproject.toml

```toml
[project]
name = "hippocampus"
version = "0.1.0"
description = "Temporal RAG for AI agents"
requires-python = ">=3.11"
dependencies = [
    "mcp>=0.1.0",
    "kafka-python>=2.0.2",
    "asyncpg>=0.29.0",
    "pgvector>=0.2.4",
    "openai>=1.0.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "black>=24.0.0",
    "ruff>=0.2.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
asyncio_mode = "auto"

[tool.ruff]
line-length = 100
```

## Key Dependencies Documentation

- **MCP SDK:** https://github.com/anthropics/anthropic-cookbook/tree/main/misc/mcp
- **kafka-python:** https://kafka-python.readthedocs.io/
- **asyncpg:** https://magicstack.github.io/asyncpg/
- **pgvector Python:** https://github.com/pgvector/pgvector-python

## Testing Strategy

1. **Unit tests:** Test causal graph logic with in-memory data
2. **Integration tests:** Test against real Postgres (use testcontainers or docker-compose)
3. **End-to-end:** Publish decisions via Kafka, verify they appear in MCP queries

## Success Criteria for v0.1

- [ ] Can publish a decision to Kafka and see it in Postgres
- [ ] Embeddings are generated and stored
- [ ] `replay_causal_chain` returns correct ancestor chain
- [ ] `find_similar` returns semantically relevant decisions
- [ ] MCP server connects to Claude Desktop
- [ ] Example agent demonstrates full loop

## Notes for Claude Code

1. Start with the consumer—it's the core data pipeline
2. Use async throughout (asyncpg, async MCP handlers)
3. Don't over-abstract storage layer initially—can refactor later
4. The MCP SDK may have quirks; consult docs and experiment
5. pg_kafka is public at https://github.com/RTrentJones/pg_kafka - use its devcontainer or build instructions

## CLAUDE.md

Create a `CLAUDE.md` file for Claude Code context (like pg_kafka has):

```markdown
# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

Hippocampus is a temporal RAG system for AI agents. It provides causal memory—
agents can understand *why* they made decisions, not just retrieve similar contexts.

**Status:** Learning/exploration project with research goals.

## Architecture

- Agents publish decisions to Kafka (fire-and-forget via pg_kafka)
- Hippocampus consumer processes decisions, generates embeddings, stores in Postgres
- MCP server provides query-only tools for agents

## Development Commands

\`\`\`bash
# Start dev environment
# (VS Code: F1 → "Dev Containers: Reopen in Container")

# Run the service
python -m hippocampus.main

# Run tests
pytest
pytest -xvs tests/test_consumer.py  # Single file, verbose

# Format and lint
black src tests
ruff check src tests

# Test Kafka connectivity
kcat -L -b postgres:9092
\`\`\`

## Key Files

- \`src/hippocampus/consumer.py\` - Kafka consumer, embedding generation
- \`src/hippocampus/mcp_server.py\` - MCP tool definitions
- \`src/hippocampus/storage/postgres.py\` - Database operations
- \`migrations/001_initial.sql\` - Schema with pgvector

## Dependencies

- **pg_kafka**: https://github.com/RTrentJones/pg_kafka (Kafka protocol for Postgres)
- **MCP SDK**: Anthropic's Model Context Protocol
- **pgvector**: Vector similarity search in Postgres
- **OpenAI**: Embedding generation

## Testing Strategy

1. Unit tests: Mock the storage layer
2. Integration tests: Use testcontainers or docker-compose Postgres
3. E2E tests: Publish via Kafka, verify via MCP queries
```

## Build Caching Tips

The pg_kafka Rust build takes 5-10 minutes. To avoid rebuilding:

1. **Use Docker layer caching** — The multi-stage Dockerfile caches the builder stage
2. **Pre-build the image** — Run `docker-compose build postgres` once, then it's cached
3. **Volume mount the build** — For active pg_kafka development, mount the built extension

```bash
# Pre-build and tag for reuse
cd .devcontainer
docker build -t hippocampus-postgres -f Dockerfile.postgres .

# Reference in docker-compose.yml:
# postgres:
#   image: hippocampus-postgres
```

## Out of Scope for Initial Scaffold

- GraduatedStorage (Neo4j + Pinecone + Kafka) — future work
- Authentication/authorization
- Multi-tenancy
- Performance optimization
- Retention policies
