# Hippocampus

Temporal RAG for AI agents—semantic search meets causal replay. Minimal infrastructure built entirely on Postgres using pg_kafka + pgvector. No Kafka cluster, no vector DB, just extensions.

## The Problem

Standard RAG returns disconnected chunks based on semantic similarity. When debugging AI agents, you need to understand *what happened before*—the causal chain that led to a decision, not just similar decisions.

**Hippocampus combines:**
- **Semantic search** → Find an anchor point ("when did the agent touch auth?")
- **Temporal traversal** → Walk backward through the event log ("what led to that?")

## Why Postgres-Only?

Most agent memory systems assume you'll spin up Kafka, a vector database, and various middleware. This project asks: **what's the minimum viable infrastructure?**

The answer: just Postgres.

| Extension | Replaces | Purpose |
|-----------|----------|---------|
| [pg_kafka](https://github.com/RTrentJones/pg_kafka) | Kafka cluster | Kafka protocol as a Postgres extension |
| [pgvector](https://github.com/pgvector/pgvector) | Pinecone/Weaviate | Vector similarity search |

One database. One connection string. Zero external services.

## Project Status

**Learning/exploration project** — not production-ready

This is an exploration of minimal agent memory infrastructure. The goal is to learn the patterns with simple tooling, then graduate to production systems when scale demands it.

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Configure
cp .env.example .env
# Edit .env: add OPENAI_API_KEY, set DATABASE_URL

# Run migrations
psql -f migrations/001_embeddings.sql

# Start the embedding consumer (processes agent decisions)
hippocampus consumer

# Start the MCP server (in another terminal)
hippocampus server
```

### Claude Desktop Integration

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "hippocampus": {
      "command": "hippocampus",
      "args": ["server"]
    }
  }
}
```

## MCP Tools

| Tool | Purpose |
|------|---------|
| `find_similar(query, limit?)` | Semantic search across all agent decisions |
| `replay_causal_chain(anchor_offset, lookback?)` | Walk backward from an anchor point |
| `replay_topic(topic_name, from_offset?, limit?)` | View a single agent's decision history |
| `temporal_context(global_offset, window?)` | See all agents at a point in time |
| `what_touched(anchor, limit?)` | Find decisions affecting a file/location |

## Scaling Roadmap

**Current:** Single Postgres instance with pg_kafka + pgvector

**Future graduation paths:**

| Component | Current | Graduate To | When |
|-----------|---------|-------------|------|
| Vector storage | pgvector | Pinecone, Weaviate | >1M vectors |
| Graph queries | SQL recursive CTE | Neo4j | Complex relationship traversals |
| Messaging | pg_kafka | Real Kafka | >10k msg/sec, multi-datacenter |

**The key insight:** The MCP tool interface stays the same. Graduation means swapping implementations, not rewriting agent integrations.

## Development

See [CLAUDE.md](CLAUDE.md) for development setup, testing infrastructure, and architecture details.

## License

MIT
