# Hippocampus

[![CI](https://github.com/RTrentJones/Hippocampus/actions/workflows/ci.yml/badge.svg)](https://github.com/RTrentJones/Hippocampus/actions/workflows/ci.yml)

Temporal RAG for AI agents—semantic search meets causal replay. Minimal infrastructure built entirely on Postgres using pg_kafka + pgvector. No Kafka cluster, no vector DB, just extensions.

## The Problem

Standard RAG returns disconnected chunks based on semantic similarity. When debugging AI agents, you need to understand *what happened before*—the causal chain that led to a decision, not just similar decisions.

**Hippocampus combines:**
- **Hybrid search** → Find an anchor point ("when did the agent touch auth?") via embedding similarity fused with keyword matching (RRF)
- **Causal traversal** → Walk explicit decision-to-decision links backward from that anchor ("what led to that?"), with a temporal-window fallback when no links were declared

### Declaring causality

Producers cannot know broker offsets at publish time, so causality is declared
with producer-assigned IDs in the decision payload:

```json
{
  "id": "d2",
  "parent_id": "d1",
  "context": "Found NullPointerException in AuthService.validateToken()",
  "reasoning": "Stack trace points at line 127; check recent changes",
  "action": "View git blame for AuthService.java:127",
  "anchor": "src/main/java/AuthService.java:127"
}
```

The embedding consumer resolves `parent_id` / `caused_by` (string or list) to
broker offsets and stores edges in `hippocampus.causal_edges`.
`replay_causal_chain` then walks those edges with a recursive CTE. Decisions
that never declare a parent still work—retrieval degrades to the events
immediately preceding the anchor in time, and every result is tagged with
which guarantee it carries (`causal_graph` vs `temporal_window`).

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

# Run migrations (in order)
psql -f migrations/001_embeddings.sql
psql -f migrations/002_causal_edges_hybrid_search.sql

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
| `find_similar(query, limit?, mode?)` | Search agent decisions: `hybrid` (default, vector + keyword RRF) or `semantic` (vector only) |
| `replay_causal_chain(anchor_offset, lookback?)` | Walk causal edges backward from an anchor; temporal-window fallback when no edges exist |
| `replay_topic(topic_name, from_offset?, limit?)` | View a single agent's decision history |
| `temporal_context(global_offset, window?)` | See all agents at a point in time |
| `what_touched(anchor, limit?)` | Find decisions affecting a file/location (trigram-indexed) |

## Evals

The central claim — *walking back from a failure beats retrieving by
similarity* — is measurable, so [`evals/`](evals/README.md) measures it:
synthetic agent sessions with planted causal chains and lexically-similar
distractors, five retrieval arms (semantic, hybrid, temporal window, causal
walk, and the full search-then-walk pipeline), scored with recall@k / MRR /
nDCG, plus an optional LLM-judge pass for end-to-end root-cause accuracy.

```bash
python -m evals.run --provider local --sessions 40 --output results.json
```

CI runs a deterministic smoke eval on every change. A long-form write-up
of the thesis, methodology, results, and the bugs the eval caught is in
[docs/writeup.md](docs/writeup.md). Measured results
(MiniLM embeddings, 40 sessions): similarity search degrades from 1.000 to
0.100 root-cause recall as look-alike noise grows, the causal walk holds at
1.000, and the full search-then-walk pipeline holds at 0.750 — see
[evals/RESULTS.md](evals/RESULTS.md) for tables, the bugs the eval caught,
and [evals/README.md](evals/README.md) for methodology.

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
