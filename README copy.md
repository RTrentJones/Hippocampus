# Hippocampus

**Temporal RAG for AI Agent Debugging**

*Named for the brain region responsible for memory formation and temporal sequencing.*

---

## The Problem

Standard RAG (Retrieval Augmented Generation) retrieves information based on semantic similarity, losing temporal causality. When an agent asks "Why did the server crash?", standard RAG returns:

- The crash log itself (high semantic match)
- A similar crash log from 2022 (also high semantic match)
- Documentation about error codes (moderate match)

What it *misses*: the config change that happened 1 second before—the actual cause.

## The Solution

Hippocampus combines **semantic search** (find an entry point) with **temporal traversal** (walk backward through the causal chain).

Built on [pg_kafka](https://github.com/RTrentJones/pg_kafka), it transforms Postgres into a temporal knowledge graph where:

1. **Vector search** finds an anchor point ("What event looks like what I'm investigating?")
2. **Offset traversal** walks backward through the event log ("What happened before this?")
3. **MCP tools** expose these queries to AI agents (Claude, Cline, etc.)

## Quick Start

### Prerequisites

- Docker and Docker Compose
- OpenAI API key (for embeddings)

### Dev Container (Recommended)

```bash
# Clone the repo
git clone https://github.com/RTrentJones/hippocampus.git
cd hippocampus

# Create .env file
echo "OPENAI_API_KEY=sk-your-key-here" > .env

# Open in VS Code with Dev Containers extension
code .
# Then: Cmd+Shift+P → "Dev Containers: Reopen in Container"
```

### Manual Setup

```bash
# Install dependencies
pip install -e ".[dev]"

# Run migrations (requires Postgres with pg_kafka + pgvector)
psql -f migrations/001_embeddings.sql

# Start the consumer (generates embeddings)
hippocampus consumer &

# Start the MCP server
hippocampus server
```

## Usage

### Producing Agent Decisions

Agents publish decisions to Kafka topics:

```python
from kafka import KafkaProducer
import json

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode()
)

producer.send('decisions.myagent.session_001', {
    'id': 'uuid-here',
    'context': 'User asked to refactor auth module',
    'reasoning': 'Current implementation mixes concerns...',
    'action': 'Extract AuthService class',
    'anchor': 'src/auth.py:45-67'
})
```

### Querying via MCP

Add to your Claude Desktop config (`~/.config/claude/claude_desktop_config.json`):

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

Then ask Claude:

> "Find decisions similar to 'authentication refactor'"
> "Show me what happened before global_offset 500"
> "What touched src/auth.py recently?"

## MCP Tools

| Tool | Description |
|------|-------------|
| `find_similar(query, limit?)` | Semantic search across all agent decisions |
| `replay_causal_chain(anchor_offset, lookback?)` | Events leading up to an anchor point |
| `replay_topic(topic_name, from_offset?, limit?)` | Single agent's decision history |
| `temporal_context(global_offset, window?)` | Cross-agent activity at a point in time |
| `what_touched(anchor, limit?)` | Decisions affecting a specific file |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Postgres                                 │
│                                                                  │
│  ┌────────────────┐    ┌────────────────────────────────────┐  │
│  │  kafka.topics  │    │         kafka.messages              │  │
│  │  ────────────  │    │  ────────────────────────────────   │  │
│  │  id, name      │◄───│  topic_id, partition_id, offset    │  │
│  └────────────────┘    │  global_offset (temporal ordering)  │  │
│                        │  key, value, headers, created_at    │  │
│                        └────────────────────────────────────┘  │
│                                      │                          │
│                                      │ FK                       │
│                                      ▼                          │
│                        ┌────────────────────────────────────┐  │
│                        │    hippocampus.embeddings          │  │
│                        │  ────────────────────────────────   │  │
│                        │  topic_id, partition_id, offset    │  │
│                        │  embedding vector(1536)             │  │
│                        └────────────────────────────────────┘  │
│                                                                  │
│  pg_kafka extension: Kafka wire protocol on port 9092           │
│  pgvector extension: Vector similarity search                    │
└─────────────────────────────────────────────────────────────────┘
```

## Why Postgres?

> "The best database is the one you're already running."

Instead of deploying Kafka + Pinecone + Neo4j, Hippocampus gives you:

- **Message streaming** via pg_kafka (Kafka-compatible)
- **Vector search** via pgvector
- **Temporal ordering** via pg_kafka's global_offset

One database. One deployment. Clean graduation paths when scale demands it.

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `DATABASE_URL` | `postgresql://...` | Postgres connection string |
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | pg_kafka endpoint |
| `OPENAI_API_KEY` | (required) | For embedding generation |
| `EMBEDDING_PROVIDER` | `openai` | `openai` or `local` |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI model name |

## Related Projects

- [pg_kafka](https://github.com/RTrentJones/pg_kafka) - Kafka-compatible Postgres extension (foundation)
- [pgvector](https://github.com/pgvector/pgvector) - Vector similarity search for Postgres
- [MCP](https://modelcontextprotocol.io/) - Model Context Protocol

## License

MIT
