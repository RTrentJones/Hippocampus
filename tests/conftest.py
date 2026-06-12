"""Shared test fixtures for Hippocampus tests.

This module provides:
- Database connection fixtures
- Configurable embedding provider (mock or local via --embedding-provider flag)
- Test data seeding fixtures
- Global state patching fixtures
"""

import os
from unittest.mock import patch

import pytest

# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--embedding-provider",
        action="store",
        default=os.getenv("TEST_EMBEDDING_PROVIDER", "mock"),
        choices=["mock", "local"],
        help="Embedding provider: mock (fast, deterministic) or local (sentence-transformers)",
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests with mocked dependencies")
    config.addinivalue_line("markers", "integration: Integration tests requiring database")
    config.addinivalue_line("markers", "e2e: End-to-end tests requiring full infrastructure")
    config.addinivalue_line(
        "markers", "requires_local_embeddings: Tests that require local embedding provider"
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests that require local embeddings when using mock provider."""
    if config.getoption("--embedding-provider") == "mock":
        skip_local = pytest.mark.skip(reason="Requires --embedding-provider=local")
        for item in items:
            if "requires_local_embeddings" in item.keywords:
                item.add_marker(skip_local)


# =============================================================================
# Event Loop Configuration
# =============================================================================

# Note: Using default function-scoped event loop from pytest-asyncio.
# Each test gets its own event loop for proper isolation.

# =============================================================================
# Embedding Provider Fixtures
# =============================================================================


@pytest.fixture
def embedding_provider_type(request):
    """Get the configured embedding provider type."""
    return request.config.getoption("--embedding-provider")


@pytest.fixture
def embedding_provider(embedding_provider_type):
    """Create an embedding provider based on the --embedding-provider flag.

    Returns:
        MockEmbeddings (dimensions=1536) or LocalEmbeddings
    """
    if embedding_provider_type == "local":
        from hippocampus.embeddings import LocalEmbeddings

        try:
            return LocalEmbeddings()  # Uses sentence-transformers (all-MiniLM-L6-v2)
        except ModuleNotFoundError:
            pytest.skip("sentence-transformers not installed. Run: pip install -e '.[dev,local]'")
    else:
        from hippocampus.embeddings import MockEmbeddings

        # Use 1536 dimensions to match OpenAI text-embedding-3-small (production default)
        return MockEmbeddings(dimensions=1536)


@pytest.fixture
def mock_embedding_provider():
    """Create a mock embedding provider (always mock, for unit tests)."""
    from hippocampus.embeddings import MockEmbeddings

    return MockEmbeddings(dimensions=1536)


@pytest.fixture
def patched_provider(embedding_provider):
    """Patch the global embedding provider singleton."""
    import hippocampus.embeddings as embeddings_module

    original_provider = embeddings_module._provider
    embeddings_module._provider = embedding_provider

    with patch.object(embeddings_module, "get_provider", return_value=embedding_provider):
        yield embedding_provider

    embeddings_module._provider = original_provider


# =============================================================================
# Settings Fixtures
# =============================================================================


@pytest.fixture
def test_database_url():
    """Database URL for tests (can be overridden via env var)."""
    return os.getenv(
        "TEST_DATABASE_URL", "postgresql://postgres:postgres@localhost:5433/hippocampus_test"
    )


@pytest.fixture
def test_settings(test_database_url, embedding_provider_type):
    """Override settings for tests."""
    from hippocampus.config import Settings

    return Settings(
        database_url=test_database_url,
        kafka_bootstrap_servers=os.getenv("TEST_KAFKA_SERVERS", "localhost:9092"),
        kafka_consumer_group="hippocampus-test",
        kafka_topic_pattern="test-decisions.*",
        embedding_provider=embedding_provider_type,
        embedding_dimensions=1536,  # Match OpenAI text-embedding-3-small (production)
        batch_size=10,  # Smaller batches for faster tests
        batch_timeout_ms=100,
    )


@pytest.fixture
def patched_settings(test_settings):
    """Patch the global settings object.

    Modules import the settings object directly (`from .config import settings`),
    so each module's reference must be patched, not just the config module's.
    """
    with (
        patch("hippocampus.config.settings", test_settings),
        patch("hippocampus.consumer.settings", test_settings),
        patch("hippocampus.db.settings", test_settings),
    ):
        yield test_settings


# =============================================================================
# Database Fixtures
# =============================================================================


@pytest.fixture
async def db_pool():
    """Create a database pool for each test.

    Note: Requires PostgreSQL with pg_kafka and pgvector extensions.
    Run migrations/001_embeddings.sql before running integration tests.
    """
    import asyncpg
    from pgvector.asyncpg import register_vector

    url = os.getenv(
        "TEST_DATABASE_URL", "postgresql://postgres:postgres@localhost:5433/hippocampus_test"
    )

    try:
        pool = await asyncpg.create_pool(
            url, min_size=2, max_size=5, init=lambda conn: register_vector(conn)
        )
    except Exception as e:
        # Locally a missing database skips DB-bound tests; in CI that would
        # silently turn the whole integration suite green, so fail instead.
        if os.getenv("CI"):
            pytest.fail(f"Database not available in CI: {e}")
        pytest.skip(f"Database not available: {e}")
        return

    yield pool
    await pool.close()


@pytest.fixture
async def clean_db(db_pool):
    """Clean up test data before each test."""
    try:
        # Clean hippocampus tables but preserve schema
        await db_pool.execute("DELETE FROM hippocampus.causal_edges")
        await db_pool.execute("DELETE FROM hippocampus.embeddings")
        await db_pool.execute("DELETE FROM hippocampus.consumer_state")
        # Clean kafka tables (messages references topics, so delete messages first)
        await db_pool.execute("DELETE FROM kafka.messages")
        await db_pool.execute("DELETE FROM kafka.topics")
    except Exception:
        # Tables may not exist yet - that's OK for some tests
        pass
    yield


@pytest.fixture
async def patched_db_pool(db_pool, clean_db):
    """Patch the global db pool to use test pool."""
    import hippocampus.db as db_module

    original_pool = db_module._pool
    db_module._pool = db_pool
    yield db_pool
    db_module._pool = original_pool


@pytest.fixture
async def seeded_topic(db_pool):
    """Create a test topic and return its ID and name."""
    topic_name = "decisions.test.session_001"

    # Check if topic exists
    row = await db_pool.fetchrow("SELECT id FROM kafka.topics WHERE name = $1", topic_name)

    if row:
        topic_id = row["id"]
    else:
        # Insert topic
        topic_id = await db_pool.fetchval(
            "INSERT INTO kafka.topics (name) VALUES ($1) RETURNING id", topic_name
        )

    return {"topic_id": topic_id, "topic_name": topic_name}


@pytest.fixture
async def seeded_messages(db_pool, seeded_topic):
    """Seed the database with test messages and return their metadata.

    Returns:
        dict with topic_id, topic_name, and messages list
    """
    topic_id = seeded_topic["topic_id"]
    topic_name = seeded_topic["topic_name"]

    # Sample decision messages
    messages = [
        {
            "context": "User asked about authentication",
            "reasoning": "Need to check existing auth implementation",
            "action": "Read auth.py file",
            "anchor": "src/auth.py",
        },
        {
            "context": "Found authentication bug",
            "reasoning": "Missing null check causes crash",
            "action": "Add null check to login function",
            "anchor": "src/auth.py:42",
        },
        {
            "context": "Tests passing locally",
            "reasoning": "All edge cases covered",
            "action": "Create pull request",
        },
    ]

    import json

    inserted = []
    for i, msg in enumerate(messages):
        # Convert dict to JSON bytes for kafka.messages.value (bytea column)
        msg_bytes = json.dumps(msg).encode("utf-8")
        # Insert message into kafka.messages
        row = await db_pool.fetchrow(
            """
            INSERT INTO kafka.messages (topic_id, partition_id, partition_offset, value)
            VALUES ($1, 0, $2, $3)
            RETURNING topic_id, partition_id, partition_offset, global_offset
            """,
            topic_id,
            i,
            msg_bytes,
        )
        inserted.append(dict(row) | {"value": msg})

    return {"topic_id": topic_id, "topic_name": topic_name, "messages": inserted}


@pytest.fixture
async def seeded_embeddings(db_pool, seeded_messages, embedding_provider):
    """Seed messages with their embeddings.

    Returns:
        dict with topic_id, topic_name, messages (with embeddings)
    """
    from hippocampus.embeddings import format_for_embedding

    messages_with_embeddings = []

    for msg_record in seeded_messages["messages"]:
        # Generate embedding
        text = format_for_embedding(msg_record["value"])
        embedding = await embedding_provider.embed_one(text)

        # Store embedding
        await db_pool.execute(
            """
            INSERT INTO hippocampus.embeddings
                (topic_id, partition_id, partition_offset, embedding)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT DO NOTHING
            """,
            msg_record["topic_id"],
            msg_record["partition_id"],
            msg_record["partition_offset"],
            embedding,
        )

        messages_with_embeddings.append(msg_record | {"embedding": embedding, "text": text})

    return {
        "topic_id": seeded_messages["topic_id"],
        "topic_name": seeded_messages["topic_name"],
        "messages": messages_with_embeddings,
    }


# =============================================================================
# Kafka Fixtures
# =============================================================================


@pytest.fixture
def kafka_bootstrap_servers():
    """Kafka bootstrap servers for tests."""
    return os.getenv("TEST_KAFKA_SERVERS", "localhost:9092")


@pytest.fixture
def mock_kafka_consumer():
    """Mock KafkaConsumer for unit tests."""
    from unittest.mock import MagicMock

    consumer = MagicMock()
    consumer.poll.return_value = {}
    consumer.subscribe.return_value = None
    consumer.commit.return_value = None
    consumer.close.return_value = None
    return consumer


@pytest.fixture
def mock_consumer_record():
    """Factory for creating mock Kafka ConsumerRecords."""

    def _create(
        topic: str = "decisions.test.001",
        partition: int = 0,
        offset: int = 0,
        value: dict | None = None,
        key: str | None = None,
        use_default_value: bool = True,
    ):
        from unittest.mock import MagicMock

        record = MagicMock()
        record.topic = topic
        record.partition = partition
        record.offset = offset
        # Only use default if value is None AND use_default_value is True
        if value is None and use_default_value:
            record.value = {"context": "test", "action": "test action"}
        else:
            record.value = value
        record.key = key
        return record

    return _create


# =============================================================================
# Utility Fixtures
# =============================================================================


@pytest.fixture
def sample_decision():
    """Return a sample decision message."""
    return {
        "context": "User requested feature implementation",
        "reasoning": "The feature requires modifying the auth module",
        "action": "Edit src/auth.py to add new validation",
        "anchor": "src/auth.py:100",
    }


@pytest.fixture
def sample_decisions():
    """Return a list of sample decision messages."""
    return [
        {
            "context": "Investigating performance issue",
            "reasoning": "Profiling shows slow database query",
            "action": "Add index to users table",
            "anchor": "migrations/002_add_index.sql",
        },
        {
            "context": "Code review feedback received",
            "reasoning": "Need to handle edge case for empty input",
            "action": "Add input validation",
            "anchor": "src/validators.py:25",
        },
        {
            "context": "Documentation update needed",
            "reasoning": "New API endpoint added",
            "action": "Update API docs",
            "anchor": "docs/api.md",
        },
    ]
