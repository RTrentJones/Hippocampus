"""End-to-end tests for semantic search functionality.

Tests the vector similarity search that enables finding relevant decisions
based on natural language queries.
"""

import json

import pytest


@pytest.mark.e2e
class TestSemanticSearchBasics:
    """Basic semantic search tests."""

    async def test_find_similar_returns_results(
        self, patched_db_pool, seeded_embeddings, embedding_provider, patched_provider
    ):
        """Test that semantic search returns results."""
        from hippocampus.db import find_similar

        query_embedding = await embedding_provider.embed_one("authentication bug")
        results = await find_similar(query_embedding, limit=10)

        assert len(results) > 0
        assert "value" in results[0]
        assert "distance" in results[0]

    async def test_find_similar_respects_limit(
        self, patched_db_pool, seeded_embeddings, embedding_provider, patched_provider
    ):
        """Test that limit parameter is respected."""
        from hippocampus.db import find_similar

        query_embedding = await embedding_provider.embed_one("test query")

        results_5 = await find_similar(query_embedding, limit=5)
        results_2 = await find_similar(query_embedding, limit=2)

        assert len(results_2) <= 2
        assert len(results_5) <= 5

    async def test_find_similar_with_topic_filter(
        self, patched_db_pool, seeded_embeddings, embedding_provider, patched_provider
    ):
        """Test filtering by topic pattern."""
        from hippocampus.db import find_similar

        query_embedding = await embedding_provider.embed_one("test query")

        results = await find_similar(
            query_embedding, limit=10, topic_pattern="decisions.test.%"
        )

        for result in results:
            assert result["topic"].startswith("decisions.test.")


@pytest.mark.e2e
class TestSemanticSearchRanking:
    """Tests for search result ranking."""

    async def test_results_ordered_by_distance(
        self, patched_db_pool, seeded_embeddings, embedding_provider, patched_provider
    ):
        """Test that results are ordered by distance (ascending)."""
        from hippocampus.db import find_similar

        query_embedding = await embedding_provider.embed_one("authentication")

        results = await find_similar(query_embedding, limit=10)

        if len(results) > 1:
            distances = [r["distance"] for r in results]
            assert distances == sorted(distances)

    @pytest.mark.requires_local_embeddings
    async def test_semantic_similarity_accuracy(
        self, patched_db_pool, seeded_topic, embedding_provider, patched_provider
    ):
        """Test that semantically similar queries find relevant results.

        This test requires local embeddings for meaningful semantic comparison.
        Mock embeddings use hash-based vectors which don't capture semantics.
        """
        from hippocampus.db import find_similar, store_embedding
        from hippocampus.embeddings import format_for_embedding

        topic_id = seeded_topic["topic_id"]

        # Insert messages with distinct topics
        test_messages = [
            {
                "context": "Database connection pool exhausted",
                "reasoning": "Too many concurrent connections",
                "action": "Increase pool size",
                "offset": 4000,
            },
            {
                "context": "User login failed with invalid password",
                "reasoning": "Password validation error",
                "action": "Check password hashing",
                "offset": 4001,
            },
            {
                "context": "API rate limit exceeded",
                "reasoning": "Too many requests from single IP",
                "action": "Implement backoff strategy",
                "offset": 4002,
            },
            {
                "context": "Memory usage high on server",
                "reasoning": "Memory leak in cache handler",
                "action": "Fix memory leak",
                "offset": 4003,
            },
        ]

        # Insert and embed
        for msg in test_messages:
            msg_data = {k: v for k, v in msg.items() if k != "offset"}
            await patched_db_pool.execute(
                """
                INSERT INTO kafka.messages (topic_id, partition_id, partition_offset, value)
                VALUES ($1, 0, $2, $3)
                """,
                topic_id,
                msg["offset"],
                json.dumps(msg_data).encode("utf-8"),
            )

            text = format_for_embedding(msg_data)
            embedding = await embedding_provider.embed_one(text)
            await store_embedding(topic_id, 0, msg["offset"], embedding)

        # Query for database-related issues
        query_embedding = await embedding_provider.embed_one(
            "database connection problems"
        )
        results = await find_similar(query_embedding, limit=4)

        # The database-related message should be the top result
        assert len(results) > 0
        top_result = results[0]["value"]
        assert "database" in top_result.get("context", "").lower() or \
               "pool" in top_result.get("context", "").lower()


@pytest.mark.e2e
class TestSemanticSearchWithMixedContent:
    """Tests for semantic search with varied content."""

    async def test_search_across_multiple_topics(
        self, patched_db_pool, embedding_provider, patched_provider
    ):
        """Test searching across messages from multiple topics."""
        from hippocampus.db import find_similar, store_embedding
        from hippocampus.embeddings import format_for_embedding

        # Create multiple topics
        topics = []
        for i, name in enumerate(["decisions.agent1.001", "decisions.agent2.001"]):
            topic_id = await patched_db_pool.fetchval(
                "INSERT INTO kafka.topics (name) VALUES ($1) RETURNING id", name
            )
            topics.append({"id": topic_id, "name": name})

        # Insert messages in different topics
        messages = [
            (topics[0]["id"], 0, {"context": "Agent1 auth issue", "action": "Fix login"}),
            (topics[1]["id"], 0, {"context": "Agent2 auth issue", "action": "Update token"}),
            (topics[0]["id"], 1, {"context": "Agent1 performance", "action": "Optimize"}),
            (topics[1]["id"], 1, {"context": "Agent2 database", "action": "Add index"}),
        ]

        for topic_id, offset, msg in messages:
            await patched_db_pool.execute(
                """
                INSERT INTO kafka.messages (topic_id, partition_id, partition_offset, value)
                VALUES ($1, 0, $2, $3)
                """,
                topic_id,
                4100 + offset,
                json.dumps(msg).encode("utf-8"),
            )

            text = format_for_embedding(msg)
            embedding = await embedding_provider.embed_one(text)
            await store_embedding(topic_id, 0, 4100 + offset, embedding)

        # Search for auth issues
        query_embedding = await embedding_provider.embed_one("authentication problem")
        results = await find_similar(query_embedding, limit=10)

        # Should find results from both topics
        result_topics = {r["topic"] for r in results}
        assert len(result_topics) >= 1  # At least one topic with auth issues

    async def test_empty_search_results(
        self, patched_db_pool, seeded_embeddings, embedding_provider, patched_provider
    ):
        """Test search with non-matching topic filter returns empty."""
        from hippocampus.db import find_similar

        query_embedding = await embedding_provider.embed_one("test query")

        results = await find_similar(
            query_embedding, limit=10, topic_pattern="nonexistent.topic.%"
        )

        assert len(results) == 0


@pytest.mark.e2e
class TestSemanticSearchEdgeCases:
    """Edge case tests for semantic search."""

    async def test_search_with_special_characters(
        self, patched_db_pool, seeded_topic, embedding_provider, patched_provider
    ):
        """Test searching with special characters in query."""
        from hippocampus.db import find_similar, store_embedding
        from hippocampus.embeddings import format_for_embedding

        topic_id = seeded_topic["topic_id"]

        # Insert message with special characters
        msg = {
            "context": "Error: NullPointerException at line 42",
            "action": "Fix null check",
            "anchor": "src/utils.java:42",
        }

        await patched_db_pool.execute(
            """
            INSERT INTO kafka.messages (topic_id, partition_id, partition_offset, value)
            VALUES ($1, 0, $2, $3)
            """,
            topic_id,
            4200,
            json.dumps(msg).encode("utf-8"),
        )

        text = format_for_embedding(msg)
        embedding = await embedding_provider.embed_one(text)
        await store_embedding(topic_id, 0, 4200, embedding)

        # Search with special characters
        query_embedding = await embedding_provider.embed_one(
            "NullPointerException error"
        )
        results = await find_similar(query_embedding, limit=10)

        assert len(results) > 0

    async def test_search_with_unicode(
        self, patched_db_pool, seeded_topic, embedding_provider, patched_provider
    ):
        """Test searching with unicode characters."""
        from hippocampus.db import find_similar, store_embedding
        from hippocampus.embeddings import format_for_embedding

        topic_id = seeded_topic["topic_id"]

        # Insert message with unicode
        msg = {
            "context": "処理中にエラーが発生しました",  # "An error occurred during processing"
            "action": "エラーハンドリングを追加",  # "Add error handling"
        }

        await patched_db_pool.execute(
            """
            INSERT INTO kafka.messages (topic_id, partition_id, partition_offset, value)
            VALUES ($1, 0, $2, $3)
            """,
            topic_id,
            4300,
            json.dumps(msg).encode("utf-8"),
        )

        text = format_for_embedding(msg)
        embedding = await embedding_provider.embed_one(text)
        await store_embedding(topic_id, 0, 4300, embedding)

        # Search should not fail
        query_embedding = await embedding_provider.embed_one("エラー")
        results = await find_similar(query_embedding, limit=10)

        # Should execute without error (results depend on embedding quality)
        assert isinstance(results, list)

    async def test_search_with_long_content(
        self, patched_db_pool, seeded_topic, embedding_provider, patched_provider
    ):
        """Test searching messages with long content."""
        from hippocampus.db import find_similar, store_embedding
        from hippocampus.embeddings import format_for_embedding

        topic_id = seeded_topic["topic_id"]

        # Insert message with long content
        msg = {
            "context": "This is a very long context " * 50,
            "reasoning": "This is very detailed reasoning " * 30,
            "action": "Perform complex action",
        }

        await patched_db_pool.execute(
            """
            INSERT INTO kafka.messages (topic_id, partition_id, partition_offset, value)
            VALUES ($1, 0, $2, $3)
            """,
            topic_id,
            4400,
            json.dumps(msg).encode("utf-8"),
        )

        text = format_for_embedding(msg)
        embedding = await embedding_provider.embed_one(text)
        await store_embedding(topic_id, 0, 4400, embedding)

        # Search should work
        query_embedding = await embedding_provider.embed_one("long context")
        results = await find_similar(query_embedding, limit=10)

        assert isinstance(results, list)
