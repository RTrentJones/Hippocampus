"""End-to-end tests for the full Hippocampus pipeline.

Tests the complete flow from message ingestion through embedding to query.
"""

import json

import pytest


@pytest.mark.e2e
class TestFullPipeline:
    """Tests for the complete message pipeline."""

    async def test_message_to_embedding_to_query(
        self, patched_db_pool, seeded_topic, patched_provider, embedding_provider
    ):
        """Test the complete pipeline: message -> embedding -> semantic search.

        Flow:
        1. Insert a message into kafka.messages
        2. Generate and store its embedding
        3. Query for similar messages
        4. Verify the message is found
        """
        from hippocampus.db import find_similar, store_embedding
        from hippocampus.embeddings import format_for_embedding

        topic_id = seeded_topic["topic_id"]

        # 1. Create a distinctive message
        message = {
            "context": "User reported critical security vulnerability in authentication",
            "reasoning": "Need to patch immediately before exploit",
            "action": "Apply security patch to login handler",
            "anchor": "src/security/auth_handler.py",
        }

        # Insert into kafka.messages (value is bytea, so encode as JSON)
        await patched_db_pool.execute(
            """
            INSERT INTO kafka.messages (topic_id, partition_id, partition_offset, value)
            VALUES ($1, 0, $2, $3)
            """,
            topic_id,
            1000,
            json.dumps(message).encode("utf-8"),
        )

        # 2. Generate and store embedding
        text = format_for_embedding(message)
        embedding = await embedding_provider.embed_one(text)

        await store_embedding(
            topic_id=topic_id,
            partition_id=0,
            partition_offset=1000,
            embedding=embedding,
        )

        # 3. Query for similar messages
        query_embedding = await embedding_provider.embed_one(
            "security vulnerability authentication"
        )
        results = await find_similar(query_embedding, limit=10)

        # 4. Verify our message is found
        assert len(results) > 0

        # The security-related message should be among top results
        found = False
        for result in results:
            if "security vulnerability" in result["value"].get("context", ""):
                found = True
                break

        assert found, "Expected message not found in search results"

    async def test_batch_pipeline(
        self, patched_db_pool, seeded_topic, patched_provider, embedding_provider
    ):
        """Test batch processing pipeline."""
        from hippocampus.db import store_embeddings_batch
        from hippocampus.embeddings import format_for_embedding

        topic_id = seeded_topic["topic_id"]

        # Create a batch of messages
        messages = [
            {
                "context": "Performance profiling shows database queries are slow",
                "reasoning": "Adding index should improve query time",
                "action": "Create index on users.email column",
            },
            {
                "context": "Unit tests failing on CI",
                "reasoning": "Mock was not properly configured",
                "action": "Fix mock setup in test_auth.py",
            },
            {
                "context": "Documentation is outdated",
                "reasoning": "API has changed since last update",
                "action": "Update API documentation",
            },
        ]

        # Insert messages (value is bytea, so encode as JSON)
        inserted = []
        for i, msg in enumerate(messages):
            row = await patched_db_pool.fetchrow(
                """
                INSERT INTO kafka.messages (topic_id, partition_id, partition_offset, value)
                VALUES ($1, 0, $2, $3)
                RETURNING topic_id, partition_id, partition_offset
                """,
                topic_id,
                1100 + i,
                json.dumps(msg).encode("utf-8"),
            )
            inserted.append(dict(row))

        # Format and embed
        texts = [format_for_embedding(msg) for msg in messages]
        embeddings = await embedding_provider.embed(texts)

        # Store batch
        records = [
            (ins["topic_id"], ins["partition_id"], ins["partition_offset"], emb)
            for ins, emb in zip(inserted, embeddings)
        ]
        inserted, _offsets = await store_embeddings_batch(records)

        assert inserted == len(messages)

        # Verify stored
        count = await patched_db_pool.fetchval(
            """
            SELECT COUNT(*) FROM hippocampus.embeddings
            WHERE topic_id = $1 AND partition_offset >= 1100
            """,
            topic_id,
        )
        assert count == len(messages)


@pytest.mark.e2e
class TestConsumerSimulation:
    """Tests simulating consumer behavior."""

    async def test_consumer_like_processing(
        self, patched_db_pool, seeded_topic, patched_provider, embedding_provider
    ):
        """Simulate what the consumer does without actual Kafka.

        This tests the core consumer logic without Kafka dependencies.
        """
        from hippocampus.db import (
            get_consumer_offset,
            store_embeddings_batch,
            update_consumer_offset,
        )
        from hippocampus.embeddings import format_for_embedding

        topic_id = seeded_topic["topic_id"]
        consumer_id = "test-e2e-consumer"

        # 1. Check initial offset
        initial_offset = await get_consumer_offset(consumer_id)
        assert initial_offset == 0

        # 2. Simulate receiving messages
        messages = [
            {
                "context": f"E2E test message {i}",
                "reasoning": f"Testing consumer flow {i}",
                "action": f"Action {i}",
            }
            for i in range(5)
        ]

        # 3. Insert messages (simulating Kafka producing them)
        global_offsets = []
        for i, msg in enumerate(messages):
            row = await patched_db_pool.fetchrow(
                """
                INSERT INTO kafka.messages (topic_id, partition_id, partition_offset, value)
                VALUES ($1, 0, $2, $3)
                RETURNING global_offset
                """,
                topic_id,
                1200 + i,
                json.dumps(msg).encode("utf-8"),
            )
            global_offsets.append(row["global_offset"])

        # 4. Process like consumer would
        texts = [format_for_embedding(msg) for msg in messages]
        embeddings = await embedding_provider.embed(texts)

        records = [(topic_id, 0, 1200 + i, emb) for i, emb in enumerate(embeddings)]
        await store_embeddings_batch(records)

        # 5. Update consumer offset
        await update_consumer_offset(consumer_id, max(global_offsets))

        # 6. Verify final state
        final_offset = await get_consumer_offset(consumer_id)
        assert final_offset == max(global_offsets)


@pytest.mark.e2e
class TestMCPToolsE2E:
    """End-to-end tests for MCP tools."""

    async def test_find_and_replay_workflow(
        self, patched_db_pool, seeded_embeddings, patched_provider
    ):
        """Test typical workflow: find similar -> replay causal chain.

        This simulates how an agent would use the tools:
        1. Search for relevant decisions
        2. Use the offset to replay what happened before
        """
        import json

        from hippocampus.mcp_server import call_tool

        # Step 1: Find similar decisions
        find_result = await call_tool(
            "find_similar",
            {"query": "authentication", "limit": 5},
        )

        find_data = json.loads(find_result[0].text)
        assert len(find_data) > 0

        # Get the global_offset of the first result
        anchor_offset = find_data[0]["global_offset"]

        # Step 2: Replay the causal chain leading to this decision
        replay_result = await call_tool(
            "replay_causal_chain",
            {"anchor_offset": anchor_offset, "lookback": 10},
        )

        replay_data = json.loads(replay_result[0].text)
        assert len(replay_data) > 0

        # Verify chronological order
        offsets = [item["global_offset"] for item in replay_data]
        assert offsets == sorted(offsets)

    async def test_what_touched_and_replay_topic(
        self, patched_db_pool, seeded_messages, patched_provider
    ):
        """Test workflow: what_touched -> replay_topic.

        Simulates finding changes to a file, then viewing the full session.
        """
        import json

        from hippocampus.mcp_server import call_tool

        # Step 1: Find what touched auth.py
        touched_result = await call_tool(
            "what_touched",
            {"anchor": "auth.py", "limit": 10},
        )

        touched_data = json.loads(touched_result[0].text)
        assert len(touched_data) > 0

        # Get topic from first result
        topic = touched_data[0]["topic"]

        # Step 2: Replay that topic
        replay_result = await call_tool(
            "replay_topic",
            {"topic_name": topic, "limit": 100},
        )

        replay_data = json.loads(replay_result[0].text)
        assert len(replay_data) > 0

    async def test_temporal_context_investigation(
        self, patched_db_pool, seeded_messages, patched_provider
    ):
        """Test workflow: find issue -> check temporal context.

        Simulates investigating what was happening when an issue occurred.
        """
        import json

        from hippocampus.mcp_server import call_tool

        # Get a global_offset from seeded messages
        mid_msg = seeded_messages["messages"][1]
        center_offset = mid_msg["global_offset"]

        # Get temporal context
        context_result = await call_tool(
            "temporal_context",
            {"global_offset": center_offset, "window": 5},
        )

        context_data = json.loads(context_result[0].text)
        assert len(context_data) > 0

        # All results should be within window
        for item in context_data:
            assert abs(item["global_offset"] - center_offset) <= 5
