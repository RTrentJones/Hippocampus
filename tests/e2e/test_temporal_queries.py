"""End-to-end tests for temporal query functionality.

Tests the temporal traversal capabilities that make Hippocampus different
from standard RAG - the ability to walk backward through the event log.
"""

import json

import pytest


@pytest.mark.e2e
class TestCausalChainTraversal:
    """Tests for causal chain (walking backward through events)."""

    async def test_replay_causal_chain_preserves_order(
        self, patched_db_pool, seeded_topic, patched_provider
    ):
        """Verify causal chain returns events in chronological order."""
        from hippocampus.db import replay_causal_chain

        topic_id = seeded_topic["topic_id"]

        # Insert a sequence of causally related messages
        messages = [
            {
                "context": "Received bug report about login failure",
                "reasoning": "Need to investigate auth flow",
                "action": "Check error logs",
            },
            {
                "context": "Found NullPointerException in logs",
                "reasoning": "Variable not initialized",
                "action": "Trace variable source",
            },
            {
                "context": "Found the uninitialized variable",
                "reasoning": "Missing null check after refactor",
                "action": "Add null check",
            },
            {
                "context": "Fix applied",
                "reasoning": "Tests pass now",
                "action": "Create PR for review",
            },
        ]

        global_offsets = []
        for i, msg in enumerate(messages):
            row = await patched_db_pool.fetchrow(
                """
                INSERT INTO kafka.messages (topic_id, partition_id, partition_offset, value)
                VALUES ($1, 0, $2, $3)
                RETURNING global_offset
                """,
                topic_id,
                2000 + i,
                json.dumps(msg).encode("utf-8"),
            )
            global_offsets.append(row["global_offset"])

        # Get causal chain from the last message
        anchor_offset = global_offsets[-1]
        results = await replay_causal_chain(anchor_offset, lookback=10)

        # Verify chronological order (oldest first)
        result_offsets = [r["global_offset"] for r in results]
        assert result_offsets == sorted(result_offsets)

        # Verify we got the expected messages
        assert len(results) >= len(messages)

    async def test_causal_chain_with_lookback_limit(
        self, patched_db_pool, seeded_topic, patched_provider
    ):
        """Test that lookback parameter limits results correctly."""
        from hippocampus.db import replay_causal_chain

        topic_id = seeded_topic["topic_id"]

        # Insert 10 messages
        global_offsets = []
        for i in range(10):
            row = await patched_db_pool.fetchrow(
                """
                INSERT INTO kafka.messages (topic_id, partition_id, partition_offset, value)
                VALUES ($1, 0, $2, $3)
                RETURNING global_offset
                """,
                topic_id,
                2100 + i,
                json.dumps({"context": f"Message {i}", "action": f"Action {i}"}).encode("utf-8"),
            )
            global_offsets.append(row["global_offset"])

        # Request only last 3
        anchor_offset = global_offsets[-1]
        results = await replay_causal_chain(anchor_offset, lookback=3)

        assert len(results) == 3


@pytest.mark.e2e
class TestTemporalContextQueries:
    """Tests for temporal context (cross-agent activity at a point in time)."""

    async def test_temporal_context_shows_concurrent_activity(
        self, patched_db_pool, patched_provider
    ):
        """Test that temporal_context shows events from multiple topics."""
        from hippocampus.db import temporal_context

        # Create two topics (simulating different agents)
        topic1_id = await patched_db_pool.fetchval(
            "INSERT INTO kafka.topics (name) VALUES ($1) RETURNING id",
            "decisions.agent1.session_001",
        )
        topic2_id = await patched_db_pool.fetchval(
            "INSERT INTO kafka.topics (name) VALUES ($1) RETURNING id",
            "decisions.agent2.session_001",
        )

        # Insert interleaved messages
        offsets = []
        for i in range(6):
            topic_id = topic1_id if i % 2 == 0 else topic2_id
            row = await patched_db_pool.fetchrow(
                """
                INSERT INTO kafka.messages (topic_id, partition_id, partition_offset, value)
                VALUES ($1, 0, $2, $3)
                RETURNING global_offset
                """,
                topic_id,
                i,
                json.dumps({"context": f"Agent{1 if i % 2 == 0 else 2} action {i}"}).encode("utf-8"),
            )
            offsets.append(row["global_offset"])

        # Query temporal context around the middle
        center = offsets[len(offsets) // 2]
        results = await temporal_context(center, window=3)

        # Should have results from both topics
        topics = {r["topic"] for r in results}
        assert len(topics) >= 2

    async def test_temporal_context_window_boundaries(
        self, patched_db_pool, seeded_topic, patched_provider
    ):
        """Test that window boundaries are respected."""
        from hippocampus.db import temporal_context

        topic_id = seeded_topic["topic_id"]

        # Insert messages with known offsets
        offsets = []
        for i in range(20):
            row = await patched_db_pool.fetchrow(
                """
                INSERT INTO kafka.messages (topic_id, partition_id, partition_offset, value)
                VALUES ($1, 0, $2, $3)
                RETURNING global_offset
                """,
                topic_id,
                2200 + i,
                json.dumps({"context": f"Message {i}"}).encode("utf-8"),
            )
            offsets.append(row["global_offset"])

        # Pick a center point
        center = offsets[10]

        # Query with window=2
        results = await temporal_context(center, window=2)

        # All results should be within [-2, +2] of center
        for result in results:
            assert abs(result["global_offset"] - center) <= 2


@pytest.mark.e2e
class TestTopicReplay:
    """Tests for replaying a single topic's history."""

    async def test_replay_single_agent_session(
        self, patched_db_pool, patched_provider
    ):
        """Test replaying a complete agent session."""
        from hippocampus.db import replay_topic

        # Create a topic for a specific session
        topic_name = "decisions.cline.session_test_001"
        topic_id = await patched_db_pool.fetchval(
            "INSERT INTO kafka.topics (name) VALUES ($1) RETURNING id",
            topic_name,
        )

        # Insert session messages in order
        session_messages = [
            {"context": "Session start", "action": "Initialize workspace"},
            {"context": "Reading codebase", "action": "Explore src/"},
            {"context": "Found issue", "action": "Create fix"},
            {"context": "Testing fix", "action": "Run tests"},
            {"context": "Session end", "action": "Commit changes"},
        ]

        for i, msg in enumerate(session_messages):
            await patched_db_pool.execute(
                """
                INSERT INTO kafka.messages (topic_id, partition_id, partition_offset, value)
                VALUES ($1, 0, $2, $3)
                """,
                topic_id,
                i,
                json.dumps(msg).encode("utf-8"),
            )

        # Replay the session
        results = await replay_topic(topic_name, from_offset=0, limit=100)

        # Should get all messages in order
        assert len(results) == len(session_messages)

        # Verify order
        for i, result in enumerate(results):
            assert result["partition_offset"] == i

    async def test_replay_topic_pagination(
        self, patched_db_pool, patched_provider
    ):
        """Test paginating through a topic."""
        from hippocampus.db import replay_topic

        topic_name = "decisions.test.pagination"
        topic_id = await patched_db_pool.fetchval(
            "INSERT INTO kafka.topics (name) VALUES ($1) RETURNING id",
            topic_name,
        )

        # Insert 10 messages
        for i in range(10):
            await patched_db_pool.execute(
                """
                INSERT INTO kafka.messages (topic_id, partition_id, partition_offset, value)
                VALUES ($1, 0, $2, $3)
                """,
                topic_id,
                i,
                json.dumps({"message": f"Message {i}"}).encode("utf-8"),
            )

        # Page 1
        page1 = await replay_topic(topic_name, from_offset=0, limit=5)
        assert len(page1) == 5
        assert page1[0]["partition_offset"] == 0
        assert page1[-1]["partition_offset"] == 4

        # Page 2
        page2 = await replay_topic(topic_name, from_offset=5, limit=5)
        assert len(page2) == 5
        assert page2[0]["partition_offset"] == 5
        assert page2[-1]["partition_offset"] == 9


@pytest.mark.e2e
class TestFileHistoryTracking:
    """Tests for tracking what decisions affected specific files."""

    async def test_what_touched_tracks_file_history(
        self, patched_db_pool, patched_provider
    ):
        """Test finding all decisions that touched a file."""
        from hippocampus.db import what_touched

        topic_id = await patched_db_pool.fetchval(
            "INSERT INTO kafka.topics (name) VALUES ($1) RETURNING id",
            "decisions.test.file_tracking",
        )

        # Messages touching different files
        messages = [
            {"anchor": "src/api/routes.py", "action": "Add new endpoint"},
            {"anchor": "src/api/routes.py", "action": "Fix route handler"},
            {"anchor": "src/models/user.py", "action": "Add field"},
            {"anchor": "src/api/routes.py", "action": "Update response format"},
            {"anchor": "src/utils/helpers.py", "action": "Refactor helper"},
        ]

        for i, msg in enumerate(messages):
            await patched_db_pool.execute(
                """
                INSERT INTO kafka.messages (topic_id, partition_id, partition_offset, value)
                VALUES ($1, 0, $2, $3)
                """,
                topic_id,
                3000 + i,
                json.dumps(msg).encode("utf-8"),
            )

        # Find all changes to routes.py
        results = await what_touched("routes.py", limit=10)

        # Should find 3 results
        assert len(results) == 3

        # All should reference routes.py
        for result in results:
            assert "routes.py" in result["value"]["anchor"]

    async def test_what_touched_newest_first(
        self, patched_db_pool, patched_provider
    ):
        """Test that what_touched returns results newest first."""
        from hippocampus.db import what_touched

        topic_id = await patched_db_pool.fetchval(
            "INSERT INTO kafka.topics (name) VALUES ($1) RETURNING id",
            "decisions.test.newest_first",
        )

        # Insert messages with same file
        for i in range(5):
            await patched_db_pool.execute(
                """
                INSERT INTO kafka.messages (topic_id, partition_id, partition_offset, value)
                VALUES ($1, 0, $2, $3)
                """,
                topic_id,
                3100 + i,
                json.dumps({"anchor": "src/target_file.py", "action": f"Change {i}"}).encode("utf-8"),
            )

        results = await what_touched("target_file.py", limit=10)

        # Should be in descending order (newest first)
        offsets = [r["global_offset"] for r in results]
        assert offsets == sorted(offsets, reverse=True)
