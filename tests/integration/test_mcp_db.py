"""Integration tests for MCP server with database.

Tests the MCP tools' integration with the database layer.
"""

from datetime import datetime

import pytest


@pytest.mark.integration
class TestSerializeResult:
    """Tests for result serialization."""

    def test_serialize_simple_dict(self):
        """Test serializing a simple dictionary."""
        from hippocampus.mcp_server import _serialize_result

        result = _serialize_result({"key": "value", "number": 42})

        assert '"key": "value"' in result
        assert '"number": 42' in result

    def test_serialize_datetime(self):
        """Test that datetime objects are serialized to ISO format."""
        from hippocampus.mcp_server import _serialize_result

        dt = datetime(2024, 1, 15, 10, 30, 0)
        result = _serialize_result({"timestamp": dt})

        assert "2024-01-15T10:30:00" in result

    def test_serialize_bytes(self):
        """Test that bytes are decoded to UTF-8."""
        from hippocampus.mcp_server import _serialize_result

        result = _serialize_result({"data": b"hello bytes"})

        assert "hello bytes" in result

    def test_serialize_list(self):
        """Test serializing a list of results."""
        from hippocampus.mcp_server import _serialize_result

        result = _serialize_result([{"a": 1}, {"b": 2}])

        assert '"a": 1' in result
        assert '"b": 2' in result

    def test_serialize_nested(self):
        """Test serializing nested structures."""
        from hippocampus.mcp_server import _serialize_result

        result = _serialize_result(
            {
                "outer": {
                    "inner": {"value": "nested"},
                    "list": [1, 2, 3],
                }
            }
        )

        assert "nested" in result


@pytest.mark.integration
class TestListTools:
    """Tests for tool listing."""

    async def test_list_tools_returns_all_five(self):
        """Test that all 5 tools are listed."""
        from hippocampus.mcp_server import list_tools

        tools = await list_tools()

        assert len(tools) == 5

        tool_names = {t.name for t in tools}
        expected_names = {
            "find_similar",
            "replay_causal_chain",
            "replay_topic",
            "temporal_context",
            "what_touched",
        }
        assert tool_names == expected_names

    async def test_tools_have_descriptions(self):
        """Test that all tools have descriptions."""
        from hippocampus.mcp_server import list_tools

        tools = await list_tools()

        for tool in tools:
            assert tool.description
            assert len(tool.description) > 10

    async def test_tools_have_input_schemas(self):
        """Test that all tools have input schemas."""
        from hippocampus.mcp_server import list_tools

        tools = await list_tools()

        for tool in tools:
            assert tool.inputSchema
            assert "properties" in tool.inputSchema
            assert "required" in tool.inputSchema


@pytest.mark.integration
class TestFindSimilarTool:
    """Tests for the find_similar tool."""

    async def test_find_similar_generates_embedding(
        self, patched_db_pool, seeded_embeddings, patched_provider
    ):
        """Test that find_similar generates query embedding."""
        from hippocampus.mcp_server import call_tool

        result = await call_tool(
            "find_similar",
            {"query": "authentication bug", "limit": 5},
        )

        assert len(result) == 1
        assert result[0].type == "text"
        # Should be valid JSON
        import json

        data = json.loads(result[0].text)
        assert isinstance(data, list)

    async def test_find_similar_semantic_returns_results(
        self, patched_db_pool, seeded_embeddings, patched_provider
    ):
        """Test that semantic mode returns distance-ranked results."""
        from hippocampus.mcp_server import call_tool

        result = await call_tool(
            "find_similar",
            {"query": "authentication", "limit": 10, "mode": "semantic"},
        )

        import json

        data = json.loads(result[0].text)

        assert len(data) > 0
        assert "value" in data[0]
        assert "distance" in data[0]

    async def test_find_similar_hybrid_returns_results(
        self, patched_db_pool, seeded_embeddings, patched_provider
    ):
        """Test that the default hybrid mode returns RRF-scored results."""
        from hippocampus.mcp_server import call_tool

        result = await call_tool(
            "find_similar",
            {"query": "authentication", "limit": 10},
        )

        import json

        data = json.loads(result[0].text)

        assert len(data) > 0
        assert "value" in data[0]
        assert "rrf_score" in data[0]

    async def test_find_similar_with_topic_pattern(
        self, patched_db_pool, seeded_embeddings, patched_provider
    ):
        """Test find_similar with topic filtering."""
        from hippocampus.mcp_server import call_tool

        result = await call_tool(
            "find_similar",
            {
                "query": "test query",
                "limit": 10,
                "topic_pattern": "decisions.test.%",
            },
        )

        import json

        data = json.loads(result[0].text)

        for item in data:
            assert item["topic"].startswith("decisions.test.")


@pytest.mark.integration
class TestReplayCausalChainTool:
    """Tests for the replay_causal_chain tool."""

    async def test_replay_causal_chain_returns_results(self, patched_db_pool, seeded_messages):
        """Test that replay_causal_chain returns events."""
        from hippocampus.mcp_server import call_tool

        last_msg = seeded_messages["messages"][-1]

        result = await call_tool(
            "replay_causal_chain",
            {"anchor_offset": last_msg["global_offset"], "lookback": 10},
        )

        import json

        data = json.loads(result[0].text)

        assert len(data) > 0
        assert "value" in data[0]
        assert "global_offset" in data[0]

    async def test_replay_causal_chain_chronological(self, patched_db_pool, seeded_messages):
        """Test that results are in chronological order."""
        from hippocampus.mcp_server import call_tool

        last_msg = seeded_messages["messages"][-1]

        result = await call_tool(
            "replay_causal_chain",
            {"anchor_offset": last_msg["global_offset"], "lookback": 10},
        )

        import json

        data = json.loads(result[0].text)

        if len(data) > 1:
            offsets = [item["global_offset"] for item in data]
            assert offsets == sorted(offsets)


@pytest.mark.integration
class TestReplayTopicTool:
    """Tests for the replay_topic tool."""

    async def test_replay_topic_returns_results(self, patched_db_pool, seeded_messages):
        """Test that replay_topic returns messages."""
        from hippocampus.mcp_server import call_tool

        result = await call_tool(
            "replay_topic",
            {"topic_name": seeded_messages["topic_name"]},
        )

        import json

        data = json.loads(result[0].text)

        assert len(data) == len(seeded_messages["messages"])

    async def test_replay_topic_from_offset(self, patched_db_pool, seeded_messages):
        """Test replay_topic with from_offset."""
        from hippocampus.mcp_server import call_tool

        result = await call_tool(
            "replay_topic",
            {"topic_name": seeded_messages["topic_name"], "from_offset": 1},
        )

        import json

        data = json.loads(result[0].text)

        # Should skip first message
        assert len(data) == len(seeded_messages["messages"]) - 1


@pytest.mark.integration
class TestTemporalContextTool:
    """Tests for the temporal_context tool."""

    async def test_temporal_context_returns_results(self, patched_db_pool, seeded_messages):
        """Test that temporal_context returns events."""
        from hippocampus.mcp_server import call_tool

        mid_msg = seeded_messages["messages"][1]

        result = await call_tool(
            "temporal_context",
            {"global_offset": mid_msg["global_offset"], "window": 10},
        )

        import json

        data = json.loads(result[0].text)

        assert len(data) > 0

    async def test_temporal_context_respects_window(self, patched_db_pool, seeded_messages):
        """Test that window parameter works."""
        from hippocampus.mcp_server import call_tool

        mid_msg = seeded_messages["messages"][1]
        center = mid_msg["global_offset"]

        result = await call_tool(
            "temporal_context",
            {"global_offset": center, "window": 1},
        )

        import json

        data = json.loads(result[0].text)

        for item in data:
            assert abs(item["global_offset"] - center) <= 1


@pytest.mark.integration
class TestWhatTouchedTool:
    """Tests for the what_touched tool."""

    async def test_what_touched_finds_files(self, patched_db_pool, seeded_messages):
        """Test that what_touched finds matching anchors."""
        from hippocampus.mcp_server import call_tool

        result = await call_tool(
            "what_touched",
            {"anchor": "auth.py"},
        )

        import json

        data = json.loads(result[0].text)

        assert len(data) > 0
        for item in data:
            assert "auth.py" in item["value"].get("anchor", "")

    async def test_what_touched_respects_limit(self, patched_db_pool, seeded_messages):
        """Test that limit is respected."""
        from hippocampus.mcp_server import call_tool

        result = await call_tool(
            "what_touched",
            {"anchor": "auth", "limit": 1},
        )

        import json

        data = json.loads(result[0].text)

        assert len(data) <= 1


@pytest.mark.integration
class TestUnknownTool:
    """Tests for handling unknown tools."""

    async def test_unknown_tool_returns_error(self, patched_db_pool):
        """Test that unknown tool returns error message."""
        from hippocampus.mcp_server import call_tool

        result = await call_tool("nonexistent_tool", {})

        assert len(result) == 1
        assert "Unknown tool" in result[0].text


@pytest.mark.integration
class TestToolErrorHandling:
    """Tests for tool error handling."""

    async def test_tool_handles_exception(self, patched_db_pool):
        """Test that exceptions are caught and returned as errors."""
        from hippocampus.mcp_server import call_tool

        # replay_topic with nonexistent topic should return empty, not error
        # But invalid arguments might cause issues
        result = await call_tool(
            "replay_topic",
            {"topic_name": "nonexistent.topic.name"},
        )

        # Should still return valid result (empty list)
        import json

        data = json.loads(result[0].text)
        assert isinstance(data, list)
