"""MCP Server for Hippocampus.

Exposes semantic and temporal query tools for AI agents.

Tools:
    - find_similar: Semantic search across all agent decisions
    - replay_causal_chain: Walk backward from an anchor point
    - replay_topic: View a single agent's decision history
    - temporal_context: See what all agents were doing at a point in time
    - what_touched: Find decisions affecting a specific file
"""

import json
import logging
from datetime import datetime
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .config import settings
from .db import (
    find_similar as db_find_similar,
)
from .db import (
    hybrid_search as db_hybrid_search,
)
from .db import (
    replay_causal_chain as db_replay_causal_chain,
)
from .db import (
    replay_topic as db_replay_topic,
)
from .db import (
    temporal_context as db_temporal_context,
)
from .db import (
    what_touched as db_what_touched,
)
from .embeddings import get_provider

logger = logging.getLogger(__name__)


def _serialize_result(result: Any) -> str:
    """Serialize query results to JSON string."""

    def default(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, bytes):
            return obj.decode("utf-8", errors="replace")
        return str(obj)

    return json.dumps(result, default=default, indent=2)


# Create the MCP server
server = Server(settings.mcp_server_name)


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="find_similar",
            description=(
                "Find agent decisions matching a query. "
                "Use this to find relevant context, similar problems, or related decisions. "
                "Mode 'semantic' ranks by embedding similarity only; 'hybrid' fuses "
                "embedding similarity with keyword (full-text) matching, which is better "
                "for queries containing exact identifiers like file names or error codes."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query to search for",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return (default: 10)",
                        "default": 10,
                    },
                    "topic_pattern": {
                        "type": "string",
                        "description": "Optional LIKE pattern for topics (e.g., 'decisions.%')",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["semantic", "hybrid"],
                        "description": "Retrieval mode (default: 'hybrid')",
                        "default": "hybrid",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="replay_causal_chain",
            description=(
                "Replay the events that led to a specific point. "
                "Follows explicit causal links (decisions that declared a parent) when "
                "they exist; otherwise falls back to the events immediately preceding "
                "the anchor in time. Each event's `retrieval` field says which you got "
                "('causal_graph' or 'temporal_window'). "
                "Returns events in chronological order (oldest first)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "anchor_offset": {
                        "type": "integer",
                        "description": "The global_offset to walk back from",
                    },
                    "lookback": {
                        "type": "integer",
                        "description": "Number of events to retrieve (default: 10)",
                        "default": 10,
                    },
                    "topic_pattern": {
                        "type": "string",
                        "description": "Optional LIKE pattern to filter topics",
                    },
                },
                "required": ["anchor_offset"],
            },
        ),
        Tool(
            name="replay_topic",
            description=(
                "View all decisions from a specific agent/topic. "
                "Use this to see one agent's complete decision history."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "topic_name": {
                        "type": "string",
                        "description": "Exact topic name (e.g., 'decisions.cline.session_123')",
                    },
                    "from_offset": {
                        "type": "integer",
                        "description": "Starting offset (default: 0 for beginning)",
                        "default": 0,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max messages to return (default: 100)",
                        "default": 100,
                    },
                },
                "required": ["topic_name"],
            },
        ),
        Tool(
            name="temporal_context",
            description=(
                "See what all agents were doing around a specific point in time. "
                "Use this to understand the broader context of an event."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "global_offset": {
                        "type": "integer",
                        "description": "The center point (global_offset) to examine",
                    },
                    "window": {
                        "type": "integer",
                        "description": "Events before and after (default: 10, total = window*2+1)",
                        "default": 10,
                    },
                },
                "required": ["global_offset"],
            },
        ),
        Tool(
            name="what_touched",
            description=(
                "Find all decisions that affected a specific file or location. "
                "Use this to understand the history of changes to a file."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "anchor": {
                        "type": "string",
                        "description": "File path or location to search for (partial match)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default: 20)",
                        "default": 20,
                    },
                },
                "required": ["anchor"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    try:
        if name == "find_similar":
            result = await _find_similar(
                query=arguments["query"],
                limit=arguments.get("limit", 10),
                topic_pattern=arguments.get("topic_pattern"),
                mode=arguments.get("mode", "hybrid"),
            )
        elif name == "replay_causal_chain":
            result = await db_replay_causal_chain(
                anchor_offset=arguments["anchor_offset"],
                lookback=arguments.get("lookback", 10),
                topic_pattern=arguments.get("topic_pattern"),
            )
        elif name == "replay_topic":
            result = await db_replay_topic(
                topic_name=arguments["topic_name"],
                from_offset=arguments.get("from_offset", 0),
                limit=arguments.get("limit", 100),
            )
        elif name == "temporal_context":
            result = await db_temporal_context(
                global_offset=arguments["global_offset"],
                window=arguments.get("window", 10),
            )
        elif name == "what_touched":
            result = await db_what_touched(
                anchor=arguments["anchor"],
                limit=arguments.get("limit", 20),
            )
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

        return [TextContent(type="text", text=_serialize_result(result))]

    except Exception as e:
        logger.exception(f"Error in tool {name}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def _find_similar(
    query: str,
    limit: int = 10,
    topic_pattern: str | None = None,
    mode: str = "hybrid",
) -> list[dict]:
    """Find similar messages (wraps db query with embedding generation)."""
    # Generate embedding for the query
    provider = get_provider()
    query_embedding = await provider.embed_one(query)

    if mode == "hybrid":
        return await db_hybrid_search(
            query_text=query,
            query_embedding=query_embedding,
            limit=limit,
            topic_pattern=topic_pattern,
        )
    return await db_find_similar(
        query_embedding=query_embedding,
        limit=limit,
        topic_pattern=topic_pattern,
    )


async def run_server():
    """Run the MCP server via stdio."""
    logger.info(f"Starting MCP server '{settings.mcp_server_name}'")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())
