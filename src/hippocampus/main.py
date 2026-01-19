"""Hippocampus: Temporal RAG for AI Agents.

Entry points:
    - hippocampus server: Run MCP server (for Claude Desktop integration)
    - hippocampus consumer: Run embedding consumer
    - hippocampus both: Run both (dev mode)
"""

import argparse
import asyncio
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def run_both():
    """Run both consumer and MCP server concurrently."""
    from .consumer import EmbeddingConsumer
    from .mcp_server import run_server

    consumer = EmbeddingConsumer()

    async def consumer_task():
        await consumer.start(mode="batch")

    # Run both concurrently
    await asyncio.gather(
        consumer_task(),
        run_server(),
    )


async def run_server():
    """Run the MCP server only."""
    from .mcp_server import run_server as _run_server

    await _run_server()


async def run_consumer(mode: str = "batch"):
    """Run the embedding consumer only."""
    from .consumer import EmbeddingConsumer

    consumer = EmbeddingConsumer()
    await consumer.start(mode=mode)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Hippocampus: Temporal RAG for AI Agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    hippocampus server       # Run MCP server (for Claude Desktop)
    hippocampus consumer     # Run embedding consumer
    hippocampus both         # Run both (dev mode)
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Server command
    subparsers.add_parser("server", help="Run MCP server")

    # Consumer command
    consumer_parser = subparsers.add_parser("consumer", help="Run embedding consumer")
    consumer_parser.add_argument(
        "--mode",
        choices=["sync", "batch"],
        default="batch",
        help="Consumer mode (default: batch)",
    )

    # Both command
    subparsers.add_parser("both", help="Run both server and consumer")

    args = parser.parse_args()

    if args.command == "server":
        asyncio.run(run_server())
    elif args.command == "consumer":
        asyncio.run(run_consumer(mode=args.mode))
    elif args.command == "both":
        asyncio.run(run_both())
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
