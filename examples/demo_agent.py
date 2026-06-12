#!/usr/bin/env python3
"""Demo: Simulate an AI agent publishing decisions.

This script:
1. Produces a series of agent decisions to Kafka (via pg_kafka)
2. Simulates a debugging scenario where we trace back to find a root cause

Usage:
    python examples/demo_agent.py
"""

import json
import time
import uuid
from datetime import datetime

from kafka import KafkaProducer

# Configuration
KAFKA_SERVERS = "localhost:9092"
TOPIC = "decisions.demo.session_001"


def create_producer() -> KafkaProducer:
    """Create a Kafka producer."""
    return KafkaProducer(
        bootstrap_servers=KAFKA_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
    )


def publish_decision(
    producer: KafkaProducer,
    context: str,
    reasoning: str,
    action: str,
    anchor: str | None = None,
    parent_id: str | None = None,
) -> str:
    """Publish a decision and return its ID."""
    decision_id = str(uuid.uuid4())

    message = {
        "id": decision_id,
        "parent_id": parent_id,
        "context": context,
        "reasoning": reasoning,
        "action": action,
        "anchor": anchor,
        "timestamp": datetime.now().isoformat(),
    }

    producer.send(TOPIC, key=decision_id, value=message)
    producer.flush()

    print(f"Published: {action[:50]}...")
    return decision_id


def main():
    """Run the demo scenario."""
    print("=" * 60)
    print("Hippocampus Demo: AI Agent Decision Trail")
    print("=" * 60)
    print()

    producer = create_producer()

    # Scenario: Agent is debugging a server crash
    # The decisions form a causal chain

    print("Simulating agent debugging session...\n")

    # Decision 1: Start investigation
    d1 = publish_decision(
        producer,
        context="User reported server crash at 3:47 PM",
        reasoning="Need to check error logs first to understand the failure mode",
        action="Open server logs from /var/log/app/error.log",
        anchor="/var/log/app/error.log",
    )
    time.sleep(0.5)

    # Decision 2: Found error, investigate further
    d2 = publish_decision(
        producer,
        context="Found NullPointerException in AuthService.validateToken()",
        reasoning=(
            "The stack trace shows the error originates from line 127. "
            "Need to check recent changes to this file."
        ),
        action="View git blame for AuthService.java around line 127",
        anchor="src/main/java/AuthService.java:127",
        parent_id=d1,
    )
    time.sleep(0.5)

    # Decision 3: Found the commit
    d3 = publish_decision(
        producer,
        context=(
            "Git blame shows commit abc123 by developer@company.com modified this line yesterday"
        ),
        reasoning=(
            "The commit message says 'Optimize token validation'. "
            "This might have introduced a null check issue."
        ),
        action="View full diff for commit abc123",
        anchor="commit:abc123",
        parent_id=d2,
    )
    time.sleep(0.5)

    # Decision 4: Found the bug
    d4 = publish_decision(
        producer,
        context=(
            "Commit abc123 removed a null check for userSession.getToken() "
            "as part of 'optimization'"
        ),
        reasoning=(
            "The removed null check was protecting against expired sessions. When a "
            "session expires, getToken() returns null, causing the NPE."
        ),
        action="Create fix: restore null check and add unit test",
        anchor="src/main/java/AuthService.java:125-130",
        parent_id=d3,
    )
    time.sleep(0.5)

    # Decision 5: Fix applied
    d5 = publish_decision(
        producer,
        context="Fix implemented and tested locally",
        reasoning=(
            "Unit test confirms the fix handles null tokens correctly. Ready for code review."
        ),
        action="Create pull request with fix and regression test",
        anchor="PR:fix-auth-npe-456",
        parent_id=d4,
    )

    print()
    print("=" * 60)
    print(f"Demo complete! Published 5 causally-linked decisions (last: {d5})")
    print("to topic:")
    print(f"  {TOPIC}")
    print()
    print("Now you can query this with Hippocampus MCP tools:")
    print()
    print('  find_similar("NullPointerException authentication")')
    print("  replay_causal_chain(<last_offset>, lookback=5)")
    print(f'  replay_topic("{TOPIC}")')
    print('  what_touched("AuthService.java")')
    print("=" * 60)


if __name__ == "__main__":
    main()
