"""Synthetic agent-trajectory dataset generator for retrieval evals.

Generates sessions of agent decisions where a *planted causal chain* leads
from a root cause to a failure event, surrounded by distractors that are
lexically similar to the failure but causally unrelated. This reproduces the
failure mode temporal RAG exists to fix: pure semantic similarity retrieves
look-alike events instead of the actual cause.

Generation is deterministic for a given seed so results are reproducible and
the smoke eval can run in CI without an LLM.
"""

import random
from dataclasses import dataclass, field

# =============================================================================
# Data model
# =============================================================================


@dataclass
class Decision:
    """One agent decision in a session."""

    id: str
    context: str
    reasoning: str
    action: str
    anchor: str | None = None
    parent_id: str | None = None
    # Eval bookkeeping (not part of the published payload)
    role: str = "noise"  # "chain" | "distractor" | "noise"

    def payload(self) -> dict:
        """The message payload as a producer would publish it."""
        message = {
            "id": self.id,
            "context": self.context,
            "reasoning": self.reasoning,
            "action": self.action,
        }
        if self.anchor:
            message["anchor"] = self.anchor
        if self.parent_id:
            message["parent_id"] = self.parent_id
        return message


@dataclass
class EvalCase:
    """Ground truth for one session: find the root cause of the failure."""

    session: str
    query: str  # natural-language query an engineer would ask
    failure_id: str  # the anchor event (end of the chain)
    root_cause_id: str  # the decision retrieval should surface
    chain_ids: list[str]  # full causal chain, root first


@dataclass
class Session:
    """An ordered sequence of decisions published to one topic."""

    name: str
    decisions: list[Decision] = field(default_factory=list)


# =============================================================================
# Scenario templates
#
# Each scenario defines: a causal chain (root cause -> ... -> failure), a
# query phrased the way an engineer would ask it, and distractor templates
# that share the failure's vocabulary without being causally related.
# =============================================================================

SCENARIOS: list[dict] = [
    {
        "theme": "auth-crash",
        "chain": [
            {
                "context": "Reviewing token validation for performance in {service}",
                "reasoning": "The null check on userSession.getToken() looks redundant",
                "action": "Remove null check from validateToken as an optimization",
                "anchor": "src/{service}/AuthService.java:127",
            },
            {
                "context": "Deploying the token validation optimization to production",
                "reasoning": "Change is small and passed unit tests",
                "action": "Deploy {service} release with validateToken change",
                "anchor": "deploy/{service}",
            },
            {
                "context": "Users with expired sessions are hitting errors in {service}",
                "reasoning": "Expired sessions return a null token, which is now dereferenced",
                "action": "Observe spike in errors for requests with expired sessions",
                "anchor": "src/{service}/AuthService.java",
            },
            {
                "context": "{service} crashed with NullPointerException in validateToken",
                "reasoning": "Stack trace points at AuthService.validateToken line 127",
                "action": "Page on-call: production crash with NullPointerException",
                "anchor": "src/{service}/AuthService.java:127",
            },
        ],
        "query": "Why did the server crash with a NullPointerException in token validation?",
        "distractors": [
            {
                "context": "Triaging an old NullPointerException report from last month in {other}",
                "reasoning": "That crash was caused by a missing config file, unrelated to auth",
                "action": "Close stale NullPointerException ticket as resolved",
                "anchor": "tickets/npe-{n}",
            },
            {
                "context": "Writing documentation about common NullPointerException causes",
                "reasoning": "Token validation errors are a frequent support topic",
                "action": "Update troubleshooting guide for token validation crashes",
                "anchor": "docs/troubleshooting.md",
            },
            {
                "context": "A similar crash in {other} was reported during the last release",
                "reasoning": "That NullPointerException came from a third-party SDK bug",
                "action": "Pin the third-party SDK version in {other}",
                "anchor": "src/{other}/build.gradle",
            },
        ],
    },
    {
        "theme": "db-timeout",
        "chain": [
            {
                "context": "Adding a reporting feature that filters orders by customer email",
                "reasoning": "A simple WHERE clause on the orders table should be fine",
                "action": "Ship query filtering orders by email without an index",
                "anchor": "src/reports/orders_query.py:44",
            },
            {
                "context": "Orders table has grown past 50M rows in production",
                "reasoning": "The email filter now sequential-scans the whole table",
                "action": "Observe p95 latency on the reports endpoint climbing",
                "anchor": "dashboards/latency",
            },
            {
                "context": "Reports API returning 504 gateway timeouts under load",
                "reasoning": "Database connections are saturated by the slow report query",
                "action": "Declare incident: reports endpoint timing out",
                "anchor": "src/reports/orders_query.py",
            },
        ],
        "query": "What caused the 504 gateway timeouts on the reports endpoint?",
        "distractors": [
            {
                "context": "Investigating a 504 timeout reported by a single customer in {other}",
                "reasoning": "Their corporate proxy was misconfigured; not our infrastructure",
                "action": "Close customer 504 ticket as external network issue",
                "anchor": "tickets/timeout-{n}",
            },
            {
                "context": "Load testing the new gateway configuration in staging",
                "reasoning": "Timeouts under synthetic load are expected at this tier",
                "action": "Record staging gateway timeout baseline",
                "anchor": "loadtest/gateway",
            },
            {
                "context": "Reviewing database connection pool settings for {other}",
                "reasoning": "Pool size matches vendor recommendations; no change needed",
                "action": "Document connection pool tuning decision for {other}",
                "anchor": "docs/db-tuning.md",
            },
        ],
    },
    {
        "theme": "oom-kill",
        "chain": [
            {
                "context": "Simplifying the cache layer configuration in {service}",
                "reasoning": "Entries are small, so expiry bookkeeping looks like overhead",
                "action": "Disable TTL eviction on the in-memory cache",
                "anchor": "src/{service}/cache.py:88",
            },
            {
                "context": "Memory usage of {service} growing steadily since the last deploy",
                "reasoning": "The cache now grows without bound because nothing evicts entries",
                "action": "Observe RSS climbing on {service} pods",
                "anchor": "dashboards/memory",
            },
            {
                "context": "{service} pods are being OOM-killed every few hours",
                "reasoning": "Kernel kills the process when the unbounded cache exhausts memory",
                "action": "Declare incident: {service} crash-looping from OOM kills",
                "anchor": "src/{service}/cache.py",
            },
        ],
        "query": "Why are the pods getting OOM-killed and crash-looping?",
        "distractors": [
            {
                "context": "An OOM kill in the nightly batch job for {other} last month",
                "reasoning": "Batch job memory spike was caused by an oversized input file",
                "action": "Add input size guard to the {other} batch job",
                "anchor": "src/{other}/batch.py",
            },
            {
                "context": "Capacity planning review for memory headroom across services",
                "reasoning": "Most services run below 60 percent memory utilization",
                "action": "Publish quarterly memory capacity report",
                "anchor": "docs/capacity.md",
            },
            {
                "context": "Tuning JVM heap flags for {other} after a garbage collection alert",
                "reasoning": "Old-gen pressure was transient and resolved after the alert",
                "action": "Revert experimental heap flags in {other}",
                "anchor": "deploy/{other}/jvm.flags",
            },
        ],
    },
    {
        "theme": "dependency-regression",
        "chain": [
            {
                "context": "Routine dependency upgrade sweep for {service}",
                "reasoning": "The HTTP client minor version bump should be backwards compatible",
                "action": "Bump http client library to next minor version",
                "anchor": "src/{service}/requirements.txt",
            },
            {
                "context": "The upgraded client now sends chunked encoding by default",
                "reasoning": "The downstream billing API rejects chunked requests",
                "action": "Observe 500 errors from billing API calls after deploy",
                "anchor": "src/{service}/billing_client.py",
            },
            {
                "context": "Checkout flow failing with 500 errors for all card payments",
                "reasoning": "Every checkout calls the billing API, which rejects our requests",
                "action": "Declare incident: checkout broken with 500 errors",
                "anchor": "src/{service}/checkout.py",
            },
        ],
        "query": "Why is checkout failing with 500 errors on card payments?",
        "distractors": [
            {
                "context": "A burst of 500 errors in {other} traced to a bad canary",
                "reasoning": "Canary was rolled back automatically; errors stopped",
                "action": "Write postmortem for the {other} canary 500 errors",
                "anchor": "postmortems/{other}-canary.md",
            },
            {
                "context": "Customer reported a payment declined at checkout",
                "reasoning": "Card was declined by the issuer; not a system error",
                "action": "Reply to support ticket about declined payment",
                "anchor": "tickets/payment-{n}",
            },
            {
                "context": "Auditing dependency licenses across {other}",
                "reasoning": "Two transitive dependencies changed license terms",
                "action": "Flag license changes for legal review",
                "anchor": "docs/licenses.md",
            },
        ],
    },
]

_SERVICES = ["checkout", "identity", "search", "ledger", "ingest", "gateway"]
_NOISE = [
    {
        "context": "Routine review of open pull requests",
        "reasoning": "Two PRs are approved and waiting on merge",
        "action": "Merge approved pull requests",
        "anchor": None,
    },
    {
        "context": "Updating the team on-call rotation",
        "reasoning": "Next sprint starts Monday",
        "action": "Publish on-call schedule",
        "anchor": None,
    },
    {
        "context": "Renaming variables for clarity in a utility module",
        "reasoning": "Reviewer asked for more descriptive names",
        "action": "Apply rename refactor",
        "anchor": "src/utils/strings.py",
    },
    {
        "context": "Adding a lint rule for import ordering",
        "reasoning": "Recent diffs churn on import order",
        "action": "Enable import sorting in CI",
        "anchor": ".github/workflows/ci.yml",
    },
]


# =============================================================================
# Generator
# =============================================================================


def _fill(
    template: dict, slots: dict, decision_id: str, parent_id: str | None, role: str
) -> Decision:
    def render(text: str | None) -> str | None:
        return text.format(**slots) if text else None

    return Decision(
        id=decision_id,
        context=render(template["context"]) or "",
        reasoning=render(template["reasoning"]) or "",
        action=render(template["action"]) or "",
        anchor=render(template.get("anchor")),
        parent_id=parent_id,
        role=role,
    )


def generate_dataset(
    num_sessions: int = 20,
    distractors_per_session: int = 6,
    seed: int = 42,
) -> tuple[list[Session], list[EvalCase]]:
    """Generate sessions with planted causal chains plus distractors.

    Chain decisions keep their causal order; distractor and noise events are
    interleaved at random positions (never between being published and the
    failure being last is NOT guaranteed — distractors can land after the
    failure too, as in real logs).

    Returns:
        (sessions, cases) — one eval case per session.
    """
    rng = random.Random(seed)
    sessions: list[Session] = []
    cases: list[EvalCase] = []

    for i in range(num_sessions):
        scenario = SCENARIOS[i % len(SCENARIOS)]
        service, other = rng.sample(_SERVICES, 2)
        slots = {"service": service, "other": other, "n": rng.randint(100, 999)}
        session_name = f"eval.s{i:03d}.{scenario['theme']}"

        chain: list[Decision] = []
        for step, template in enumerate(scenario["chain"]):
            decision_id = f"s{i:03d}-chain-{step}"
            parent_id = chain[-1].id if chain else None
            chain.append(_fill(template, slots, decision_id, parent_id, role="chain"))

        extras: list[Decision] = []
        for d in range(distractors_per_session):
            if d % 2 == 0:
                template = scenario["distractors"][d // 2 % len(scenario["distractors"])]
                role = "distractor"
            else:
                template = _NOISE[d // 2 % len(_NOISE)]
                role = "noise"
            extras.append(_fill(template, slots, f"s{i:03d}-x{d}", None, role=role))

        # Interleave: chain keeps relative order; extras land at random slots
        decisions = list(chain)
        for extra in extras:
            decisions.insert(rng.randint(0, len(decisions)), extra)

        sessions.append(Session(name=session_name, decisions=decisions))
        cases.append(
            EvalCase(
                session=session_name,
                query=scenario["query"],
                failure_id=chain[-1].id,
                root_cause_id=chain[0].id,
                chain_ids=[c.id for c in chain],
            )
        )

    return sessions, cases
