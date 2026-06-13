"""Run the retrieval eval: four arms + optional LLM judge.

Usage:
    # Smoke run with mock embeddings (plumbing + causal arm only;
    # semantic numbers are meaningless with the mock provider)
    python -m evals.run --provider mock --sessions 8

    # Real run with local sentence-transformers embeddings
    python -m evals.run --provider local --sessions 40

    # Add the end-to-end LLM judge (requires ANTHROPIC_API_KEY)
    python -m evals.run --provider local --judge

Requires a database with the Hippocampus migrations applied; set
EVAL_DATABASE_URL (falls back to TEST_DATABASE_URL, then settings).
"""

import argparse
import asyncio
import json
import logging
import os
import sys

from hippocampus import db
from hippocampus.config import settings

from . import arms, metrics
from . import seed as seed_module
from .dataset import Decision, EvalCase, Session, generate_dataset
from .providers import make_provider
from .seed import clean_eval_data, seed_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("evals")

ARM_NAMES = [
    "semantic",
    "hybrid",
    "temporal",
    "causal",
    "anchor_causal_hybrid",
    "anchor_causal_semantic",
]


async def run_arms(
    case: EvalCase,
    provider,
    offsets: dict[str, int],
    k: int,
    scope: str = "session",
) -> dict[str, list[str]]:
    """Run every retrieval arm for one case; returns arm -> ranked ids.

    scope="session" restricts retrieval to the incident's own topic — the
    realistic setting (you are debugging a specific agent run, and the
    distractors live in the same log). scope="global" searches every topic;
    with repeated scenario templates across sessions, near-identical failure
    events in *other* sessions become an additional confounder for the
    query-driven arms.
    """
    anchor = offsets[case.failure_id]
    pattern = (
        seed_module.EVAL_TOPIC_PREFIX + case.session if scope == "session" else arms.TOPIC_PATTERN
    )
    return {
        "semantic": await arms.semantic_arm(case.query, provider, k, pattern),
        "hybrid": await arms.hybrid_arm(case.query, provider, k, pattern),
        "temporal": await arms.temporal_arm(anchor, k, pattern),
        "causal": await arms.causal_arm(anchor, k),
        "anchor_causal_hybrid": await arms.anchor_causal_arm(
            case.query, provider, k, pattern, anchor_mode="hybrid"
        ),
        "anchor_causal_semantic": await arms.anchor_causal_arm(
            case.query, provider, k, pattern, anchor_mode="semantic"
        ),
    }


def score(
    results: dict[str, dict[str, list[str]]],
    cases: list[EvalCase],
    k: int,
) -> dict[str, dict[str, float]]:
    """Aggregate per-arm metrics across cases."""
    scores: dict[str, dict[str, float]] = {}
    for arm in ARM_NAMES:
        root_hits, chain_recalls, mrrs, ndcgs = [], [], [], []
        for case in cases:
            retrieved = results[case.session][arm]
            chain = set(case.chain_ids)
            # Graded relevance: root cause worth most, rest of chain less
            relevance = {cid: 1.0 for cid in case.chain_ids}
            relevance[case.root_cause_id] = 2.0
            root_hits.append(metrics.hit_at_k(retrieved, case.root_cause_id, k))
            chain_recalls.append(metrics.recall_at_k(retrieved, chain, k))
            mrrs.append(metrics.mrr(retrieved, case.root_cause_id))
            ndcgs.append(metrics.ndcg_at_k(retrieved, relevance, k))
        scores[arm] = {
            f"root_cause_recall@{k}": metrics.mean(root_hits),
            f"chain_recall@{k}": metrics.mean(chain_recalls),
            "mrr": metrics.mean(mrrs),
            f"ndcg@{k}": metrics.mean(ndcgs),
        }
    return scores


def render_table(scores: dict[str, dict[str, float]]) -> str:
    metric_names = list(next(iter(scores.values())).keys())
    header = "| arm | " + " | ".join(metric_names) + " |"
    sep = "|" + "---|" * (len(metric_names) + 1)
    rows = [
        f"| {arm} | " + " | ".join(f"{scores[arm][m]:.3f}" for m in metric_names) + " |"
        for arm in scores
    ]
    return "\n".join([header, sep, *rows])


async def run_judge(
    results: dict[str, dict[str, list[str]]],
    cases: list[EvalCase],
    sessions: list[Session],
    judge_model: str,
) -> dict[str, float]:
    """End-to-end accuracy: judge names the root cause from retrieved context."""
    from .judge import judge_case, make_client

    client = make_client()
    by_id: dict[str, Decision] = {d.id: d for session in sessions for d in session.decisions}

    accuracy: dict[str, float] = {}
    for arm in ARM_NAMES:
        correct = []
        for case in cases:
            retrieved = [by_id[i] for i in results[case.session][arm] if i in by_id]
            correct.append(
                1.0
                if await asyncio.to_thread(judge_case, client, case, retrieved, judge_model)
                else 0.0
            )
        accuracy[arm] = metrics.mean(correct)
    return accuracy


async def main_async(args: argparse.Namespace) -> dict:
    database_url = (
        args.database_url
        or os.getenv("EVAL_DATABASE_URL")
        or os.getenv("TEST_DATABASE_URL")
        or settings.database_url
    )
    settings.database_url = database_url
    logger.info(f"Database: {database_url.rsplit('@', 1)[-1]}")

    # Exact vector search for the eval. The schema no longer ships an ANN
    # index (see migrations/001), but if one exists in the target database
    # (e.g. created from an older migration), probing every list keeps the
    # scan exhaustive so eval numbers measure retrieval, not index recall.
    _orig_init = db._init_connection

    async def _eval_init(conn):
        await _orig_init(conn)
        await conn.execute("SET ivfflat.probes = 100")

    db._init_connection = _eval_init

    provider = make_provider(args.provider)
    if args.provider == "mock":
        logger.warning(
            "Mock embeddings carry no semantic signal — semantic/hybrid/"
            "anchor_causal numbers are meaningless in this mode; use it only "
            "to validate plumbing and the temporal/causal arms."
        )

    sessions, cases = generate_dataset(
        num_sessions=args.sessions,
        distractors_per_session=args.distractors,
        seed=args.seed,
    )

    try:
        await clean_eval_data()
        offsets = await seed_dataset(sessions, provider)

        results = {
            case.session: await run_arms(case, provider, offsets, args.k, args.scope)
            for case in cases
        }
        scores = score(results, cases, args.k)

        report: dict = {
            "config": {
                "provider": args.provider,
                "sessions": args.sessions,
                "distractors": args.distractors,
                "k": args.k,
                "seed": args.seed,
                "scope": args.scope,
            },
            "scores": scores,
        }

        if args.judge:
            report["judge"] = {
                "model": args.judge_model,
                "root_cause_accuracy": await run_judge(results, cases, sessions, args.judge_model),
            }

        print()
        print(render_table(scores))
        if "judge" in report:
            print()
            print(f"Judge ({args.judge_model}) root-cause accuracy:")
            for arm, acc in report["judge"]["root_cause_accuracy"].items():
                print(f"  {arm}: {acc:.3f}")

        if args.output:
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Wrote {args.output}")

        return report
    finally:
        await db.close_pool()


def main() -> None:
    parser = argparse.ArgumentParser(description="Hippocampus retrieval eval")
    parser.add_argument(
        "--provider",
        choices=["mock", "local", "fastembed", "openai"],
        default="mock",
        help="Embedding provider (mock validates plumbing only; "
        "fastembed = ONNX MiniLM, no torch/HF needed)",
    )
    parser.add_argument("--sessions", type=int, default=20)
    parser.add_argument("--distractors", type=int, default=6)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--scope",
        choices=["session", "global"],
        default="session",
        help="Restrict retrieval to the incident's session (realistic) or search all topics",
    )
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--output", default=None, help="Write JSON report here")
    parser.add_argument("--judge", action="store_true", help="Run the LLM judge")
    parser.add_argument(
        "--judge-model",
        default="claude-opus-4-8",
        help="Judge model (default: claude-opus-4-8; use claude-haiku-4-5 for cheap bulk runs)",
    )
    args = parser.parse_args()

    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
