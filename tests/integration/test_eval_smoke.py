"""Smoke test for the eval harness against the database.

Runs the full pipeline (generate -> seed -> arms -> score) with mock
embeddings. Mock vectors carry no semantic signal, so semantic-arm numbers are
not asserted — what this proves is that the harness runs end to end and that
the causal machinery is exact: with declared edges, the causal arm must
recover the entire chain every time.
"""

import pytest

from evals.dataset import generate_dataset
from evals.run import run_arms, score
from evals.seed import seed_dataset


@pytest.mark.integration
class TestEvalSmoke:
    async def test_end_to_end_with_mock_embeddings(self, patched_db_pool, embedding_provider):
        k = 5
        sessions, cases = generate_dataset(num_sessions=4, distractors_per_session=4, seed=42)

        offsets = await seed_dataset(sessions, embedding_provider)
        assert all(case.failure_id in offsets for case in cases)

        results = {
            case.session: await run_arms(case, embedding_provider, offsets, k) for case in cases
        }
        scores = score(results, cases, k)

        # Causal edges are exact: given the true anchor, the walk must
        # recover the root cause and the full chain, distractors excluded.
        assert scores["causal"][f"root_cause_recall@{k}"] == 1.0
        assert scores["causal"][f"chain_recall@{k}"] == 1.0
        assert scores["causal"]["mrr"] > 0.0

        # Temporal window always contains the failure's neighborhood; the
        # chain precedes the failure so some of it must appear.
        assert scores["temporal"][f"chain_recall@{k}"] > 0.0

        # Plumbing: every arm returned rankings for every case.
        for case in cases:
            for arm_name in ("semantic", "hybrid", "temporal", "causal"):
                assert isinstance(results[case.session][arm_name], list)

    async def test_seed_is_idempotent(self, patched_db_pool, embedding_provider):
        sessions, _ = generate_dataset(num_sessions=2, seed=7)
        first = await seed_dataset(sessions, embedding_provider)
        second = await seed_dataset(sessions, embedding_provider)

        # Second run inserts nothing new (ON CONFLICT DO NOTHING), so the
        # offsets resolved on the first pass are not duplicated.
        assert set(second.keys()) <= set(first.keys())
