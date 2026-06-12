"""Unit tests for the eval dataset generator and metrics (no database)."""

import pytest

from evals.dataset import generate_dataset
from evals.metrics import hit_at_k, mean, mrr, ndcg_at_k, recall_at_k


@pytest.mark.unit
class TestDatasetGenerator:
    def test_deterministic_for_seed(self):
        a_sessions, a_cases = generate_dataset(num_sessions=6, seed=7)
        b_sessions, b_cases = generate_dataset(num_sessions=6, seed=7)

        assert [s.name for s in a_sessions] == [s.name for s in b_sessions]
        assert [[d.id for d in s.decisions] for s in a_sessions] == [
            [d.id for d in s.decisions] for s in b_sessions
        ]
        assert [c.query for c in a_cases] == [c.query for c in b_cases]

    def test_different_seeds_differ(self):
        a_sessions, _ = generate_dataset(num_sessions=6, seed=1)
        b_sessions, _ = generate_dataset(num_sessions=6, seed=2)
        a_order = [[d.id for d in s.decisions] for s in a_sessions]
        b_order = [[d.id for d in s.decisions] for s in b_sessions]
        assert a_order != b_order

    def test_chain_is_causally_linked_and_ordered(self):
        sessions, cases = generate_dataset(num_sessions=4, seed=42)
        for session, case in zip(sessions, cases):
            by_id = {d.id: d for d in session.decisions}
            # Every chain decision after the root declares its predecessor
            for parent, child in zip(case.chain_ids, case.chain_ids[1:]):
                assert by_id[child].parent_id == parent
            assert by_id[case.root_cause_id].parent_id is None
            # Chain keeps relative order within the session
            positions = [[d.id for d in session.decisions].index(cid) for cid in case.chain_ids]
            assert positions == sorted(positions)

    def test_distractors_present_and_unlinked(self):
        sessions, _ = generate_dataset(num_sessions=4, distractors_per_session=6, seed=42)
        for session in sessions:
            distractors = [d for d in session.decisions if d.role == "distractor"]
            assert len(distractors) >= 2
            assert all(d.parent_id is None for d in distractors)

    def test_payload_shape(self):
        sessions, _ = generate_dataset(num_sessions=1, seed=42)
        for decision in sessions[0].decisions:
            payload = decision.payload()
            assert payload["id"] == decision.id
            assert "role" not in payload  # eval bookkeeping must not leak


@pytest.mark.unit
class TestMetrics:
    def test_hit_at_k(self):
        assert hit_at_k(["a", "b", "c"], "b", 2) == 1.0
        assert hit_at_k(["a", "b", "c"], "c", 2) == 0.0
        assert hit_at_k([], "a", 5) == 0.0

    def test_recall_at_k(self):
        assert recall_at_k(["a", "b", "c"], {"a", "c"}, 3) == 1.0
        assert recall_at_k(["a", "b", "c"], {"a", "z"}, 3) == 0.5
        assert recall_at_k(["a"], set(), 3) == 0.0

    def test_mrr(self):
        assert mrr(["x", "target"], "target") == 0.5
        assert mrr(["target"], "target") == 1.0
        assert mrr(["x", "y"], "target") == 0.0

    def test_ndcg_perfect_ranking(self):
        relevance = {"root": 2.0, "mid": 1.0}
        assert ndcg_at_k(["root", "mid"], relevance, 5) == pytest.approx(1.0)

    def test_ndcg_worse_when_root_ranked_low(self):
        relevance = {"root": 2.0, "mid": 1.0}
        good = ndcg_at_k(["root", "mid", "x"], relevance, 5)
        bad = ndcg_at_k(["x", "mid", "root"], relevance, 5)
        assert good > bad > 0.0

    def test_mean(self):
        assert mean([1.0, 0.0]) == 0.5
        assert mean([]) == 0.0
