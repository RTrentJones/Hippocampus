"""Unit tests for consumer logic that needs no database or Kafka."""

import pytest

from hippocampus.consumer import extract_cause_ids


@pytest.mark.unit
class TestExtractCauseIds:
    """Tests for causal parent extraction from decision payloads."""

    def test_no_causes(self):
        assert extract_cause_ids({"context": "x", "action": "y"}) == []

    def test_parent_id(self):
        assert extract_cause_ids({"parent_id": "abc"}) == ["abc"]

    def test_caused_by_string(self):
        assert extract_cause_ids({"caused_by": "abc"}) == ["abc"]

    def test_caused_by_list(self):
        assert extract_cause_ids({"caused_by": ["a", "b"]}) == ["a", "b"]

    def test_parent_id_and_caused_by(self):
        assert extract_cause_ids({"parent_id": "a", "caused_by": ["b", "c"]}) == ["a", "b", "c"]

    def test_duplicates_removed_order_preserved(self):
        assert extract_cause_ids({"parent_id": "a", "caused_by": ["b", "a"]}) == ["a", "b"]

    def test_non_string_values_ignored(self):
        assert extract_cause_ids({"parent_id": 42, "caused_by": [None, 1, "ok"]}) == ["ok"]

    def test_empty_strings_ignored(self):
        assert extract_cause_ids({"parent_id": "", "caused_by": ["", "x"]}) == ["x"]

    def test_caused_by_null(self):
        assert extract_cause_ids({"parent_id": None, "caused_by": None}) == []
