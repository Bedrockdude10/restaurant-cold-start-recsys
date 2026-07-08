"""Tests for evaluation metrics."""

import math

import pytest

from src.evaluation.metrics import hit_at_k, ndcg_at_k, evaluate_recommendations


class TestHitAtK:
    def test_hit_found_in_top_k(self):
        ranked = ["a", "b", "c", "d", "e"]
        relevant = {"c"}
        assert hit_at_k(ranked, relevant, k=5) == 1.0

    def test_hit_at_first_position(self):
        ranked = ["a", "b", "c"]
        relevant = {"a"}
        assert hit_at_k(ranked, relevant, k=5) == 1.0

    def test_hit_not_found(self):
        ranked = ["a", "b", "c", "d", "e"]
        relevant = {"f"}
        assert hit_at_k(ranked, relevant, k=5) == 0.0

    def test_hit_outside_k(self):
        ranked = ["a", "b", "c", "d", "e", "f"]
        relevant = {"f"}
        assert hit_at_k(ranked, relevant, k=5) == 0.0

    def test_empty_relevant(self):
        ranked = ["a", "b", "c"]
        relevant = set()
        assert hit_at_k(ranked, relevant, k=5) == 0.0


class TestNDCGAtK:
    def test_perfect_ranking(self):
        ranked = ["a", "b", "c"]
        relevant = {"a"}
        assert ndcg_at_k(ranked, relevant, k=10) == 1.0

    def test_relevant_at_second_position(self):
        ranked = ["x", "a", "b"]
        relevant = {"a"}
        expected = (1.0 / math.log2(3)) / (1.0 / math.log2(2))  # DCG / IDCG
        assert abs(ndcg_at_k(ranked, relevant, k=10) - expected) < 1e-6

    def test_no_relevant_items(self):
        ranked = ["a", "b", "c"]
        relevant = {"d"}
        assert ndcg_at_k(ranked, relevant, k=10) == 0.0

    def test_empty_relevant_set(self):
        ranked = ["a", "b"]
        relevant = set()
        assert ndcg_at_k(ranked, relevant, k=10) == 0.0


class TestEvaluateRecommendations:
    def test_aggregate_metrics(self):
        all_ranked = [["a", "b", "c"], ["d", "e", "f"]]
        all_relevant = [{"a"}, {"f"}]
        results = evaluate_recommendations(all_ranked, all_relevant, hit_k=3, ndcg_k=3)
        assert results["Hit@3"] == 1.0  # Both have a hit in top 3
        assert results["NDCG@3"] > 0
