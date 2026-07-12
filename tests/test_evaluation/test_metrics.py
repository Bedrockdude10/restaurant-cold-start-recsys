"""Tests for evaluation metrics."""

import math

import pytest

from src.evaluation.metrics import (
    evaluate_recommendations,
    hit_at_k,
    ndcg_at_k,
    reciprocal_rank,
)


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


class TestReciprocalRank:
    def test_first_position(self):
        assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0

    def test_third_position(self):
        assert reciprocal_rank(["a", "b", "c"], {"c"}) == pytest.approx(1.0 / 3.0)

    def test_no_relevant(self):
        assert reciprocal_rank(["a", "b", "c"], {"z"}) == 0.0

    def test_uses_first_relevant(self):
        assert reciprocal_rank(["a", "b", "c"], {"b", "c"}) == pytest.approx(0.5)


class TestEvaluateRecommendations:
    def test_aggregate_metrics(self):
        all_ranked = [["a", "b", "c"], ["d", "e", "f"]]
        all_relevant = [{"a"}, {"f"}]
        results = evaluate_recommendations(all_ranked, all_relevant, hit_k=3, ndcg_k=3)
        assert results["Hit@3"] == 1.0  # Both have a hit in top 3
        assert results["NDCG@3"] > 0
        assert results["n_cases"] == 2

    def test_metric_suite_over_multiple_cutoffs(self):
        # positive at rank 1 (case 1) and rank 6 (case 2, outside k=5)
        all_ranked = [["a", "x", "y", "z", "w", "v"], ["x", "y", "z", "w", "v", "b"]]
        all_relevant = [{"a"}, {"b"}]
        r = evaluate_recommendations(
            all_ranked, all_relevant, hit_ks=[5, 10], ndcg_ks=[5, 10],
        )
        # legacy scalar keys always present, plus the requested cutoffs + MRR
        for key in ("Hit@5", "Hit@10", "NDCG@5", "NDCG@10", "MRR", "n_cases"):
            assert key in r
        assert r["Hit@5"] == 0.5   # only case 1 hits within top 5
        assert r["Hit@10"] == 1.0  # both hit within top 10
        assert r["MRR"] == pytest.approx((1.0 + 1.0 / 6.0) / 2.0)

    def test_empty_input(self):
        r = evaluate_recommendations([], [], hit_ks=[5, 10], ndcg_ks=[10])
        assert r["n_cases"] == 0
        assert r["Hit@5"] == 0.0 and r["MRR"] == 0.0
