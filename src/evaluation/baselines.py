"""Baseline recommendation models for comparison.

1. RandomBaseline: Random scores for candidates (uniform shuffle)
2. PopularityBaseline: Score candidates by review count

Both implement `score_candidates(user_context, candidate_ids) -> dict[str, float]`
so they plug into the evaluation harness with the same interface as the two-tower model.

Usage in evaluation:
    baseline = PopularityBaseline(restaurants)
    scores = baseline.score_candidates(user_context, candidate_ids)
    ranked = sorted(candidate_ids, key=lambda bid: scores[bid], reverse=True)
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseRecommender(ABC):
    """Interface that all models/baselines must implement for evaluation."""

    @abstractmethod
    def score_candidates(
        self,
        user_context: dict,
        candidate_ids: list[str],
    ) -> dict[str, float]:
        """Score a set of candidate restaurants for a given user context.

        Args:
            user_context: Dict with user info. Baselines may ignore most of this.
                Expected keys (not all required by all models):
                - cuisine_preferences: list[str]
                - latitude: float
                - longitude: float
                - user_id: str
            candidate_ids: List of business_ids to score.

        Returns:
            Dict mapping business_id -> score (higher = better).
        """
        ...


class RandomBaseline(BaseRecommender):
    """Score candidates randomly (uniform).

    Expected Hit@5 ≈ 5%, Hit@10 ≈ 10% on 100 candidates.
    Serves as the absolute floor — anything worse than this is broken.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def score_candidates(
        self,
        user_context: dict,
        candidate_ids: list[str],
    ) -> dict[str, float]:
        scores = self.rng.random(len(candidate_ids))
        return dict(zip(candidate_ids, scores))


class PopularityBaseline(BaseRecommender):
    """Score candidates by review count (most-reviewed = highest score).

    This is a strong non-personalized baseline: popular restaurants
    accumulate reviews precisely because many people like them. No
    contextual features, no personalization — pure popularity.
    """

    def __init__(self, restaurants: pd.DataFrame):
        """
        Args:
            restaurants: Processed restaurants DataFrame with columns
                [business_id, review_count].
        """
        self.review_counts = restaurants.set_index("business_id")["review_count"].to_dict()

    def score_candidates(
        self,
        user_context: dict,
        candidate_ids: list[str],
    ) -> dict[str, float]:
        return {
            bid: float(self.review_counts.get(bid, 0))
            for bid in candidate_ids
        }