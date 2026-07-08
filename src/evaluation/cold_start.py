"""Cold start simulation protocols for user and restaurant evaluation.

User Cold Start (Antonio):
    Take users with 10+ reviews, mask all history, retain only inferred
    cuisine preferences. Simulate context from their review patterns.
    Evaluate if recommendations match what they actually rated highly.

Restaurant Cold Start (Danny):
    Take restaurants with 20+ reviews, mask all reviews. Check if the
    masked restaurant appears in recommendations for compatible users.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.features import extract_cuisine_preferences, compute_user_centroids


@dataclass
class ColdStartUser:
    """A simulated cold start user with masked history."""
    user_id: str
    cuisine_preferences: list[str]    # Top-k cuisines (from masked reviews)
    latitude: float                    # Centroid of their review locations
    longitude: float
    relevant_business_ids: set[str]   # Ground truth: businesses they rated 4+


@dataclass
class ColdStartRestaurant:
    """A simulated cold start restaurant with masked reviews."""
    business_id: str
    categories: list[str]
    price_tier: int
    attributes: dict
    latitude: float
    longitude: float
    compatible_user_ids: set[str]     # Users who rated this 4+ (ground truth)


class UserColdStartSimulator:
    """Simulate new user cold start by masking power user histories.

    Protocol:
    1. Select users with >= min_reviews reviews
    2. For each user: extract cuisine preferences from their reviews
    3. Mask all reviews (model only sees preferences + context)
    4. Ground truth: businesses the user rated >= rating_threshold
    """

    def __init__(
        self,
        reviews: pd.DataFrame,
        businesses: pd.DataFrame,
        min_reviews: int = 10,
        rating_threshold: float = 4.0,
        top_k_cuisines: int = 3,
        n_users: int = 1000,
        seed: int = 42,
    ):
        self.reviews = reviews
        self.businesses = businesses
        self.min_reviews = min_reviews
        self.rating_threshold = rating_threshold
        self.top_k_cuisines = top_k_cuisines
        self.n_users = n_users
        self.rng = np.random.default_rng(seed)

    def generate(self) -> list[ColdStartUser]:
        """Generate cold start user simulations."""
        # Find eligible users
        user_counts = self.reviews.groupby("user_id").size()
        eligible = user_counts[user_counts >= self.min_reviews].index.tolist()

        # Sample
        selected = self.rng.choice(eligible, size=min(self.n_users, len(eligible)), replace=False)

        # Compute centroids
        selected_reviews = self.reviews[self.reviews["user_id"].isin(selected)]
        centroids = compute_user_centroids(selected_reviews, self.businesses)

        cold_start_users = []
        for uid in selected:
            # Extract cuisine preferences
            cuisines = extract_cuisine_preferences(
                uid, self.reviews, self.businesses, self.top_k_cuisines
            )
            if not cuisines:
                continue

            # Ground truth: highly rated businesses
            user_reviews = self.reviews[self.reviews["user_id"] == uid]
            relevant = set(
                user_reviews[user_reviews["stars"] >= self.rating_threshold]["business_id"]
            )
            if not relevant:
                continue

            centroid = centroids.get(uid, (0.0, 0.0))
            cold_start_users.append(ColdStartUser(
                user_id=uid,
                cuisine_preferences=cuisines,
                latitude=centroid[0],
                longitude=centroid[1],
                relevant_business_ids=relevant,
            ))

        return cold_start_users


class RestaurantColdStartSimulator:
    """Simulate new restaurant cold start by masking restaurant reviews.

    Protocol:
    1. Select restaurants with >= min_reviews reviews
    2. Mask all their reviews
    3. Ground truth: users who rated the restaurant >= rating_threshold
    """

    def __init__(
        self,
        reviews: pd.DataFrame,
        businesses: pd.DataFrame,
        min_reviews: int = 20,
        rating_threshold: float = 4.0,
        n_restaurants: int = 500,
        seed: int = 42,
    ):
        self.reviews = reviews
        self.businesses = businesses
        self.min_reviews = min_reviews
        self.rating_threshold = rating_threshold
        self.n_restaurants = n_restaurants
        self.rng = np.random.default_rng(seed)

    def generate(self) -> list[ColdStartRestaurant]:
        """Generate cold start restaurant simulations."""
        # Find eligible restaurants
        biz_counts = self.reviews.groupby("business_id").size()
        eligible = biz_counts[biz_counts >= self.min_reviews].index.tolist()

        selected = self.rng.choice(
            eligible, size=min(self.n_restaurants, len(eligible)), replace=False
        )

        cold_start_restaurants = []
        for bid in selected:
            biz = self.businesses[self.businesses["business_id"] == bid]
            if biz.empty:
                continue
            biz = biz.iloc[0]

            # Ground truth: users who rated highly
            biz_reviews = self.reviews[self.reviews["business_id"] == bid]
            compatible = set(
                biz_reviews[biz_reviews["stars"] >= self.rating_threshold]["user_id"]
            )

            cold_start_restaurants.append(ColdStartRestaurant(
                business_id=bid,
                categories=biz.get("categories_list", []),
                price_tier=biz.get("price_tier", 2),
                attributes=biz.get("attributes", {}),
                latitude=biz.get("latitude", 0.0),
                longitude=biz.get("longitude", 0.0),
                compatible_user_ids=compatible,
            ))

        return cold_start_restaurants
