"""Restaurant tower subclasses, feature groups, and factory functions.

Owners: Ben & Danny (Pair B)

Two sub-towers on the restaurant side:

    RestaurantContentTower (static, cacheable forever):
        Categories, price tier, structured attributes.
        What the restaurant IS.

    RestaurantContextTower (static per-restaurant, cacheable):
        Checkin-derived temporal profile (hour distribution, weekend
        ratio, hour entropy). WHEN the restaurant is popular.

    Kept separate so temporal features get dedicated capacity to learn
    temporal archetypes without competing with the dominant content
    signal for shared MLP weights.
"""

import torch
import torch.nn as nn

from src.data.features import ATTR_DIM
from src.models.tower import Tower, FeatureGroup


# ── Constants ────────────────────────────────────────────────────────────────

TEMPORAL_DIM = 31  # 24 hour_dist + 7 dow_dist

# ── Content feature groups ───────────────────────────────────────────────────


class CategoryGroup(FeatureGroup):
    """Multi-hot category IDs → mean-pooled embedding."""

    def __init__(self, num_categories: int, emb_dim: int = 32):
        super().__init__()
        self.output_dim = emb_dim
        self.embedding = nn.Embedding(num_categories, emb_dim, padding_idx=0)

    def feature_keys(self) -> set[str]:
        return {"category_ids"}

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        category_ids = features["category_ids"]
        cat_emb = self.embedding(category_ids)
        cat_mask = (category_ids != 0).float().unsqueeze(-1)
        cat_pooled = (cat_emb * cat_mask).sum(dim=1)
        cat_count = cat_mask.sum(dim=1).clamp(min=1)
        return cat_pooled / cat_count


class PriceGroup(FeatureGroup):
    """Price tier (0-4) → learned embedding."""

    def __init__(self, emb_dim: int = 4):
        super().__init__()
        self.output_dim = emb_dim
        self.embedding = nn.Embedding(5, emb_dim)

    def feature_keys(self) -> set[str]:
        return {"price_tier"}

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.embedding(features["price_tier"])


class AttributeGroup(FeatureGroup):
    """Structured restaurant attributes → learned embedding.

    Input is 32 dims of mixed feature types: binary flags (reservations,
    outdoor seating, etc.), one-hot categoricals (WiFi, alcohol, noise,
    attire), and multi-hot nested features (GoodForMeal, Ambience).

    A two-layer MLP lets the group learn conjunctive patterns like
    "quiet + dressy + full_bar → upscale" that a linear layer cannot
    represent. The first layer preserves full dimensionality while ReLU
    enables pattern-specific activation. The second layer compresses to
    the output dim with no activation — the tower MLP provides the next
    non-linearity.
    """

    def __init__(self, input_dim: int = ATTR_DIM, hidden_dim: int = 32, proj_dim: int = 16):
        super().__init__()
        self.output_dim = proj_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def feature_keys(self) -> set[str]:
        return {"attr_vec"}

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.net(features["attr_vec"])


# ── Context feature groups ───────────────────────────────────────────────────


class TemporalGroup(FeatureGroup):
    """Restaurant's checkin-derived temporal profile.

    Input is 26 dims: 24-bin hour distribution, weekend_ratio, and hour
    entropy. The hourly distribution has spatial structure (adjacent hours
    are correlated) and meaningful multi-modal patterns (lunch + dinner
    peaks look fundamentally different from dinner-only but have similar
    means under a linear projection).

    A two-layer MLP lets neurons specialize as "temporal archetype
    detectors" — one fires for late-night weekend spots, another for
    weekday lunch rush, etc. The first layer provides moderate compression
    while ReLU enables pattern-specific activation. The second layer
    compresses to the output dim with no activation.
    """

    def __init__(self, input_dim: int = TEMPORAL_DIM, hidden_dim: int = 32, proj_dim: int = 16):
        super().__init__()
        self.output_dim = proj_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def feature_keys(self) -> set[str]:
        return {"temporal_vec"}

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.net(features["temporal_vec"])


# ── Tower subclasses ─────────────────────────────────────────────────────────


class RestaurantContentTower(Tower):
    """Restaurant content tower: categories, price, attributes.

    What the restaurant IS. Static features — output can be computed
    once per restaurant and cached indefinitely.
    """
    pass


class RestaurantContextTower(Tower):
    """Restaurant context tower: checkin-derived temporal profile.

    WHEN the restaurant is popular. Static per-restaurant (derived from
    historical checkin data), but kept separate from content so it gets
    dedicated capacity to learn temporal archetypes without competing
    with the dominant content signal.
    """
    pass


# ── Group constants ──────────────────────────────────────────────────────────

RESTAURANT_CONTENT_GROUPS = {"categories", "price", "attributes"}
RESTAURANT_CONTEXT_GROUPS = {"temporal"}
RESTAURANT_ALL_GROUPS = RESTAURANT_CONTENT_GROUPS | RESTAURANT_CONTEXT_GROUPS


# ── Factory functions ────────────────────────────────────────────────────────


def build_restaurant_content_groups(
    enabled: set[str],
    num_categories: int = 0,
    category_emb_dim: int = 32,
    price_emb_dim: int = 4,
    attribute_proj_dim: int = 16,
) -> list[FeatureGroup]:
    """Build content feature groups for the restaurant content tower."""
    groups = []
    if "categories" in enabled:
        if num_categories == 0:
            raise ValueError("num_categories required when 'categories' group is enabled")
        groups.append(CategoryGroup(num_categories, category_emb_dim))
    if "price" in enabled:
        groups.append(PriceGroup(price_emb_dim))
    if "attributes" in enabled:
        groups.append(AttributeGroup(proj_dim=attribute_proj_dim))
    return groups


def build_restaurant_context_groups(
    enabled: set[str],
    temporal_proj_dim: int = 16,
) -> list[FeatureGroup]:
    """Build context feature groups for the restaurant context tower."""
    groups = []
    if "temporal" in enabled:
        groups.append(TemporalGroup(proj_dim=temporal_proj_dim))
    return groups


def build_restaurant_groups(
    enabled: set[str],
    num_categories: int = 0,
    category_emb_dim: int = 32,
    price_emb_dim: int = 4,
    attribute_proj_dim: int = 16,
    temporal_proj_dim: int = 16,
) -> tuple[list[FeatureGroup], list[FeatureGroup]]:
    """Build both restaurant sub-tower group lists from a flat set of names.

    Returns:
        (content_groups, context_groups) — context may be empty.
    """
    content_groups = build_restaurant_content_groups(
        enabled & RESTAURANT_CONTENT_GROUPS,
        num_categories=num_categories,
        category_emb_dim=category_emb_dim,
        price_emb_dim=price_emb_dim,
        attribute_proj_dim=attribute_proj_dim,
    )
    context_groups = build_restaurant_context_groups(
        enabled & RESTAURANT_CONTEXT_GROUPS,
        temporal_proj_dim=temporal_proj_dim,
    )
    return content_groups, context_groups