"""TwoTowerModel: orchestrates four sub-towers into a single scoring model.

Architecture (v4 — four-tower with fusion):

    Each side has two sub-towers (content + context) whose outputs are
    concatenated and passed through a fusion MLP that learns how content
    and context interact before the final dot-product scoring.

    Restaurant side:
        content_emb    = RestaurantContentTower(features)     → 64 dims
        context_emb    = RestaurantContextTower(features)     → 64 dims
        restaurant_emb = RestaurantFusion(concat(content, context)) → 64 dims

    User side:
        content_emb    = UserContentTower(features)           → 64 dims
        context_emb    = UserContextTower(features)           → 64 dims
        user_emb       = UserFusion(concat(content, context)) → 64 dims

    score = dot(user_emb, restaurant_emb)

    Why fusion over addition:
        The v3 additive combination treated sub-tower outputs as independent
        contributions. This prevented the model from learning that "Italian
        restaurant + weekend brunch pattern" means something different from
        "Italian restaurant + weeknight dinner pattern." The fusion MLP
        receives both embeddings concatenated and can learn these cross-concern
        interactions.

    Cold-start degradation:
        When a sub-tower is None, its slot in the concatenation is filled
        with zeros. The fusion MLP learns to produce a reasonable output
        from partial input — no special-casing needed.

Interface:
    forward() takes a SINGLE flat feature dict containing all keys for all
    towers. The model internally routes each key to the correct sub-tower
    via FeatureGroup.feature_keys(). Dataset construction, training loops,
    and evaluation code never need to know about tower architecture.

Ablation:
    Any sub-tower except restaurant_content_tower can be None. The fusion
    MLP gracefully handles zeros in the missing sub-tower's slot.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.restaurant_tower import (
    TEMPORAL_DIM,
    RestaurantContentTower,
    RestaurantContextTower,
    build_restaurant_groups,
    RESTAURANT_ALL_GROUPS,
)
from src.models.user_tower import (
    PREFERENCE_DIM,
    UserContextTower,
    UserContentTower,
    build_user_groups,
    USER_ALL_GROUPS,
)


# ── Fusion MLP ───────────────────────────────────────────────────────────────


class FusionMLP(nn.Module):
    """Fuses content and context embeddings into a single representation.

    Takes the concatenation of two sub-tower outputs (or one real + one
    zero-filled) and learns cross-concern interactions via a small MLP.

    Input:  concat(content_emb, context_emb) → 2 * input_dim
    Output: fused embedding → output_dim
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 64,
        output_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        content_emb: torch.Tensor,
        context_emb: torch.Tensor,
    ) -> torch.Tensor:
        return self.mlp(torch.cat([content_emb, context_emb], dim=-1))


class GatedFusionMLP(nn.Module):
    """Content-gated fusion: the content embedding controls how context is mixed in.

    Instead of a single global MLP that learns one interaction pattern for
    all items (which memorizes training-restaurant-specific combinations),
    this module lets each item's content determine its own fusion behavior.

    Mechanism:
        1. Content embedding produces per-dimension gate weights via sigmoid.
           These gates answer: "for THIS type of restaurant, how much should
           each dimension of the temporal signal matter?"
        2. Context embedding is element-wise gated before fusion.
        3. The gated context + original content are concatenated and projected.

    Cold-start behavior:
        When content_emb is zeros (missing content tower), gates are sigmoid(0)
        = 0.5, giving context a neutral pass-through — no memorized pattern.
        When context_emb is zeros (missing context tower / cold restaurant),
        the gate has no effect — output depends only on content, same as
        FusionMLP with zero context input.

    Inspired by EmerG (KDD 2024), which uses hypernetworks to generate
    item-specific feature interaction graphs. This is a lightweight
    approximation: instead of a full GNN over a learned adjacency matrix,
    we use content-conditioned gating on the context signal.
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 64,
        output_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        # Content → gate weights for context dimensions
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid(),
        )
        # Fuse gated context + content → output
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        content_emb: torch.Tensor,
        context_emb: torch.Tensor,
    ) -> torch.Tensor:
        gate_weights = self.gate(content_emb)           # (batch, dim)
        gated_context = context_emb * gate_weights       # (batch, dim)
        return self.mlp(torch.cat([content_emb, gated_context], dim=-1))


# ── Two-Tower Model ──────────────────────────────────────────────────────────


class TwoTowerModel(nn.Module):
    """Four-tower model with fusion MLPs for cross-concern interaction.

    forward() takes a SINGLE flat feature dict. The model routes keys
    to the correct sub-towers internally.

    Any sub-tower can be None. Missing sub-towers contribute a zero
    vector to the fusion MLP, which learns to handle partial input
    gracefully. At least one sub-tower per side is required.
    """

    def __init__(
        self,
        restaurant_fusion: nn.Module,
        user_fusion: nn.Module,
        restaurant_content_tower: RestaurantContentTower | None = None,
        restaurant_context_tower: RestaurantContextTower | None = None,
        user_context_tower: UserContextTower | None = None,
        user_content_tower: UserContentTower | None = None,
        similarity: str = "dot_product",
        output_dim: int = 64,
    ):
        super().__init__()
        self.restaurant_content_tower = restaurant_content_tower
        self.restaurant_context_tower = restaurant_context_tower
        self.user_context_tower = user_context_tower
        self.user_content_tower = user_content_tower
        self.restaurant_fusion = restaurant_fusion
        self.user_fusion = user_fusion
        self.similarity = similarity

        # Cache the output dim for zero-filling missing sub-towers
        self._tower_dim = output_dim

    def _fuse_side(
        self,
        content_tower,
        context_tower,
        fusion: nn.Module,
        features: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Run both sub-towers and fuse. Missing towers get zeros."""
        # We need a batch size reference from whichever tower exists
        if content_tower is not None:
            content_emb = content_tower(features)
            batch_size = content_emb.shape[0]
        else:
            content_emb = None
            batch_size = None

        if context_tower is not None:
            context_emb = context_tower(features)
            if batch_size is None:
                batch_size = context_emb.shape[0]
        else:
            context_emb = None

        assert batch_size is not None, "At least one sub-tower is required per side"
        device = next(fusion.parameters()).device

        if content_emb is None:
            content_emb = torch.zeros(batch_size, self._tower_dim, device=device)
        if context_emb is None:
            context_emb = torch.zeros(batch_size, self._tower_dim, device=device)

        return fusion(content_emb, context_emb)

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Score from a single flat feature dict.

        During training, applies input-level temporal dropout: zeros the
        raw temporal_vec for a fraction of samples so the full pipeline
        (TemporalGroup MLP → RestaurantContextTower → FusionMLP) learns
        to produce reasonable restaurant embeddings without checkin data.
        This directly simulates the cold-restaurant scenario where a new
        restaurant has metadata but no user activity.
        """
        if self.training and "temporal_vec" in features:
            b = features["temporal_vec"].shape[0]
            device = features["temporal_vec"].device
            keep_mask = (torch.rand(b, 1, device=device) > 0.2).float()
            features = {**features, "temporal_vec": features["temporal_vec"] * keep_mask}

        rest_emb = self._fuse_side(
            self.restaurant_content_tower,
            self.restaurant_context_tower,
            self.restaurant_fusion,
            features,
        )
        user_emb = self._fuse_side(
            self.user_content_tower,
            self.user_context_tower,
            self.user_fusion,
            features,
        )

        if self.similarity == "cosine":
            user_emb = F.normalize(user_emb, dim=-1)
            rest_emb = F.normalize(rest_emb, dim=-1)

        return (user_emb * rest_emb).sum(dim=-1)

    def encode_restaurant(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode restaurant features only (for precomputing and caching)."""
        return self._fuse_side(
            self.restaurant_content_tower,
            self.restaurant_context_tower,
            self.restaurant_fusion,
            features,
        )

    def encode_user(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode user features only."""
        return self._fuse_side(
            self.user_content_tower,
            self.user_context_tower,
            self.user_fusion,
            features,
        )

    def expected_feature_keys(self) -> set[str]:
        """All feature keys expected in the input dict (union of all towers)."""
        keys: set[str] = set()
        if self.restaurant_content_tower is not None:
            keys |= self.restaurant_content_tower.expected_feature_keys()
        if self.restaurant_context_tower is not None:
            keys |= self.restaurant_context_tower.expected_feature_keys()
        if self.user_context_tower is not None:
            keys |= self.user_context_tower.expected_feature_keys()
        if self.user_content_tower is not None:
            keys |= self.user_content_tower.expected_feature_keys()
        return keys


# ── Factory ──────────────────────────────────────────────────────────────────


def build_model(
    restaurant_group_names: set[str],
    user_group_names: set[str],
    num_categories: int,
    output_dim: int = 64,
    similarity: str = "dot_product",
    fusion: str = "mlp",
) -> TwoTowerModel:
    """Build a fully assembled TwoTowerModel from flat group name sets.

    This is the single public entry point for model construction. Callers
    never need to import tower subclasses or factory functions from the
    individual tower modules.

    Args:
        restaurant_group_names: Enabled restaurant groups (from RESTAURANT_ALL_GROUPS).
        user_group_names: Enabled user groups (from USER_ALL_GROUPS).
        num_categories: Size of category vocabulary.
        output_dim: Embedding dimension for all sub-towers.
        similarity: "dot_product" or "cosine".
        fusion: "mlp" for standard FusionMLP, "gated" for GatedFusionMLP.

    Returns:
        Fully assembled TwoTowerModel ready for .to(device).
    """
    rest_content_groups, rest_context_groups = build_restaurant_groups(
        enabled=restaurant_group_names,
        num_categories=num_categories,
    )
    user_context_groups, user_content_groups = build_user_groups(
        enabled=user_group_names,
    )

    if not rest_content_groups and not rest_context_groups:
        raise ValueError("Restaurant side requires at least one feature group")

    restaurant_content_tower = (
        RestaurantContentTower(rest_content_groups, output_dim=output_dim)
        if rest_content_groups else None
    )
    restaurant_context_tower = (
        RestaurantContextTower(rest_context_groups, output_dim=output_dim)
        if rest_context_groups else None
    )
    user_context_tower = (
        UserContextTower(user_context_groups, hidden_dims=[64, 64], output_dim=output_dim)
        if user_context_groups else None
    )
    user_content_tower = (
        UserContentTower(user_content_groups, hidden_dims=[64, 64], output_dim=output_dim)
        if user_content_groups else None
    )

    FusionClass = GatedFusionMLP if fusion == "gated" else FusionMLP

    restaurant_fusion = FusionClass(
        input_dim=output_dim, hidden_dim=output_dim, output_dim=output_dim,
    )
    user_fusion = FusionClass(
        input_dim=output_dim, hidden_dim=output_dim, output_dim=output_dim,
    )

    return TwoTowerModel(
        restaurant_fusion=restaurant_fusion,
        user_fusion=user_fusion,
        restaurant_content_tower=restaurant_content_tower,
        restaurant_context_tower=restaurant_context_tower,
        user_context_tower=user_context_tower,
        user_content_tower=user_content_tower,
        similarity=similarity,
        output_dim=output_dim,
    )