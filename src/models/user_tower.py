"""User tower subclasses, feature groups, and factory functions.

Owners: Rohith & Antonio (Pair A)

Two sub-towers on the user side:

    UserContentTower (slow-changing):
        Cuisine/food-type preferences and onboarding selections.
        WHO the user is. Cold users → zero input → neutral embedding.

    UserContextTower (changes every query):
        Day-of-week, distance to candidate.
        The user's CURRENT SITUATION.
"""

import torch
import torch.nn as nn

from src.data.features import PREFERENCE_FEATURES
from src.models.tower import Tower, FeatureGroup


# ── Constants ────────────────────────────────────────────────────────────────

PREFERENCE_DIM = len(PREFERENCE_FEATURES)


# ── Context feature groups ───────────────────────────────────────────────────


class DayOfWeekGroup(FeatureGroup):
    """Cyclical day-of-week encoding (sin, cos, is_weekend)."""

    def __init__(self):
        super().__init__()
        self.output_dim = 3

    def feature_keys(self) -> set[str]:
        return {"dow_sin", "dow_cos", "is_weekend"}

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.stack(
            [features["dow_sin"], features["dow_cos"], features["is_weekend"]],
            dim=-1,
        )


class DistanceGroup(FeatureGroup):
    """Distance to candidate restaurant → projected embedding.

    Gets its own projection so the model can learn a non-linear transform
    (e.g. 2km vs 5km matters more than 50km vs 53km).
    """

    def __init__(self, proj_dim: int = 8):
        super().__init__()
        self.output_dim = proj_dim
        self.proj = nn.Linear(1, proj_dim)

    def feature_keys(self) -> set[str]:
        return {"distance"}

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.proj(features["distance"].unsqueeze(-1))

class UserCheckinGroup(FeatureGroup):
    """Checkin-matched time-of-visit: hour and day-of-week sin/cos.

    Provides finer-grained temporal context than DayOfWeekGroup for reviews
    that were matched to a checkin timestamp. Unmatched reviews get zeros
    — the model learns to fall back on DayOfWeekGroup for those samples.

    No projection needed: 4 raw cyclical features are low-dimensional
    enough to feed directly into the tower MLP.
    """

    def __init__(self):
        super().__init__()
        self.output_dim = 4

    def feature_keys(self) -> set[str]:
        return {"checkin_hour_sin", "checkin_hour_cos",
                "checkin_dow_sin", "checkin_dow_cos"}

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.stack([
            features["checkin_hour_sin"],
            features["checkin_hour_cos"],
            features["checkin_dow_sin"],
            features["checkin_dow_cos"],
        ], dim=-1)

# ── Content feature groups ───────────────────────────────────────────────────


class PreferenceGroup(FeatureGroup):
    """User cuisine/food-type preference vector → projected embedding.

    Input: preference_vec of shape (batch, PREFERENCE_DIM) — normalized
    visit frequencies over curated cuisine and food-type categories.

    Users without preference data get a zero vector; the projection
    layer learns a neutral embedding for cold start users.
    """

    def __init__(self, input_dim: int = PREFERENCE_DIM, proj_dim: int = 16):
        super().__init__()
        self.output_dim = proj_dim
        self.proj = nn.Linear(input_dim, proj_dim)

    def feature_keys(self) -> set[str]:
        return {"preference_vec"}

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.proj(features["preference_vec"])


class OnboardingGroup(FeatureGroup):
    """User onboarding cuisine selections → projected embedding.

    Input: onboarding_vec of shape (batch, PREFERENCE_DIM) — binary multi-hot
    of 1-5 cuisines the user selected during simulated onboarding.

    Same category space as PreferenceGroup (PREFERENCE_FEATURES) but different
    semantics: binary explicit selections vs. normalized visit frequencies.
    Separate projection lets the model learn distinct weights for each signal.

    In training, onboarding vectors are resampled each epoch from the user's
    review-frequency distribution (Gumbel-max trick), acting as free data
    augmentation. At inference, these come from actual onboarding input.
    """

    def __init__(self, input_dim: int = PREFERENCE_DIM, proj_dim: int = 16):
        super().__init__()
        self.output_dim = proj_dim
        self.proj = nn.Linear(input_dim, proj_dim)

    def feature_keys(self) -> set[str]:
        return {"onboarding_vec"}

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.proj(features["onboarding_vec"])


# ── Tower subclasses ─────────────────────────────────────────────────────────


class UserContentTower(Tower):
    """User content tower: cuisine preferences and onboarding selections.

    WHO the user is. Slow-changing taste features. Cold-start users with
    no preference history produce zero input → neutral embedding.
    """
    pass


class UserContextTower(Tower):
    """User context tower: day-of-week, distance to candidate.

    The user's CURRENT SITUATION. Changes every query.
    """
    pass


# ── Group constants ──────────────────────────────────────────────────────────

USER_CONTEXT_GROUPS = {"day_of_week", "distance", "user_checkin"}
USER_CONTENT_GROUPS = {"preferences", "onboarding"}
USER_ALL_GROUPS = USER_CONTEXT_GROUPS | USER_CONTENT_GROUPS


# ── Factory functions ────────────────────────────────────────────────────────


def build_user_context_groups(
    enabled: set[str],
    distance_proj_dim: int = 8,
) -> list[FeatureGroup]:
    """Build context feature groups for the user context tower."""
    groups = []
    if "day_of_week" in enabled:
        groups.append(DayOfWeekGroup())
    if "distance" in enabled:
        groups.append(DistanceGroup(distance_proj_dim))
    if "user_checkin" in enabled:
        groups.append(UserCheckinGroup())
    return groups


def build_user_content_groups(
    enabled: set[str],
    preference_proj_dim: int = 16,
    onboarding_proj_dim: int = 16,
) -> list[FeatureGroup]:
    """Build content feature groups for the user content tower."""
    groups = []
    if "preferences" in enabled:
        groups.append(PreferenceGroup(proj_dim=preference_proj_dim))
    if "onboarding" in enabled:
        groups.append(OnboardingGroup(proj_dim=onboarding_proj_dim))
    return groups


def build_user_groups(
    enabled: set[str],
    distance_proj_dim: int = 8,
    preference_proj_dim: int = 16,
    onboarding_proj_dim: int = 16,
) -> tuple[list[FeatureGroup], list[FeatureGroup]]:
    """Build both user sub-tower group lists from a flat set of names.

    Returns:
        (context_groups, content_groups) — either may be empty.
    """
    context_groups = build_user_context_groups(
        enabled & USER_CONTEXT_GROUPS,
        distance_proj_dim=distance_proj_dim,
    )
    content_groups = build_user_content_groups(
        enabled & USER_CONTENT_GROUPS,
        preference_proj_dim=preference_proj_dim,
        onboarding_proj_dim=onboarding_proj_dim,
    )
    return context_groups, content_groups