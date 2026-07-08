"""Model package — four-tower additive recommendation architecture.

Public API (via two_tower.py):
    TwoTowerModel   — the assembled model
    build_model     — factory: group names → ready-to-train model
    RESTAURANT_ALL_GROUPS, USER_ALL_GROUPS — valid group name sets
    TEMPORAL_DIM, PREFERENCE_DIM — constants

Internal modules (not imported by scripts directly):
    tower.py            — Tower base class, FeatureGroup base class
    restaurant_tower.py — RestaurantContentTower, RestaurantContextTower,
                          feature groups, factory functions (Pair B: Ben & Danny)
    user_tower.py       — UserContentTower, UserContextTower,
                          feature groups, factory functions (Pair A: Rohith & Antonio)
"""

from src.models.two_tower import (
    TwoTowerModel,
    build_model,
    TEMPORAL_DIM,
    PREFERENCE_DIM,
    RESTAURANT_ALL_GROUPS,
    USER_ALL_GROUPS,
)