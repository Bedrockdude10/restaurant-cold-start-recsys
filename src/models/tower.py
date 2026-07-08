"""Base tower and feature group abstractions.

Tower: generic concatenate-feature-groups → MLP → embedding module.
FeatureGroup: base class for pluggable feature encoders.

All tower subclasses (RestaurantContentTower, UserContextTower, etc.)
extend Tower. All feature groups (CategoryGroup, TemporalGroup, etc.)
extend FeatureGroup.

Tower has zero conditional logic — ablation happens by construction,
passing different feature group lists to different tower subclasses.
"""

import torch
import torch.nn as nn


class FeatureGroup(nn.Module):
    """Base class for feature group modules.

    Contract:
        - __init__ receives config, sets self.output_dim
        - forward receives the full feature dict, extracts what it needs
        - feature_keys() returns the set of keys it reads from the dict
    """
    output_dim: int

    def feature_keys(self) -> set[str]:
        raise NotImplementedError


class Tower(nn.Module):
    """Generic tower: concatenates feature group outputs → MLP → embedding.

    Ablation by construction: pass a different list of feature groups to
    get a different model variant. The tower itself has zero conditional
    logic.
    """

    def __init__(
        self,
        feature_groups: list[FeatureGroup],
        hidden_dims: list[int] | None = None,
        output_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        if not feature_groups:
            raise ValueError("Tower requires at least one feature group")

        self.feature_groups = nn.ModuleList(feature_groups)

        input_dim = sum(g.output_dim for g in feature_groups)
        hidden_dims = hidden_dims or [128, 64]

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        group_outputs = [group(features) for group in self.feature_groups]
        x = torch.cat(group_outputs, dim=-1)
        return self.mlp(x)

    def expected_feature_keys(self) -> set[str]:
        """All feature keys this tower expects in the input dict."""
        keys: set[str] = set()
        for group in self.feature_groups:
            assert isinstance(group, FeatureGroup)
            keys |= group.feature_keys()
        return keys