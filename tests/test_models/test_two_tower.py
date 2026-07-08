"""Tests for two-tower model architecture."""

import torch
import pytest

from src.models.two_tower import UserTower, ItemTower, TwoTowerModel


@pytest.fixture
def user_tower():
    return UserTower(num_cuisines=50, num_regions=20, output_dim=64)


@pytest.fixture
def item_tower():
    return ItemTower(num_categories=100, num_attributes=20, num_regions=20, output_dim=64)


@pytest.fixture
def model(user_tower, item_tower):
    return TwoTowerModel(user_tower, item_tower, similarity="dot_product")


class TestUserTower:
    def test_output_shape(self, user_tower):
        batch_size = 8
        out = user_tower(
            cuisine_ids=torch.randint(0, 50, (batch_size, 5)),
            time_bucket=torch.randint(0, 5, (batch_size,)),
            day_of_week=torch.randint(0, 7, (batch_size,)),
            is_weekend=torch.zeros(batch_size),
            distance=torch.rand(batch_size),
            region_id=torch.randint(0, 20, (batch_size,)),
        )
        assert out.shape == (batch_size, 64)

    def test_handles_all_padding(self, user_tower):
        """User with no cuisine preferences (all padding) should not crash."""
        out = user_tower(
            cuisine_ids=torch.zeros(2, 5, dtype=torch.long),
            time_bucket=torch.zeros(2, dtype=torch.long),
            day_of_week=torch.zeros(2, dtype=torch.long),
            is_weekend=torch.zeros(2),
            distance=torch.zeros(2),
            region_id=torch.zeros(2, dtype=torch.long),
        )
        assert out.shape == (2, 64)


class TestItemTower:
    def test_output_shape(self, item_tower):
        batch_size = 8
        out = item_tower(
            category_ids=torch.randint(0, 100, (batch_size, 10)),
            price_tier=torch.randint(1, 5, (batch_size,)),
            attributes=torch.randint(0, 2, (batch_size, 20)),
            region_id=torch.randint(0, 20, (batch_size,)),
        )
        assert out.shape == (batch_size, 64)


class TestTwoTowerModel:
    def test_forward_returns_scores(self, model):
        batch_size = 4
        user_features = {
            "cuisine_ids": torch.randint(0, 50, (batch_size, 5)),
            "time_bucket": torch.randint(0, 5, (batch_size,)),
            "day_of_week": torch.randint(0, 7, (batch_size,)),
            "is_weekend": torch.zeros(batch_size),
            "distance": torch.rand(batch_size),
            "region_id": torch.randint(0, 20, (batch_size,)),
        }
        item_features = {
            "category_ids": torch.randint(0, 100, (batch_size, 10)),
            "price_tier": torch.randint(1, 5, (batch_size,)),
            "attributes": torch.randint(0, 2, (batch_size, 20)),
            "region_id": torch.randint(0, 20, (batch_size,)),
        }
        scores = model(user_features, item_features)
        assert scores.shape == (batch_size,)

    def test_cosine_similarity_bounded(self, user_tower, item_tower):
        model = TwoTowerModel(user_tower, item_tower, similarity="cosine")
        batch_size = 4
        user_features = {
            "cuisine_ids": torch.randint(0, 50, (batch_size, 5)),
            "time_bucket": torch.randint(0, 5, (batch_size,)),
            "day_of_week": torch.randint(0, 7, (batch_size,)),
            "is_weekend": torch.zeros(batch_size),
            "distance": torch.rand(batch_size),
            "region_id": torch.randint(0, 20, (batch_size,)),
        }
        item_features = {
            "category_ids": torch.randint(0, 100, (batch_size, 10)),
            "price_tier": torch.randint(1, 5, (batch_size,)),
            "attributes": torch.randint(0, 2, (batch_size, 20)),
            "region_id": torch.randint(0, 20, (batch_size,)),
        }
        scores = model(user_features, item_features)
        assert (scores >= -1.0 - 1e-5).all() and (scores <= 1.0 + 1e-5).all()

    def test_encode_user_and_item_separately(self, model):
        """Verify we can precompute item embeddings and score against user embeddings."""
        user_features = {
            "cuisine_ids": torch.randint(0, 50, (2, 5)),
            "time_bucket": torch.randint(0, 5, (2,)),
            "day_of_week": torch.randint(0, 7, (2,)),
            "is_weekend": torch.zeros(2),
            "distance": torch.rand(2),
            "region_id": torch.randint(0, 20, (2,)),
        }
        item_features = {
            "category_ids": torch.randint(0, 100, (2, 10)),
            "price_tier": torch.randint(1, 5, (2,)),
            "attributes": torch.randint(0, 2, (2, 20)),
            "region_id": torch.randint(0, 20, (2,)),
        }
        user_emb = model.encode_user(user_features)
        item_emb = model.encode_item(item_features)
        assert user_emb.shape == (2, 64)
        assert item_emb.shape == (2, 64)
