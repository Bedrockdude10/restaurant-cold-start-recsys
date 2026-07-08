"""Tests for data processing and feature extraction."""

import pytest

from src.utils.geo import haversine_distance
from src.data.features import hour_to_time_bucket, encode_time_features


class TestHaversineDistance:
    def test_same_point_is_zero(self):
        assert haversine_distance(42.3601, -71.0589, 42.3601, -71.0589) == 0.0

    def test_boston_to_nyc(self):
        # ~306 km
        dist = haversine_distance(42.3601, -71.0589, 40.7128, -74.0060)
        assert 300 < dist < 320

    def test_symmetric(self):
        d1 = haversine_distance(42.36, -71.06, 40.71, -74.01)
        d2 = haversine_distance(40.71, -74.01, 42.36, -71.06)
        assert abs(d1 - d2) < 1e-6


class TestTimeBuckets:
    def test_morning(self):
        assert hour_to_time_bucket(8) == 0

    def test_lunch(self):
        assert hour_to_time_bucket(12) == 1

    def test_afternoon(self):
        assert hour_to_time_bucket(15) == 2

    def test_dinner(self):
        assert hour_to_time_bucket(19) == 3

    def test_late_night(self):
        assert hour_to_time_bucket(23) == 4
        assert hour_to_time_bucket(2) == 4

    def test_encode_time_features(self):
        features = encode_time_features(hour=19, day_of_week=5)
        assert features["time_bucket"] == 3   # dinner
        assert features["day_of_week"] == 5   # Saturday
        assert features["is_weekend"] is True
